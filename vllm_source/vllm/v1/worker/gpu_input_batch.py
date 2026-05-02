# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Datastructures defining a GPU input batch

from dataclasses import dataclass
from typing import cast

import numpy as np
import torch

from vllm.lora.request import LoRARequest
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.utils.collection_utils import swap_dict_values
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.pool.metadata import PoolingMetadata, PoolingStates
from vllm.v1.sample.logits_processor import (
    BatchUpdateBuilder,
    LogitsProcessors,
    MoveDirectionality,
)
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.utils import is_spec_decode_unsupported
from vllm.v1.utils import copy_slice
from vllm.v1.worker.block_table import MultiGroupBlockTable


@dataclass
class CachedRequestState:
    """ 
    用于缓存单个请求(Request)核心状态的轻量级数据类
    主要作用是：
        - 在scheduler中快速存储和管理一个请求从prefill到decode阶段所需的关键信息
        - 避免频繁访问完整的Request对象,提高性能
        - 支持连续批处理和async scheduling
    """
    req_id: str
    prompt_token_ids: list[int] | None                  #输入 prompt 的 token ids(prefill 阶段使用)
    mm_features: list[MultiModalFeatureSpec]            #多模态特征
    sampling_params: SamplingParams | None              #采样参数等
    generator: torch.Generator | None                   #随机数生成器 当设置了seed时使用,用于可复现采样

    block_ids: tuple[list[int], ...]                    #该请求占用的PagedAttention block id列表
    num_computed_tokens: int                            ## 当前已经计算完成的 token 数量(用于判断是否需要继续 prefill)
    output_token_ids: list[int]                         #已生成的输出 token ids(decode 阶段不断追加)

    mrope_positions: torch.Tensor | None = None
    mrope_position_delta: int | None = None

    xdrope_positions: torch.Tensor | None = None

    lora_request: LoRARequest | None = None             ## LoRA 适配器请求(如果使用了 LoRA 微调)
    prompt_embeds: torch.Tensor | None = None           # 直接传入的 prompt embeddings(绕过 tokenizer 时使用)

    # Used when both async_scheduling and spec_decode are enabled.
    prev_num_draft_len: int = 0                         ## 异步调度 + Speculative Decoding 组合使用时的辅助字段     # 用于记录上一次 draft tokens 的长度,处理 rejection 情况


    # for pooling models
    pooling_params: PoolingParams | None = None
    pooling_states: PoolingStates | None = None

    def __post_init__(self):
        """
        dataclass 初始化完成后自动执行的钩子函数。
        用于计算一些派生字段和初始化默认对象
        """
        self.num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
            self.prompt_token_ids, self.prompt_embeds
        )

        if self.pooling_params is not None:
            self.pooling_states = PoolingStates()

    @property
    def num_tokens(self) -> int:
        """
        返回该请求当前总的 token 数量(prompt + output),这是 Scheduler 判断请求长度、分配资源的重要指标
        """
        return self.num_prompt_tokens + len(self.output_token_ids)

    def get_token_id(self, idx: int) -> int:
        #把 “prompt + output” 伪装成一个连续数组，用 idx 统一访问
        if idx < self.num_prompt_tokens:
            if self.prompt_token_ids is None:
                raise ValueError(
                    f"Tried to access token index {idx}, but that token was "
                    "provided via prompt_embeds, and its ID is unknown."
                )
            return self.prompt_token_ids[idx]
        if idx - self.num_prompt_tokens < len(self.output_token_ids):
            return self.output_token_ids[idx - self.num_prompt_tokens]
        return -1


class InputBatch:
    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        device: torch.device,
        pin_memory: bool,
        vocab_size: int,
        block_sizes: list[int],  # The block_size of each kv cache group
        kernel_block_sizes: list[int],
        logitsprocs: LogitsProcessors | None = None,
        logitsprocs_need_output_token_ids: bool = False,
        is_spec_decode: bool = False,
        is_pooling_model: bool = False,
        num_speculative_tokens: int = 0,
        cp_kv_cache_interleave_size: int = 1,
    ):
        """管理一个batch内所有request的输入数据+运行时时态"""
        self.is_pooling_model = is_pooling_model                                      #是否是pooling模型
        self.is_spec_decode = is_spec_decode                                          #是否启用投机解码
        self.max_num_reqs = max_num_reqs                                              #最多多少请求
        self.max_model_len = max_model_len                                            #每个请求最大token长度
        self.max_num_batched_tokens = max_num_batched_tokens                          #一次最多处理多少token
        self.device = device
        self.pin_memory = pin_memory                                                  ##是否使用 pinned memory加速 CPU→GPU 拷贝
        self.vocab_size = vocab_size

        self._req_ids: list[str | None] = []
        self.req_id_to_index: dict[str, int] = {}

        # TODO(woosuk): This buffer could be too large if max_model_len is big.
        # Find a way to reduce the CPU memory usage.
        # This buffer is not directly transferred to the GPU, so it does not           #这个缓冲区不会直接传输到GPU,因此不需要进行pin_memory处理
        # need to be pinned.
        self.token_ids_cpu_tensor = torch.zeros(
            (max_num_reqs, max_model_len),
            device="cpu",
            dtype=torch.int32,
            pin_memory=False,
        )
        self.token_ids_cpu = self.token_ids_cpu_tensor.numpy()                          #一个numpy视图,指向token_ids_cpu_tensor,本质作用:用numpy来操作token，比torch更轻量(CPU)
        self.is_token_ids_tensor = torch.zeros(
            (max_num_reqs, max_model_len), device="cpu", dtype=bool, pin_memory=False   #一个bool矩阵,用来表示这个位置是否真的有token
        )
        self.is_token_ids = self.is_token_ids_tensor.numpy()                            #torch → numpy view(共享内存)
        # Store prompt embeddings per request to avoid OOM from large upfront           #按请求分别存储prompt的embedding,避免在max_model_len很大时一次性分配过多内存导致OOM(溢出)
        # allocation if max_model_len is big.
        # Maps req_index -> tensor of shape (num_prompt_tokens, hidden_size)
        self.req_prompt_embeds: dict[int, torch.Tensor] = {}                            #一个字典:req_index → prompt 的 embedding,什么时候用？当输入不是token id而是已经算好的embedding时
        self.num_tokens_no_spec = np.zeros(max_num_reqs, dtype=np.int32)                #每个request当前真实token数(不包含spec)
        self.num_prompt_tokens = np.zeros(max_num_reqs, dtype=np.int32)                 #每个request的输入长度
        self.num_computed_tokens_cpu_tensor = torch.zeros(                              #每个request算了多少token：[req0算了多少token, req1算了多少token, req2算了多少token, ...]
            (max_num_reqs,),
            device="cpu",
            dtype=torch.int32,
            pin_memory=pin_memory,
        )
        self.num_computed_tokens_cpu = self.num_computed_tokens_cpu_tensor.numpy()

        # Block table.
        self.block_table = MultiGroupBlockTable(
            max_num_reqs=max_num_reqs,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            pin_memory=pin_memory,
            device=device,
            block_sizes=block_sizes,
            kernel_block_sizes=kernel_block_sizes,
            num_speculative_tokens=num_speculative_tokens,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
        )

        # Sampling-related.
        self.temperature = torch.empty(
            (max_num_reqs,), dtype=torch.float32, device=device                           #GPU 上的 temperature,每个request上1个,用于在采样前对 logits 做缩放：logits / temperature
        )
        self.temperature_cpu_tensor = torch.empty(
            (max_num_reqs,), dtype=torch.float32, device="cpu", pin_memory=pin_memory     ##GPU 上的 temperature,pin_memory=True 时可以加速 CPU → GPU 的拷贝,支持DMA
        )
        self.temperature_cpu = self.temperature_cpu_tensor.numpy()
        self.greedy_reqs: set[str] = set()                                                #需要走greedy解码的集合,这些请求可以走更快路径argmax(logits)不需要采样
        self.random_reqs: set[str] = set()                                                #需要走随机采样的请求集合(temprature>0),这些请求执行softmax+随机采样(top-k/top-p)

        self.top_p = torch.empty((max_num_reqs,), dtype=torch.float32, device=device)     #GPU上的top_p
        self.top_p_cpu_tensor = torch.empty(
            (max_num_reqs,), dtype=torch.float32, device="cpu", pin_memory=pin_memory
        )
        self.top_p_cpu = self.top_p_cpu_tensor.numpy()
        self.top_p_reqs: set[str] = set()

        self.top_k = torch.empty((max_num_reqs,), dtype=torch.int32, device=device)
        self.top_k_cpu_tensor = torch.empty(
            (max_num_reqs,), dtype=torch.int32, device="cpu", pin_memory=pin_memory
        )
        self.top_k_cpu = self.top_k_cpu_tensor.numpy()
        self.top_k_reqs: set[str] = set()

        # IDs of requests which do not support spec decoding
        self.spec_decode_unsupported_reqs: set[str] = set()

        # Frequency penalty related data structures
        self.frequency_penalties = torch.empty(
            (max_num_reqs,), dtype=torch.float, device=device
        )
        self.frequency_penalties_cpu_tensor = torch.empty(
            (max_num_reqs,), dtype=torch.float, device="cpu", pin_memory=pin_memory
        )
        self.frequency_penalties_cpu = self.frequency_penalties_cpu_tensor.numpy()
        self.frequency_penalties_reqs: set[str] = set()

        # Presence penalty related data structures
        self.presence_penalties = torch.empty(
            (max_num_reqs,), dtype=torch.float, device=device
        )
        self.presence_penalties_cpu_tensor = torch.empty(
            (max_num_reqs,), dtype=torch.float, device="cpu", pin_memory=pin_memory
        )
        self.presence_penalties_cpu = self.presence_penalties_cpu_tensor.numpy()
        self.presence_penalties_reqs: set[str] = set()

        # Repetition penalty related data structures
        self.repetition_penalties = torch.empty(
            (max_num_reqs,), dtype=torch.float, device=device
        )
        self.repetition_penalties_cpu_tensor = torch.empty(
            (max_num_reqs,), dtype=torch.float, device="cpu", pin_memory=pin_memory
        )
        self.repetition_penalties_cpu = self.repetition_penalties_cpu_tensor.numpy()
        self.repetition_penalties_reqs: set[str] = set()

        # Speculative decoding
        self.num_accepted_tokens_cpu_tensor = torch.ones(
            (max_num_reqs,), dtype=torch.int64, device="cpu", pin_memory=pin_memory
        )
        self.num_accepted_tokens_cpu = self.num_accepted_tokens_cpu_tensor.numpy()

        # lora related
        self.request_lora_mapping = np.zeros((self.max_num_reqs,), dtype=np.int64)
        self.lora_id_to_request_ids: dict[int, set[str]] = {}
        self.lora_id_to_lora_request: dict[int, LoRARequest] = {}

        # req_index -> generator
        # NOTE(woosuk): The indices of the requests that do not have their own                  随机数生成器
        # generator should not be included in the dictionary.
        self.generators: dict[int, torch.Generator] = {}

        self.num_logprobs: dict[str, int] = {}

        # To accumulate prompt logprobs tensor chunks across prefill steps.
        self.in_progress_prompt_logprobs_cpu: dict[str, LogprobsTensors] = {}

        # Internal representation of per-step batch state changes, used for                     用于表示每一步batch状态变化的内部数据机构,用于对持久化batch进行重排
        # reordering persistent batch and generating logitsprocs batch state                    并生成logits processor的batch状态更新,每一步都要重置
        # updates. Should reset each step.
        self.batch_update_builder = BatchUpdateBuilder()

        # TODO convert this to LogitsProcessor
        self.has_allowed_token_ids: set[str] = set()
        # NOTE(lufang): In the mask tensor, if the corresponding token allowed,
        # the value is False. Since we use masked_fill_ to set -inf.
        self.allowed_token_ids_mask: torch.Tensor | None = None
        self.allowed_token_ids_mask_cpu_tensor: torch.Tensor | None = None

        # req_index -> bad_words_token_ids
        self.bad_words_token_ids: dict[int, list[list[int]]] = {}

        self.logits_processing_needs_token_ids = np.zeros(max_num_reqs, dtype=bool)

        self.req_output_token_ids: list[list[int] | None] = []

        # Store provided logitsprocs. If none are provided, initialize empty
        # data structure
        self.logitsprocs = logitsprocs or LogitsProcessors()
        self.logitsprocs_need_output_token_ids = logitsprocs_need_output_token_ids

        # Store last speculative tokens for sampler.
        self.spec_token_ids: list[list[int]] = [[] for _ in range(max_num_reqs)]

        # This is updated each time the batch constituents change.
        self.sampling_metadata = self._make_sampling_metadata()

        # for pooling models
        self.pooling_params: dict[str, PoolingParams] = {}
        self.pooling_states: dict[str, PoolingStates] = {}

        # Cached reference to the GPU tensor of previously sampled tokens
        self.prev_sampled_token_ids: torch.Tensor | None = None
        self.prev_req_id_to_index: dict[str, int] | None = None
        # These are used to update output_token_ids with real sampled
        # ids from prior step, if required by current sampling params
        # (e.g. penalties).
        self.sampled_token_ids_cpu: torch.Tensor | None = None
        self.async_copy_ready_event: torch.Event | None = None

    @property
    def req_ids(self) -> list[str]:
        # None elements should only be present transiently
        # while performing state updates to the batch.
        return cast(list[str], self._req_ids)

    def _register_add_request(self, request: "CachedRequestState") -> int:
        """Track add-request operations for logits processors.
        Not applicable to pooling models.
        主要作用：
        1. 决定新请求应该放在 batch 的哪个位置(复用空位 或 在末尾追加)
        2. 记录这次添加操作,供后续 logits processors(logits 处理器)使用
        3. 更新 batch_update_builder 的状态,标记 batch 发生了变化
        
        注意：该函数主要服务于普通生成模型(Autoregressive),Pooling 模型使用较少。
        """

        # Fill the next empty index if there is one.
        # ====================== 分配请求索引的核心逻辑 ======================
        # 优先尝试复用之前被 remove_request() 留下的空位(通过 pop_removed 获取)
        # 如果没有可复用的空位(返回 None),则在 batch 末尾追加新请求
        if (new_req_index := self.batch_update_builder.pop_removed()) is None:
            # Append to end otherwise.
            new_req_index = self.num_reqs

        # 确保分配的索引没有超出batch的最大容量
        assert new_req_index < self.max_num_reqs
        ## 标记 batch 已经发生变化(后续 refresh_metadata() 会根据这个标志决定是否重建 sampling_metadata)
        self.batch_update_builder.batch_changed = True
        
        # ====================== 记录添加操作(供 logits processors 使用) ======================
        # 只有当请求带有 sampling_params(即普通生成模型)时,才需要记录详细的添加信息
        # Pooling 模型(embedding 等)不需要这些信息,所以跳过
        if request.sampling_params:
            # Detailed added request metadata is only required for non-pooling
            # models, to support logitsprocs.
            # 将本次新增请求的信息记录到 added 列表中
            # 这些信息后续会被 logits processors(如 RepetitionPenalty、Grammar、JSON Schema 等)使用,
            # 以便正确更新内部状态(例如把新 token 加入历史、更新 grammar FSM 等)
            self.batch_update_builder.added.append(
                (
                    new_req_index,                      #新请求在batch中的索引
                    request.sampling_params,            #采样参数
                    request.prompt_token_ids,           #prompt的token ids
                    request.output_token_ids,           #已经生成的ouput tokens
                )
            )

        return new_req_index

    def add_request(
        self,
        request: "CachedRequestState",
    ) -> int:
        """
        将一个新的请求(或从 waiting 队列移入的请求)加入到当前 InputBatch 中。
        这是 InputBatch 中最核心的添加函数之一。
        每次有新请求完成 prefill(或被 scheduler 调度进入 decode 阶段)时,都会调用此方法。
        
        返回值：该请求在当前 batch 中被分配的索引(req_index)
        """
        
        #第一步：在 batch_update_builder 中注册添加请求,并获取分配的索引位置
        # 如果 batch 有空位,会复用空位；否则在末尾追加
        req_index = self._register_add_request(request)

        req_id = request.req_id
        
        # ====================== 更新请求基本信息 ======================
        if req_index == len(self._req_ids):
            self._req_ids.append(req_id)
            self.req_output_token_ids.append(request.output_token_ids)
            self.spec_token_ids.append([])                                  ## speculative decoding 的 draft tokens 列表
        else:
            self._req_ids[req_index] = req_id                               # 复用之前被删除请求留下的空位
            self.req_output_token_ids[req_index] = request.output_token_ids
            self.spec_token_ids[req_index].clear()

        self.req_id_to_index[req_id] = req_index                            # 更新请求 ID 到索引的映射(用于快速查找)

        #复制token相关数据 计算 prompt 的 token 数量(支持 prompt_token_ids 或 prompt_embeds 两种方式)
        # Copy the prompt token ids and output token ids.
        num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
            request.prompt_token_ids, request.prompt_embeds
        )
        self.num_prompt_tokens[req_index] = num_prompt_tokens
        
        #计算putput token的起始和结束位置
        start_idx = num_prompt_tokens
        end_idx = start_idx + len(request.output_token_ids)
        
        # 复制 prompt token ids(如果有)
        if request.prompt_token_ids is not None:
            self.token_ids_cpu[req_index, :num_prompt_tokens] = request.prompt_token_ids
            self.is_token_ids[req_index, :num_prompt_tokens] = True
        else:
            self.is_token_ids[req_index, :num_prompt_tokens] = False
            
        # 保存 prompt embeds(主要用于多模态或直接传入 embedding 的情况)
        if request.prompt_embeds is not None:
            self.req_prompt_embeds[req_index] = request.prompt_embeds
            
        #复制已生成的 output token ids
        self.token_ids_cpu[req_index, start_idx:end_idx] = request.output_token_ids
        self.is_token_ids[req_index, start_idx:end_idx] = True
        # Number of tokens without spec decode tokens. 记录不包含 speculative tokens 的 token 数量
        self.num_tokens_no_spec[req_index] = request.num_tokens
        
        # 更新已计算的 token 数量,整个请求到目前为止真正计算完成的token总数(包含prompt+已经验证通过的generated tokens)
        self.num_computed_tokens_cpu[req_index] = request.num_computed_tokens
        # 将该请求的 KV Cache block 信息加入 block_table
        self.block_table.add_row(request.block_ids, req_index)

        # ====================== 处理 Sampling 参数(普通生成模型) =====================
        if sampling_params := request.sampling_params:
            
            # Speculative Decoding 支持性检查
            if self.is_spec_decode and is_spec_decode_unsupported(sampling_params):
                self.spec_decode_unsupported_reqs.add(req_id)
            # ==================== 采样类型分类 ====================
            if sampling_params.sampling_type == SamplingType.GREEDY:
                # Should avoid division by zero later when apply_temperature.# Greedy 解码(temperature=0),后续避免除以 0
                self.temperature_cpu[req_index] = 0.0
                self.greedy_reqs.add(req_id)
            else:
                self.temperature_cpu[req_index] = sampling_params.temperature
                self.random_reqs.add(req_id)

            self.top_p_cpu[req_index] = sampling_params.top_p
            if sampling_params.top_p < 1:
                self.top_p_reqs.add(req_id)
            top_k = sampling_params.top_k
            if 0 < top_k < self.vocab_size:
                self.top_k_reqs.add(req_id)
            else:
                top_k = self.vocab_size
            self.top_k_cpu[req_index] = top_k
            self.frequency_penalties_cpu[req_index] = sampling_params.frequency_penalty
            if sampling_params.frequency_penalty != 0.0:
                self.frequency_penalties_reqs.add(req_id)
            self.presence_penalties_cpu[req_index] = sampling_params.presence_penalty
            if sampling_params.presence_penalty != 0.0:
                self.presence_penalties_reqs.add(req_id)
            self.repetition_penalties_cpu[req_index] = (
                sampling_params.repetition_penalty
            )
            if sampling_params.repetition_penalty != 1.0:
                self.repetition_penalties_reqs.add(req_id)

            # NOTE(woosuk): self.generators should not include the requests that
            # do not have their own generator.
            if request.generator is not None:
                self.generators[req_index] = request.generator

            if sampling_params.logprobs is not None:
                self.num_logprobs[req_id] = (
                    self.vocab_size
                    if sampling_params.logprobs == -1
                    else sampling_params.logprobs
                )

            if sampling_params.allowed_token_ids:
                self.has_allowed_token_ids.add(req_id)
                if self.allowed_token_ids_mask_cpu_tensor is None:
                    # Lazy allocation for this tensor, which can be large.
                    # False means we don't fill with -inf.
                    self.allowed_token_ids_mask = torch.zeros(
                        self.max_num_reqs,
                        self.vocab_size,
                        dtype=torch.bool,
                        device=self.device,
                    )
                    self.allowed_token_ids_mask_cpu_tensor = torch.zeros(
                        self.max_num_reqs,
                        self.vocab_size,
                        dtype=torch.bool,
                        device="cpu",
                    )
                self.allowed_token_ids_mask_cpu_tensor[req_index] = True
                # False means we don't fill with -inf.
                self.allowed_token_ids_mask_cpu_tensor[req_index][
                    sampling_params.allowed_token_ids
                ] = False

            if sampling_params.bad_words_token_ids:
                self.bad_words_token_ids[req_index] = (
                    sampling_params.bad_words_token_ids
                )
        elif pooling_params := request.pooling_params:
            pooling_states = request.pooling_states
            assert pooling_states is not None

            self.pooling_params[req_id] = pooling_params
            self.pooling_states[req_id] = pooling_states
            self.logits_processing_needs_token_ids[req_index] = (
                pooling_params.requires_token_ids
            )
        else:
            raise NotImplementedError("Unrecognized request type")

        # Speculative decoding: by default 1 token is generated.
        self.num_accepted_tokens_cpu[req_index] = 1

        # Add request lora ID
        if request.lora_request:
            lora_id = request.lora_request.lora_int_id
            if lora_id not in self.lora_id_to_request_ids:
                self.lora_id_to_request_ids[lora_id] = set()

            self.request_lora_mapping[req_index] = lora_id
            self.lora_id_to_request_ids[lora_id].add(request.req_id)
            self.lora_id_to_lora_request[lora_id] = request.lora_request
        else:
            # No LoRA
            self.request_lora_mapping[req_index] = 0

        return req_index

    def remove_request(self, req_id: str) -> int | None:
        """This method must always be followed by a call to condense().从当前InputBatch中移除一个请求
        这个方法必须在调用之后紧接着调用condense(),因为该函数只是“标记”要删除,并没有真正压缩batch(为了性能),condense()才会把真正可能搞出来的位置填补上,实现batch压缩
        Args:
          req_id: request to remove

        Returns:
          Removed request index, or `None` if `req_id` not recognized
          返回被溢出的索引,否则None
        """

        req_index = self.req_id_to_index.pop(req_id, None)
        if req_index is None:
            return None

        self.batch_update_builder.removed_append(req_index)
        self._req_ids[req_index] = None
        self.req_output_token_ids[req_index] = None
        self.spec_token_ids[req_index].clear()

        # LoRA
        lora_id = self.request_lora_mapping[req_index]
        if lora_id != 0:
            lora_req_ids = self.lora_id_to_request_ids[lora_id]
            lora_req_ids.discard(req_id)
            if not lora_req_ids:
                del self.lora_id_to_request_ids[lora_id]
                del self.lora_id_to_lora_request[lora_id]
            self.request_lora_mapping[req_index] = 0

        if self.is_pooling_model:
            self.pooling_params.pop(req_id, None)
            self.pooling_states.pop(req_id, None)
            return req_index

        self.greedy_reqs.discard(req_id)
        self.random_reqs.discard(req_id)
        self.top_p_reqs.discard(req_id)
        self.top_k_reqs.discard(req_id)
        self.spec_decode_unsupported_reqs.discard(req_id)
        self.frequency_penalties_reqs.discard(req_id)
        self.presence_penalties_reqs.discard(req_id)
        self.repetition_penalties_reqs.discard(req_id)
        self.generators.pop(req_index, None)
        self.num_logprobs.pop(req_id, None)
        self.in_progress_prompt_logprobs_cpu.pop(req_id, None)
        if self.prev_req_id_to_index is not None:
            self.prev_req_id_to_index.pop(req_id, None)

        self.has_allowed_token_ids.discard(req_id)
        if self.allowed_token_ids_mask_cpu_tensor is not None:
            # False means we don't fill with -inf.
            self.allowed_token_ids_mask_cpu_tensor[req_index].fill_(False)
        self.bad_words_token_ids.pop(req_index, None)
        return req_index

    def swap_states(self, i1: int, i2: int) -> None:
        old_id_i1 = self._req_ids[i1]
        old_id_i2 = self._req_ids[i2]
        self._req_ids[i1], self._req_ids[i2] = self._req_ids[i2], self._req_ids[i1]  # noqa
        self.req_output_token_ids[i1], self.req_output_token_ids[i2] = (
            self.req_output_token_ids[i2],
            self.req_output_token_ids[i1],
        )
        self.spec_token_ids[i1], self.spec_token_ids[i2] = (
            self.spec_token_ids[i2],
            self.spec_token_ids[i1],
        )
        assert old_id_i1 is not None and old_id_i2 is not None
        self.req_id_to_index[old_id_i1], self.req_id_to_index[old_id_i2] = (
            self.req_id_to_index[old_id_i2],
            self.req_id_to_index[old_id_i1],
        )
        self.num_tokens_no_spec[i1], self.num_tokens_no_spec[i2] = (
            self.num_tokens_no_spec[i2],
            self.num_tokens_no_spec[i1],
        )
        self.num_prompt_tokens[i1], self.num_prompt_tokens[i2] = (
            self.num_prompt_tokens[i2],
            self.num_prompt_tokens[i1],
        )
        self.num_computed_tokens_cpu[i1], self.num_computed_tokens_cpu[i2] = (
            self.num_computed_tokens_cpu[i2],
            self.num_computed_tokens_cpu[i1],
        )

        # NOTE: the following is unsafe
        # self.token_ids_cpu[i1, ...], self.token_ids_cpu[i2, ...], =\
        #     self.token_ids_cpu[i2, ...], self.token_ids_cpu[i1, ...]
        # instead, we need to temporarily copy the data for one of the indices
        # TODO(lucas): optimize this by only copying valid indices
        tmp = self.token_ids_cpu[i1, ...].copy()
        self.token_ids_cpu[i1, ...] = self.token_ids_cpu[i2, ...]
        self.token_ids_cpu[i2, ...] = tmp

        self.is_token_ids[[i1, i2], ...] = self.is_token_ids[[i2, i1], ...]

        # Swap prompt embeddings if they exist
        embeds_i1 = self.req_prompt_embeds.get(i1)
        embeds_i2 = self.req_prompt_embeds.get(i2)
        if embeds_i1 is not None:
            self.req_prompt_embeds[i2] = embeds_i1
        else:
            self.req_prompt_embeds.pop(i2, None)
        if embeds_i2 is not None:
            self.req_prompt_embeds[i1] = embeds_i2
        else:
            self.req_prompt_embeds.pop(i1, None)

        self.block_table.swap_row(i1, i2)

        self.request_lora_mapping[i1], self.request_lora_mapping[i2] = (
            self.request_lora_mapping[i2],
            self.request_lora_mapping[i1],
        )

        if self.is_pooling_model:
            # Sampling and logits parameters don't apply to pooling models.
            return

        # For autoregressive models, track detailed request reordering info
        # to support logitsprocs.
        self.batch_update_builder.moved.append((i1, i2, MoveDirectionality.SWAP))

        self.temperature_cpu[i1], self.temperature_cpu[i2] = (
            self.temperature_cpu[i2],
            self.temperature_cpu[i1],
        )
        self.top_p_cpu[i1], self.top_p_cpu[i2] = self.top_p_cpu[i2], self.top_p_cpu[i1]
        self.top_k_cpu[i1], self.top_k_cpu[i2] = self.top_k_cpu[i2], self.top_k_cpu[i1]
        self.frequency_penalties_cpu[i1], self.frequency_penalties_cpu[i2] = (
            self.frequency_penalties_cpu[i2],
            self.frequency_penalties_cpu[i1],
        )
        self.presence_penalties_cpu[i1], self.presence_penalties_cpu[i2] = (
            self.presence_penalties_cpu[i2],
            self.presence_penalties_cpu[i1],
        )
        self.repetition_penalties_cpu[i1], self.repetition_penalties_cpu[i2] = (
            self.repetition_penalties_cpu[i2],
            self.repetition_penalties_cpu[i1],
        )
        self.num_accepted_tokens_cpu[i1], self.num_accepted_tokens_cpu[i2] = (
            self.num_accepted_tokens_cpu[i2],
            self.num_accepted_tokens_cpu[i1],
        )

        swap_dict_values(self.generators, i1, i2)
        swap_dict_values(self.bad_words_token_ids, i1, i2)

        if self.allowed_token_ids_mask_cpu_tensor is not None:
            (
                self.allowed_token_ids_mask_cpu_tensor[i1],
                self.allowed_token_ids_mask_cpu_tensor[i2],
            ) = (
                self.allowed_token_ids_mask_cpu_tensor[i2],
                self.allowed_token_ids_mask_cpu_tensor[i1],
            )

    def condense(self) -> None:
        """Slide non-empty requests down into lower, empty indices.压缩当前InputBatch,将有效请求[向前滑动],填补被remove_request留下的空洞

        Any consecutive empty indices at the very end of the list are not 这是一个非常重要的batch维护操作,其核心目的是：
        filled.                                                           消除batch中间的空位,让所有活跃请求排在前边
                                                                          提高GPU利用率(避免forward时计算大量无效位置) 为下一次模型执行准备紧凑的输入
        Returns:
          swaps: list of (from,to) swap tuples for moved requests
          empty_req_indices: indices not filled by condensation
        """
        num_reqs = self.num_reqs

        #如果没有需要溢出的请求(empty_req_indices为空)说明：要么没有删除请求 要么删除的请求已经被新加入的请求完美替换,此时无需condense 直接返回
        if not (empty_req_indices := self.batch_update_builder.removed):
            # All removed requests were replaced by added requests, or else no
            # requests were removed at all. No condense() needed
            return
        if num_reqs == 0:
            # The batched states are empty.
            self._req_ids.clear()
            self.req_output_token_ids.clear()
            self.spec_token_ids.clear()
            return

        # NOTE(woosuk): This function assumes that the empty_req_indices
        # is sorted in descending order.最后一个可能有数据的索引=当前请求数+ 被移除的空位数-1
        last_req_index = num_reqs + len(empty_req_indices) - 1
        
        # ==================== 核心压缩循环 ====================
        # 从后往前处理,把后面的有效请求往前移动,填补前面的空位
        while empty_req_indices:
            # Find the largest non-empty index.  找到当前最大的非空索引(即还有有效请求的位置)
            while last_req_index in empty_req_indices:
                last_req_index -= 1

            # Find the smallest empty index. 找到当前最小的空位索引(需要被填补的位置)
            empty_index = self.batch_update_builder.peek_removed()
            assert empty_index is not None
            #如果空位索引已经 >= 最后一个有效请求的索引,说明后面已经没有可移动的内容了
            if empty_index >= last_req_index:
                break
            
            # ==================== 执行移动操作 ====================
            # 弹出当前要填补的空位
            # Move active request down into empty request
            # index.
            self.batch_update_builder.pop_removed()
            
            # 把 last_req_index 处的有效请求移动到 empty_index 处
            req_id = self._req_ids[last_req_index]
            output_token_ids = self.req_output_token_ids[last_req_index]
            assert req_id is not None
            
            # 更新各种数据结构中的位置
            self._req_ids[empty_index] = req_id
            self._req_ids[last_req_index] = None
            self.req_output_token_ids[empty_index] = output_token_ids
            self.req_output_token_ids[last_req_index] = None
            self.req_id_to_index[req_id] = empty_index                  # 更新请求ID到索引的映射

            # 处理 speculative decoding 的 token
            num_tokens = self.num_tokens_no_spec[last_req_index] + len(
                self.spec_token_ids[last_req_index]
            )
            # 交换 spec_token_ids(使用 tuple 解包方式交换)
            (self.spec_token_ids[last_req_index], self.spec_token_ids[empty_index]) = (
                self.spec_token_ids[empty_index],
                self.spec_token_ids[last_req_index],
            )
            self.spec_token_ids[last_req_index].clear()

            # 复制 token ids 相关 tensor 数据
            self.token_ids_cpu[empty_index, :num_tokens] = self.token_ids_cpu[
                last_req_index, :num_tokens
            ]
            self.is_token_ids[empty_index, :num_tokens] = self.is_token_ids[
                last_req_index, :num_tokens
            ]
            # 处理 prompt_embeds(如果存在)
            if last_req_index in self.req_prompt_embeds:
                self.req_prompt_embeds[empty_index] = self.req_prompt_embeds.pop(
                    last_req_index
                )
                
            # 复制其他计数信息
            self.num_tokens_no_spec[empty_index] = self.num_tokens_no_spec[
                last_req_index
            ]
            self.num_prompt_tokens[empty_index] = self.num_prompt_tokens[last_req_index]
            self.num_computed_tokens_cpu[empty_index] = self.num_computed_tokens_cpu[
                last_req_index
            ]
            # 移动 KV Cache block table 中的对应行
            self.block_table.move_row(last_req_index, empty_index)
            # 复制 LoRA 映射
            self.request_lora_mapping[empty_index] = self.request_lora_mapping[
                last_req_index
            ]
            # ====================== Pooling 模型处理 ======================
            if self.is_pooling_model:
                last_req_index -= 1
                # Sampling state not used by pooling models.
                continue
                
            # ====================== 普通生成模型的额外处理 ======================
            # 记录这次移动操作(用于后续 logits processor 等需要知道请求位置变化的场景)
            # Autoregressive models require detailed tracking of condense
            # operations to support logitsprocs
            self.batch_update_builder.moved.append(
                (last_req_index, empty_index, MoveDirectionality.UNIDIRECTIONAL)
            )
            # 复制各类采样参数(从 CPU tensor 中移动)
            self.temperature_cpu[empty_index] = self.temperature_cpu[last_req_index]
            self.top_p_cpu[empty_index] = self.top_p_cpu[last_req_index]
            self.top_k_cpu[empty_index] = self.top_k_cpu[last_req_index]
            self.frequency_penalties_cpu[empty_index] = self.frequency_penalties_cpu[
                last_req_index
            ]
            self.presence_penalties_cpu[empty_index] = self.presence_penalties_cpu[
                last_req_index
            ]
            self.repetition_penalties_cpu[empty_index] = self.repetition_penalties_cpu[
                last_req_index
            ]
            self.num_accepted_tokens_cpu[empty_index] = self.num_accepted_tokens_cpu[
                last_req_index
            ]
            # 移动随机数生成器
            generator = self.generators.pop(last_req_index, None)
            if generator is not None:
                self.generators[empty_index] = generator

            # TODO convert these to LogitsProcessors # 移动 allowed_token_ids mask 和 bad_words
            if self.allowed_token_ids_mask_cpu_tensor is not None:
                self.allowed_token_ids_mask_cpu_tensor[empty_index] = (
                    self.allowed_token_ids_mask_cpu_tensor[last_req_index]
                )

            bad_words_token_ids = self.bad_words_token_ids.pop(last_req_index, None)
            if bad_words_token_ids is not None:
                self.bad_words_token_ids[empty_index] = bad_words_token_ids

            # Decrement last_req_index since it is now empty. # 当前 last_req_index 位置已被清空,向前移动指针
            last_req_index -= 1
        # ==================== 最终裁剪列表大小 ====================
        # 把所有列表裁剪到实际有效的请求数量,释放多余的内存空间
        # Trim lists to the batch size.
        del self._req_ids[num_reqs:]
        del self.req_output_token_ids[num_reqs:]
        del self.spec_token_ids[num_reqs:]

    def refresh_metadata(self):
        """Apply any batch updates to sampling metadata.
        刷新 InputBatch 的元数据(metadata),并将之前累积的 batch 更新应用到采样相关结构中。
        
        这个函数是每次 condense() + reorder_batch 之后必须执行的一步,
        作用相当于“提交最终变更” —— 让所有对 batch 的修改(添加、删除、移动请求)生效。
        
        它主要负责两件事：
        1. 重置 batch 更新追踪器(batch_update_builder)
        2. 如果 batch 发生了变化,则重新生成 sampling_metadata
        
        """
        # ====================== Pooling 模型(Embedding / Reward Model 等)的处理 ======================
        if self.is_pooling_model:
            #对于非生成类的pooling模型,逻辑相对简单,reset() 会返回本次是否有过添加/删除/移动操作(即 batch 是否发生变化)
            batch_changed = self.batch_update_builder.reset()
            
            #如果batch有变化,则重新胜场sampling_metadata ,pooling模型不需要Logits processors所以直接重建即可
            if batch_changed:
                self.sampling_metadata = self._make_sampling_metadata()
            return

        # For non-pooling models - generate and apply logitsprocs update;
        # reset batch update tracking.
        # Update sampling metadata if batch state is changed.
        # ====================== 普通生成模型(Autoregressive Model,如 Qwen3)的处理 ======================
        # 对于生成模型,需要额外处理 logits processors(logits 处理器)
        # 例如：Repetition Penalty、Frequency Penalty、JSON Schema Grammar、Bad Words、Allowed Tokens 等
        # 1. 获取并重置 batch 更新信息
        # get_and_reset() 会返回本次 step 中所有发生的变更(哪些请求被移动、删除、添加等)
        # 同时把 builder 重置,为下一轮
        batch_update = self.batch_update_builder.get_and_reset(self.num_reqs)
        
        # 2. 把 batch 的变更通知给所有已注册的 logits processors
        # 让它们更新内部状态(例如调整 mask、更新 token 历史等)
        for logit_proc in self.logitsprocs.all:
            logit_proc.update_state(batch_update)
            
        # 3. 如果本次 batch 有任何变更(添加、删除、移动等),则重新生成 sampling_metadata
        # sampling_metadata 包含了 temperature、top_p、top_k、logprobs、generators 等关键采样信息
        # 它会被后续的 sample_tokens() 和 logits 处理流程使用
        if batch_update:
            self.sampling_metadata = self._make_sampling_metadata()

    def _make_sampling_metadata(self) -> SamplingMetadata:
        num_reqs = self.num_reqs
        if not self.all_greedy:
            temperature = copy_slice(
                self.temperature_cpu_tensor, self.temperature, num_reqs
            )
        else:
            temperature = None
        if not self.no_top_p:
            copy_slice(self.top_p_cpu_tensor, self.top_p, num_reqs)
        if not self.no_top_k:
            copy_slice(self.top_k_cpu_tensor, self.top_k, num_reqs)

        if not self.no_penalties:
            # Since syncing these tensors is expensive only copy them
            # if necessary i.e. if there are requests which require
            # penalties to be applied during sampling.
            copy_slice(
                self.frequency_penalties_cpu_tensor, self.frequency_penalties, num_reqs
            )
            copy_slice(
                self.presence_penalties_cpu_tensor, self.presence_penalties, num_reqs
            )
            copy_slice(
                self.repetition_penalties_cpu_tensor,
                self.repetition_penalties,
                num_reqs,
            )

        needs_prompt_token_ids = (
            not self.no_penalties
            or self.logits_processing_needs_token_ids[:num_reqs].any()
        )
        # The prompt tokens are used only for applying penalties or
        # step pooling during the sampling/pooling process.
        # Hence copy these tensors only when there are requests which
        # need penalties/step_pooler to be applied.
        prompt_token_ids = (
            self._make_prompt_token_ids_tensor() if needs_prompt_token_ids else None
        )

        # Only set output_token_ids if required by the current requests'
        # sampling parameters.
        needs_output_token_ids = (
            not self.no_penalties
            or bool(self.bad_words_token_ids)
            or self.logitsprocs_need_output_token_ids
        )
        output_token_ids = (
            cast(list[list[int]], self.req_output_token_ids)
            if needs_output_token_ids
            else []
        )

        allowed_token_ids_mask: torch.Tensor | None = None
        if not self.no_allowed_token_ids:
            assert self.allowed_token_ids_mask is not None
            copy_slice(
                self.allowed_token_ids_mask_cpu_tensor,
                self.allowed_token_ids_mask,
                num_reqs,
            )
            allowed_token_ids_mask = self.allowed_token_ids_mask[:num_reqs]

        return SamplingMetadata(
            temperature=temperature,
            all_greedy=self.all_greedy,
            all_random=self.all_random,
            top_p=None if self.no_top_p else self.top_p[:num_reqs],
            top_k=None if self.no_top_k else self.top_k[:num_reqs],
            generators=self.generators,
            max_num_logprobs=self.max_num_logprobs,
            prompt_token_ids=prompt_token_ids,
            frequency_penalties=self.frequency_penalties[:num_reqs],
            presence_penalties=self.presence_penalties[:num_reqs],
            repetition_penalties=self.repetition_penalties[:num_reqs],
            output_token_ids=output_token_ids,
            spec_token_ids=cast(list[list[int]], self.spec_token_ids),
            no_penalties=self.no_penalties,
            allowed_token_ids_mask=allowed_token_ids_mask,
            bad_words_token_ids=self.bad_words_token_ids,
            logitsprocs=self.logitsprocs,
        )

    def get_pooling_params(self) -> list[PoolingParams]:
        assert len(self.req_ids) == len(self.pooling_params)
        return [self.pooling_params[req_id] for req_id in self.req_ids]

    def get_pooling_states(self) -> list[PoolingStates]:
        assert len(self.req_ids) == len(self.pooling_states)
        return [self.pooling_states[req_id] for req_id in self.req_ids]

    def get_pooling_metadata(self) -> PoolingMetadata:
        pooling_params = self.get_pooling_params()
        pooling_states = self.get_pooling_states()

        return PoolingMetadata(
            prompt_lens=torch.from_numpy(self.num_prompt_tokens[: self.num_reqs]),
            prompt_token_ids=self.sampling_metadata.prompt_token_ids,
            pooling_params=pooling_params,
            pooling_states=pooling_states,
        )

    def _make_prompt_token_ids_tensor(self) -> torch.Tensor:
        num_reqs = self.num_reqs
        max_prompt_len = self.num_prompt_tokens[:num_reqs].max()
        prompt_token_ids_cpu_tensor = torch.empty(
            (self.num_reqs, max_prompt_len),
            device="cpu",
            dtype=torch.int64,
            pin_memory=self.pin_memory,
        )
        prompt_token_ids = prompt_token_ids_cpu_tensor.numpy()
        prompt_token_ids[:] = self.token_ids_cpu[:num_reqs, :max_prompt_len]
        # Use the value of vocab_size as a pad since we don't have a
        # token_id of this value.
        for i in range(num_reqs):
            prompt_token_ids[i, self.num_prompt_tokens[i] :] = self.vocab_size
        return prompt_token_ids_cpu_tensor.to(device=self.device, non_blocking=True)

    def make_lora_inputs(
        self, num_scheduled_tokens: np.ndarray, num_sampled_tokens: np.ndarray
    ) -> tuple[tuple[int, ...], tuple[int, ...], set[LoRARequest]]:
        """
        Given the num_scheduled_tokens for each request in the batch, return
        datastructures used to activate the current LoRAs.
        Returns:
            1. prompt_lora_mapping: A tuple of size np.sum(num_sampled_tokens)
               where, prompt_lora_mapping[i] is the LoRA id to use for the ith
               sampled token.
            2. token_lora_mapping: A tuple of size np.sum(num_scheduled_tokens)
               where, token_lora_mapping[i] is the LoRA id to use for ith token.
            3. lora_requests: Set of relevant LoRA requests.
        """

        req_lora_mapping = self.request_lora_mapping[: self.num_reqs]
        prompt_lora_mapping = tuple(req_lora_mapping.repeat(num_sampled_tokens))
        token_lora_mapping = tuple(req_lora_mapping.repeat(num_scheduled_tokens))

        active_lora_requests: set[LoRARequest] = set(
            self.lora_id_to_lora_request.values()
        )

        return prompt_lora_mapping, token_lora_mapping, active_lora_requests

    def set_async_sampled_token_ids(
        self,
        sampled_token_ids_cpu: torch.Tensor,
        async_copy_ready_event: torch.Event,
    ) -> None:
        """
        In async scheduling case, store ref to sampled_token_ids_cpu
        tensor and corresponding copy-ready event. Used to repair
        output_token_ids prior to sampling, if needed by logits processors.
        """
        if self.sampling_metadata.output_token_ids:
            self.sampled_token_ids_cpu = sampled_token_ids_cpu
            self.async_copy_ready_event = async_copy_ready_event
        else:
            self.sampled_token_ids_cpu = None
            self.async_copy_ready_event = None

    def update_async_output_token_ids(self) -> None:
        """
        In async scheduling case, update output_token_ids in sampling metadata   在异步调度模式下使用
        from prior steps sampled token ids once they've finished copying to CPU. 作用：在采样得到的token ids从GPU异步复制到CPU完成后
        This is called right before they are needed by the logits processors.    把之前使用的占位符-1 替换成真实的sampled token id
        这个函数会在 logits processors(比如 grammar、JSON schema、logits bias 等)需要使用 output_token_ids 之前被调用,
        确保后续处理使用的是正确的 token 序列。
        """
        
        #sampling_metadata 中维护的每个 request 的 output_token_ids 列表
        #正常情况下里面存的是已经生成的历史output tokens
        output_token_ids = self.sampling_metadata.output_token_ids
        
        #如果没有异步复制的结果,或当前batch不需要output_token_ids,就直接返回,比如普通同步调度模式,或者pooling模型等场景
        if self.sampled_token_ids_cpu is None or not output_token_ids:
            # Output token ids not needed or not async scheduling.      为啥异步不一样 ,异步execute_model 只做 forward,采样结果先异步复制到 CPU
            return                                                      # 同步在exexcute_model内部或sample_tokens中立即采样

        # prev_req_id_to_index：记录了“上一个 step 的 request 在 batch 中的索引”
        # 因为异步调度下,本次 batch 的 request 顺序可能和上一次不同,需要做映射
        assert self.prev_req_id_to_index is not None
        sampled_token_ids = None                                        # 延迟初始化,真正需要时才从 CPU tensor 转成 list
        
        # ==================== 核心循环：遍历当前 batch 中的所有请求 ====================
        for index, req_id in enumerate(self.req_ids):
            prev_index = self.prev_req_id_to_index.get(req_id)          # 尝试找到这个 request 在“上一个 step”中的索引位置# (因为 sampled_token_ids_cpu 是上一次 execute_model + sampling 的结果)
            if prev_index is None:
                continue                                                # 这个请求是新加入的,本次不需要更新(比如刚 prefill 完进入 decode)
            
            
            #获取当前request 的 output_token_ids 列表(引用)
            req_output_token_ids = output_token_ids[index]
            
            # 如果该 request 还没有 output tokens,或者最后一个 token 已经不是占位符 -1,
            # 说明之前已经处理过了,或者因为 KV load 失败导致 token 被丢弃了
            if not req_output_token_ids or req_output_token_ids[-1] != -1:
                # Final output id is not a placeholder, some tokens must have
                # been discarded after a kv-load failure.
                continue
            
            # 第一次需要使用 sampled_token_ids 时,才真正执行同步操作
            # 这是一种延迟同步(lazy synchronize),尽量减少不必要的 GPU-CPU 同步开销
            if sampled_token_ids is None:
                assert self.async_copy_ready_event is not None
                # 等待异步复制完成(GPU → CPU)
                # 这行会阻塞,直到 sampled_token_ids_cpu 里的数据真正可用
                self.async_copy_ready_event.synchronize()
                # 把 GPU 上采样得到的 token ids(形状 [batch, 1])转成 Python list
                # squeeze(-1) 是去掉最后一个维度,变成一维
                sampled_token_ids = self.sampled_token_ids_cpu.squeeze(-1).tolist()
            # Replace placeholder token id with actual sampled id.
            # ==================== 关键操作：替换占位符 ====================
            # 把之前临时填的 -1 替换成真正采样出来的 token id
            # 例如：原来可能是 [151667, -1]  →  替换后变成 [151667, 151668](<think> ... </think>)
            req_output_token_ids[-1] = sampled_token_ids[prev_index]

    @property
    def num_reqs(self) -> int:
        return len(self.req_id_to_index)

    @property
    def all_greedy(self) -> bool:
        return len(self.random_reqs) == 0

    @property
    def all_random(self) -> bool:
        return len(self.greedy_reqs) == 0

    @property
    def no_top_p(self) -> bool:
        return len(self.top_p_reqs) == 0

    @property
    def no_top_k(self) -> bool:
        return len(self.top_k_reqs) == 0

    @property
    def no_penalties(self) -> bool:
        return (
            len(self.presence_penalties_reqs) == 0
            and len(self.frequency_penalties_reqs) == 0
            and len(self.repetition_penalties_reqs) == 0
        )

    @property
    def max_num_logprobs(self) -> int | None:
        return max(self.num_logprobs.values()) if self.num_logprobs else None

    @property
    def no_allowed_token_ids(self) -> bool:
        return len(self.has_allowed_token_ids) == 0
