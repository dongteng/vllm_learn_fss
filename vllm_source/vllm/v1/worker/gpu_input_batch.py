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
    主要作用是:
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
        #把 “prompt + output” 伪装成一个连续数组,用 idx 统一访问
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
        self.is_pooling_model = is_pooling_model                                            #是否是pooling模型
        self.is_spec_decode = is_spec_decode                                                #是否启用投机解码
        self.max_num_reqs = max_num_reqs                                                    #最多多少请求
        self.max_model_len = max_model_len                                                  #每个请求最大token长度
        self.max_num_batched_tokens = max_num_batched_tokens                                #一次最多处理多少token
        self.device = device
        self.pin_memory = pin_memory                                                        #是否使用 pinned memory加速 CPU→GPU 拷贝
        self.vocab_size = vocab_size

        self._req_ids: list[str | None] = []
        self.req_id_to_index: dict[str, int] = {}

        # TODO(woosuk): This buffer could be too large if max_model_len is big.
        # Find a way to reduce the CPU memory usage.
        # This buffer is not directly transferred to the GPU, so it does not                #这个缓冲区不会直接传输到GPU,因此不需要进行pin_memory处理
        # need to be pinned.
        self.token_ids_cpu_tensor = torch.zeros(
            (max_num_reqs, max_model_len),
            device="cpu",
            dtype=torch.int32,
            pin_memory=False,
        )
        self.token_ids_cpu = self.token_ids_cpu_tensor.numpy()                              #一个numpy视图,指向token_ids_cpu_tensor,本质作用:用numpy来操作token,比torch更轻量(CPU)
        self.is_token_ids_tensor = torch.zeros(
            (max_num_reqs, max_model_len), device="cpu", dtype=bool, pin_memory=False       #一个bool矩阵,用来表示这个位置是否真的有token
        )
        self.is_token_ids = self.is_token_ids_tensor.numpy()                                #torch → numpy view(共享内存)
        # Store prompt embeddings per request to avoid OOM from large upfront               #按请求分别存储prompt的embedding,避免在max_model_len很大时一次性分配过多内存导致OOM(溢出)
        # allocation if max_model_len is big.
        # Maps req_index -> tensor of shape (num_prompt_tokens, hidden_size)
        self.req_prompt_embeds: dict[int, torch.Tensor] = {}                                #一个字典:req_index → prompt 的 embedding,什么时候用？当输入不是token id而是已经算好的embedding时
        self.num_tokens_no_spec = np.zeros(max_num_reqs, dtype=np.int32)                    #每个request当前真实token数(不包含spec)
        self.num_prompt_tokens = np.zeros(max_num_reqs, dtype=np.int32)                     #每个request的输入长度
        self.num_computed_tokens_cpu_tensor = torch.zeros(                                  #每个request算了多少token:[req0算了多少token, req1算了多少token, req2算了多少token, ...]
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
            (max_num_reqs,), dtype=torch.float32, device=device                             #GPU上的temperature,每个request上1个,用于在采样前对logits 做缩放:logits/temperature
        )
        self.temperature_cpu_tensor = torch.empty(
            (max_num_reqs,), dtype=torch.float32, device="cpu", pin_memory=pin_memory       #GPU 上的 temperature,pin_memory=True 时可以加速 CPU → GPU 的拷贝,支持DMA
        )
        self.temperature_cpu = self.temperature_cpu_tensor.numpy()
        self.greedy_reqs: set[str] = set()                                                  #需要走greedy解码的集合,这些请求可以走更快路径argmax(logits)不需要采样
        self.random_reqs: set[str] = set()                                                  #需要走随机采样的请求集合(temprature>0),这些请求执行softmax+随机采样(top-k/top-p)

        self.top_p = torch.empty((max_num_reqs,), dtype=torch.float32, device=device)       #GPU上的top_p
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

        # TODO convert this to LogitsProcessor                                                  后续可以把这些逻辑统一抽象成LogitsProcessor(更模块化)
        self.has_allowed_token_ids: set[str] = set()                                            #记录那些request使用了allowed_token_ids限制(白名单约束),set里边村的是req_id(字符串)
        # NOTE(lufang): In the mask tensor, if the corresponding token allowed,
        # the value is False. Since we use masked_fill_ to set -inf.
        self.allowed_token_ids_mask: torch.Tensor | None = None                                 #token级别的约束(shape:[batch_size,vocab_size]),语义通常反直觉:False表示允许这个token,原因是后边会用logits.masked_fill(mask,-inf) true的地方会填成-inf
        self.allowed_token_ids_mask_cpu_tensor: torch.Tensor | None = None                      #上面mask的cpu版本

        # req_index -> bad_words_token_ids                                                      #每个request的bad words(黑名单),eg. { 0: [[10, 20], [30]]   # request 0 禁止出现 token序列 [10,20] 或 [30]}
        self.bad_words_token_ids: dict[int, list[list[int]]] = {}

        self.logits_processing_needs_token_ids = np.zeros(max_num_reqs, dtype=bool)             #shape[max_num_reqs]标记每个request的logits处理是否依赖已生成的token,True表示依赖如如 repetition penalty / bad words
                                                                                                #
        self.req_output_token_ids: list[list[int] | None] = []                                  #每个request当前已经生成的output token列表,index对应req_index

        # Store provided logitsprocs. If none are provided, initialize empty
        # data structure
        self.logitsprocs = logitsprocs or LogitsProcessors()                                    #如果没有传入 则初始化一个空的LogitsProcessors(避免后续判空)
        self.logitsprocs_need_output_token_ids = logitsprocs_need_output_token_ids              #标记这些logit processors是否依赖已生成的token. 如repetition penalty / bad words → 需要历史 token(True).  topk/tempture不需要

        # Store last speculative tokens for sampler.
        self.spec_token_ids: list[list[int]] = [[] for _ in range(max_num_reqs)]                #每个request最近一轮speculative decoding产生的token, shape[max_num_reqs][variable]  用于sampler判断哪些spec tokens被接收/拒绝

        # This is updated each time the batch constituents change.
        self.sampling_metadata = self._make_sampling_metadata()                                 #当前batch的采样元信息,每当batch发生变化(增删/移动request)需要重新构建

        # for pooling models
        self.pooling_params: dict[str, PoolingParams] = {}
        self.pooling_states: dict[str, PoolingStates] = {}

        # Cached reference to the GPU tensor of previously sampled tokens                       #上一轮采样结果缓存.上一轮采样得到的token(tensor形式) 用于避免重复拷贝 与当前batch对齐做更新
        self.prev_sampled_token_ids: torch.Tensor | None = None
        self.prev_req_id_to_index: dict[str, int] | None = None                                 #上一轮的req_id->index映射 因为batch会重排(index会变),需要这个来对齐旧数据到新位置
        # These are used to update output_token_ids with real sampled
        # ids from prior step, if required by current sampling params
        # (e.g. penalties).
        self.sampled_token_ids_cpu: torch.Tensor | None = None                                  #cpu侧token同步(给logits processor用),上一轮采样结果(cpu tensor版本)用于更新:output_token_ids, 给依赖历史token的logits processor(如penalty)使用
        self.async_copy_ready_event: torch.Event | None = None                                  #异步拷贝完成的事件(gpu->cpu) 用于确保sampled_token_ids_cpu已经准备好,避免阻塞GPU pipeline

    @property
    def req_ids(self) -> list[str]:
        # None elements should only be present transiently                                      #对外暴露当前batch中的request_id(按index对齐),这里有设计哲学存在先不管
        # while performing state updates to the batch.
        return cast(list[str], self._req_ids)

    def _register_add_request(self, request: "CachedRequestState") -> int:
        """Track add-request operations for logits processors.                                  #当有一个新请求进入batch时,决定这个请求在batch中的位置index,记录这次新增请求的信息(给logits processor用)
        Not applicable to pooling models.                                                       #为logits processors(逻辑处理器)追踪添加请求的操作,不适合pooling模型

        """
        # Fill the next empty index if there is one.                                            #找一个可用的位置,优先复用之前被删除的请求留下的空位,比如batch原来是[A,B,C]删除B后是[A,_,C],pop_removed()会返回index=1
        if (new_req_index := self.batch_update_builder.pop_removed()) is None:
            # Append to end otherwise.
            new_req_index = self.num_reqs                                                       #如果没有空位 就追加到末尾 比如当前有三个请求,index=3

        assert new_req_index < self.max_num_reqs                                                
    
        self.batch_update_builder.batch_changed = True                                          #标记本轮batch发生了变化(调度器会用到)
        
        if request.sampling_params:                                                             #记录新增请求的详细信息(用于logits processors)
            # Detailed added request metadata is only required for non-pooling
            # models, to support logitsprocs.
            self.batch_update_builder.added.append(                                             #只有生成模型才需要这些信息(tempture/top_p等)
                (
                    new_req_index,                                                              #新请求在batch中的索引
                    request.sampling_params,                                                    #采样参数
                    request.prompt_token_ids,                                                   #prompt的token ids
                    request.output_token_ids,                                                   #已经生成的ouput tokens
                )
            )

        return new_req_index                                                                    #返回这个请求在batch中的位置

    def add_request(
        self,
        request: "CachedRequestState",
    ) -> int:
        """
        将一个新的请求(或从 waiting 队列移入的请求)加入到当前 InputBatch 中。这是 InputBatch 中最核心的添加函数之一。每次有新请求完成 prefill(或被 scheduler 调度进入 decode 阶段)时,都会调用此方法。
        和 EngineCore.add_request 的区别:EngineCore只是把请求交给调度器(逻辑层),这里真正写入batch内存(数据层)
        返回值:该请求在当前 batch 中被分配的索引(req_index)
        """
        
        req_index = self._register_add_request(request)                                          #分配batch位置,返回一个索引

        req_id = request.req_id
        
        if req_index == len(self._req_ids):
            self._req_ids.append(req_id)
            self.req_output_token_ids.append(request.output_token_ids)
            self.spec_token_ids.append([])                                                        #speculative decoding 的 draft tokens 列表
        else:
            self._req_ids[req_index] = req_id                                                     #复用之前被删除请求留下的空位
            self.req_output_token_ids[req_index] = request.output_token_ids
            self.spec_token_ids[req_index].clear()

        self.req_id_to_index[req_id] = req_index                                                   #建立req_id -> index映射

        # Copy the prompt token ids and output token ids.                                          #计算prompt token数
        num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
            request.prompt_token_ids, request.prompt_embeds
        )
        self.num_prompt_tokens[req_index] = num_prompt_tokens
        
        start_idx = num_prompt_tokens                                                              #计算output token在序列中的位置:例prompt:[A,B,C] output:[D,E] ->start=3, end=5
        end_idx = start_idx + len(request.output_token_ids)
        
        
        if request.prompt_token_ids is not None:                                                   #写入prompt tokens
            self.token_ids_cpu[req_index, :num_prompt_tokens] = request.prompt_token_ids           #token_ids_cpu是一个二维矩阵 shape=[max_num_reqs, max_seq_len]
            self.is_token_ids[req_index, :num_prompt_tokens] = True
        else:
            self.is_token_ids[req_index, :num_prompt_tokens] = False
            
        
        if request.prompt_embeds is not None:                                                       #保存 prompt embeds(主要用于多模态或直接传入 embedding 的情况)
            self.req_prompt_embeds[req_index] = request.prompt_embeds
            
    
        self.token_ids_cpu[req_index, start_idx:end_idx] = request.output_token_ids                 #写入已生成的tokens
        self.is_token_ids[req_index, start_idx:end_idx] = True
        
        self.num_tokens_no_spec[req_index] = request.num_tokens                                     #不包含spec token的长度
        
    
        self.num_computed_tokens_cpu[req_index] = request.num_computed_tokens                       #计算已经完成的token数(影响kv cache/prefix)
       
        self.block_table.add_row(request.block_ids, req_index)                                      #把kv block绑定到这个请求 例request.block_ids = [10,11,12]

        if sampling_params := request.sampling_params:
            if self.is_spec_decode and is_spec_decode_unsupported(sampling_params):                 #spec decode不支持的请求
                self.spec_decode_unsupported_reqs.add(req_id)
            
            if sampling_params.sampling_type == SamplingType.GREEDY:
                # Should avoid division by zero later when apply_temperature                        # Greedy 解码(temperature=0)
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
            self.repetition_penalties_cpu[req_index] = (                                            #惩罚出现过的token
                sampling_params.repetition_penalty
            )
            if sampling_params.repetition_penalty != 1.0:
                self.repetition_penalties_reqs.add(req_id)

            # NOTE(woosuk): self.generators should not include the requests that                    随机数生成器
            # do not have their own generator.
            if request.generator is not None:
                self.generators[req_index] = request.generator

            if sampling_params.logprobs is not None:
                self.num_logprobs[req_id] = (
                    self.vocab_size
                    if sampling_params.logprobs == -1
                    else sampling_params.logprobs
                )

            if sampling_params.allowed_token_ids:                                                   #只允许生成某些token
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
        """This method must always be followed by a call to condense().                               从当前batch中移除一个请求,这里只是标记删除+清理数据,真正的内存压缩要靠后续condense()
        Args:
          req_id: request to remove
        Returns:
          Removed request index, or `None` if `req_id` not recognized
        
        example:
            当前batch: [0,1,2] [A,B,C]   remove_request("B")后,[A,None,C]
        """

        req_index = self.req_id_to_index.pop(req_id, None)                                            #找到请求位置 从req_id->index映射中删除,拿到index
        if req_index is None:                                                                         #如果不存在可能已经被删除过,直接返回
            return None

        self.batch_update_builder.removed_append(req_index)                                           #batch_update_builder记录这次删除(给scheduler/engine用)
        self._req_ids[req_index] = None                                                               #清掉req_id
        self.req_output_token_ids[req_index] = None                                                   #清掉已生成tokens
        self.spec_token_ids[req_index].clear()                                                        #spec decode的token清空

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

        self.greedy_reqs.discard(req_id)                                                                #清理采样状态
        self.random_reqs.discard(req_id)
        self.top_p_reqs.discard(req_id)
        self.top_k_reqs.discard(req_id)
        self.spec_decode_unsupported_reqs.discard(req_id)
        self.frequency_penalties_reqs.discard(req_id)
        self.presence_penalties_reqs.discard(req_id)
        self.repetition_penalties_reqs.discard(req_id)
        self.generators.pop(req_index, None)
        self.num_logprobs.pop(req_id, None)
        self.in_progress_prompt_logprobs_cpu.pop(req_id, None)                                          #删除中间状态
        if self.prev_req_id_to_index is not None:
            self.prev_req_id_to_index.pop(req_id, None)

        self.has_allowed_token_ids.discard(req_id)
        if self.allowed_token_ids_mask_cpu_tensor is not None:
            # False means we don't fill with -inf.
            self.allowed_token_ids_mask_cpu_tensor[req_index].fill_(False)
        self.bad_words_token_ids.pop(req_index, None)
        return req_index

    def swap_states(self, i1: int, i2: int) -> None:
        """
        交换batch中两个请求的位置(i1 - i2)
        example:  
        原始:
        index: 0    1    2
               A    B    C
        swap_states(0, 2) 后:
               C    B    A
        
        """
        
        old_id_i1 = self._req_ids[i1]
        old_id_i2 = self._req_ids[i2]
        self._req_ids[i1], self._req_ids[i2] = self._req_ids[i2], self._req_ids[i1]  # noqa             #交换req_id
        self.req_output_token_ids[i1], self.req_output_token_ids[i2] = (                                #交换已生成token列表
            self.req_output_token_ids[i2],
            self.req_output_token_ids[i1],
        )
        self.spec_token_ids[i1], self.spec_token_ids[i2] = (                                            #交换spec的token
            self.spec_token_ids[i2],
            self.spec_token_ids[i1],
        )
        assert old_id_i1 is not None and old_id_i2 is not None                                          #确保2个位置都有请求
        self.req_id_to_index[old_id_i1], self.req_id_to_index[old_id_i2] = (                            #更新req_id -> index映射
            self.req_id_to_index[old_id_i2],
            self.req_id_to_index[old_id_i1],
        )
        self.num_tokens_no_spec[i1], self.num_tokens_no_spec[i2] = (                                    #交换token总数
            self.num_tokens_no_spec[i2],
            self.num_tokens_no_spec[i1],
        )
        self.num_prompt_tokens[i1], self.num_prompt_tokens[i2] = (                                      #交换prompt token数
            self.num_prompt_tokens[i2],
            self.num_prompt_tokens[i1],
        )
        self.num_computed_tokens_cpu[i1], self.num_computed_tokens_cpu[i2] = (                          #交换已经计算完成的token数
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

        # Swap prompt embeddings if they exist                                                          #交换多模态
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

        self.block_table.swap_row(i1, i2)                                                               #交换kv cache block

        self.request_lora_mapping[i1], self.request_lora_mapping[i2] = (
            self.request_lora_mapping[i2],
            self.request_lora_mapping[i1],
        )

        if self.is_pooling_model:
            # Sampling and logits parameters don't apply to pooling models.
            return

        # For autoregressive models, track detailed request reordering info
        # to support logitsprocs.
        self.batch_update_builder.moved.append((i1, i2, MoveDirectionality.SWAP))                       #记录交换事件

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

        swap_dict_values(self.generators, i1, i2)                                                      #交换generators
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
        """Slide non-empty requests down into lower, empty indices.                                     压缩当前InputBatch,将有效请求[向前滑动],填补被remove_request留下的空洞

        Any consecutive empty indices at the very end of the list are not  filled.                      尾部连续空位不会填充(会直接裁剪掉)
                                                                                                 
        Returns:
          swaps: list of (from,to) swap tuples for moved requests
          empty_req_indices: indices not filled by condensation 
        example:
            初始 batch(num_reqs = 2,但内部数组长度=4):
            index:   0     1      2     3
                    A   None     B   None
            removed = [1, 3]
            目标(condense 后)
            index:   0     1
                     A     B
        """
        num_reqs = self.num_reqs                                                                        #当前有效请求数

        if not (empty_req_indices := self.batch_update_builder.removed):                                #没有空洞说明:要么没有删除请求 要么删除的请求已经被新加入的请求完美替换,此时无需condense 直接返回
            # All removed requests were replaced by added requests, or else no
            # requests were removed at all. No condense() needed
            return
        if num_reqs == 0:                                                                               #没有任何请求,清空所有数据结构
            # The batched states are empty.
            self._req_ids.clear()
            self.req_output_token_ids.clear()
            self.spec_token_ids.clear()
            return

        # NOTE(woosuk): This function assumes that the empty_req_indices                                #计算最后一个可能有数据的位置
        # is sorted in descending order.                                                                例num_reqs=2,removed=2 ——> last= 2+2-1=3
        last_req_index = num_reqs + len(empty_req_indices) - 1
        
        # ==================== 核心压缩循环 ====================                                        从后往前处理,把后面的有效请求往前移动,填补前面的空位
        while empty_req_indices:                                                                        #只要还有空洞就继续
            # Find the largest non-empty index.                                                         找到当前最大的非空索引(即还有有效请求的位置)
            while last_req_index in empty_req_indices:                                                  #跳过空洞(例3是空, 变成2)
                last_req_index -= 1

            # Find the smallest empty index.                                                            找到当前最小的空位索引(需要被填补的位置)
            empty_index = self.batch_update_builder.peek_removed()
            assert empty_index is not None
            
            if empty_index >= last_req_index:                                                           #如果空位已经在最后有效元素之后,不用填
                break
            
            # Move active request down into empty request                                               #移除这个空位
            # index.
            self.batch_update_builder.pop_removed()
            
            req_id = self._req_ids[last_req_index]                                                      #取出后面的有效请求
            output_token_ids = self.req_output_token_ids[last_req_index]
            assert req_id is not None
            
                                                                                                        # 更新各种数据结构中的位置
            self._req_ids[empty_index] = req_id                                                         #移动req_id
            self._req_ids[last_req_index] = None
            self.req_output_token_ids[empty_index] = output_token_ids
            self.req_output_token_ids[last_req_index] = None
            self.req_id_to_index[req_id] = empty_index                                                  # 更新请求ID到索引的映射

            
            num_tokens = self.num_tokens_no_spec[last_req_index] + len(
                self.spec_token_ids[last_req_index]
            )
            
            (self.spec_token_ids[last_req_index], self.spec_token_ids[empty_index]) = (                 # 交换 spec_token_ids(使用 tuple 解包方式交换)
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
            if self.is_pooling_model:
                last_req_index -= 1
                # Sampling state not used by pooling models.
                continue
                
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

            # Decrement last_req_index since it is now empty.                                               # 当前 last_req_index 位置已被清空,向前移动指针
            last_req_index -= 1

        # Trim lists to the batch size.
        del self._req_ids[num_reqs:]                                                                        #剪裁尾部
        del self.req_output_token_ids[num_reqs:]
        del self.spec_token_ids[num_reqs:]

    def refresh_metadata(self):
        """Apply any batch updates to sampling metadata.                                                    将batch结构变化(add/remove/swap/condense)同步到logits/sampling相关状态
        Example:
        假设当前 batch 为:
        index: 0  1  2
        req:   A  B  C

        sampling:
        A: temperature=0.7
        B: temperature=1.0
        C: temperature=0.2

        此时 scheduler 发生 batch 变化:
        - remove B
        - swap A 和 C

        变化后 batch 变为:
        index: 0  1
        req:   C  A

        问题:
        sampling/logits processor 原本是按“index → request”绑定的,现在 index 对应关系已经变化:
        - index 0 从 A 变成 C
        - index 1 从 B(已删除) 变成 A
        如果不更新 metadata:GPU decode 时会出现 sampling 参数错位(例如 C 使用 A 的 temperature)
        因此 refresh_metadata 的作用是:
        - 将 batch_update(add/remove/swap/condense)同步到 logits processors
        - 重建 sampling_metadata,使 index ↔ request ↔ sampling参数重新对齐
        """

        if self.is_pooling_model:
            batch_changed = self.batch_update_builder.reset()                                               #是否发生过 add/remove/swap
            if batch_changed:
                self.sampling_metadata = self._make_sampling_metadata()
            return

        # For non-pooling models - generate and apply logitsprocs update;                                   #非pooling模型,从batch_update_builder取出本轮所有变化,并清空记录
        # reset batch update tracking.                                                                      #如add A , remove B , swap(0,1)
        # Update sampling metadata if batch state is changed.
        batch_update = self.batch_update_builder.get_and_reset(self.num_reqs)

        for logit_proc in self.logitsprocs.all:                                                             #logits processors更新,logits processor需要知道哪些request被移动/删除/新增
            logit_proc.update_state(batch_update)
            

        if batch_update:                                                                                    #是否要刷新samling metadate
            self.sampling_metadata = self._make_sampling_metadata()                                         #sampling_metadata 包含:temperature topp topk   logits proceesor context

    def _make_sampling_metadata(self) -> SamplingMetadata:
        """
        根据当前 batch 的索引顺序，把所有 request 的采样状态“重新打包成 GPU 可用的连续张量
        """
        num_reqs = self.num_reqs
        if not self.all_greedy:
            temperature = copy_slice(
                self.temperature_cpu_tensor, self.temperature, num_reqs
            )
        else:
            temperature = None                                                                              #如果全部greedy, tempterature恒等于0,不需要传GPU(省内存+计算)
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
        """把ragged的prompt token(不等长)->padding成(batch_size * max_len)的tensor"""
        num_reqs = self.num_reqs
        max_prompt_len = self.num_prompt_tokens[:num_reqs].max()
        prompt_token_ids_cpu_tensor = torch.empty(                                                          #创建CPU tensor(未初始化)
            (self.num_reqs, max_prompt_len),
            device="cpu",
            dtype=torch.int64,
            pin_memory=self.pin_memory,
        )
        prompt_token_ids = prompt_token_ids_cpu_tensor.numpy()                                              #转Numpy(为了快速写入 目的是用numpy做快速batch copy,比python loop快)
        prompt_token_ids[:] = self.token_ids_cpu[:num_reqs, :max_prompt_len]                                #拷贝真实数据
        # Use the value of vocab_size as a pad since we don't have a
        # token_id of this value.
        for i in range(num_reqs):
            prompt_token_ids[i, self.num_prompt_tokens[i] :] = self.vocab_size                              #vocab_size 不可能是合法 token id,可以当mask用                      
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
        In async scheduling case, store ref to sampled_token_ids_cpu                                           在异步调度模式下,保存sampled_token_ids_cpu张量的引用以及对应的拷贝完成事件
        tensor and corresponding copy-ready event. Used to repair                                              当logits processor需要时,可在下一次采样前用这些信息来修正(补全)output_token_ids
        output_token_ids prior to sampling, if needed by logits processors.
        
        Example:
            1. GPU 采样完成，产生最新的 token (id: 42)
            2. 启动异步拷贝: sampled_token_ids_cpu.copy_(gpu_tensor, non_blocking=True)
            3. 记录事件: async_copy_ready_event.record()
            4. 注册到 Batch:
            batch.set_async_sampled_token_ids(sampled_token_ids_cpu, async_copy_ready_event)
            
             --- 随后在 LogitsProcessor 中 ---
            如果需要计算重复惩罚，它会执行:
            batch.async_copy_ready_event.synchronize() # 等待拷贝完成
            full_history = batch.output_token_ids + batch.sampled_token_ids_cpu
            这样就能拿到最新的 42,即使它还没正式进入 output_token_ids 列表。
        """
        if self.sampling_metadata.output_token_ids:
            self.sampled_token_ids_cpu = sampled_token_ids_cpu
            self.async_copy_ready_event = async_copy_ready_event
        else:
            self.sampled_token_ids_cpu = None
            self.async_copy_ready_event = None

    def update_async_output_token_ids(self) -> None:
        """
        In async scheduling case, update output_token_ids in sampling metadata                                 在异步模式下,用上一轮采样得到的token(sampled_token_ids)去填充当前output_token_ids中的占位符-1
        from prior steps sampled token ids once they've finished copying to CPU.                               -1是槽位值
        This is called right before they are needed by the logits processors.                             

        Example:
            GPU sampling 得到:
            sampled_token_ids_cpu = [[42], [17]]
            当前 step:  index 0 → reqB  index 1 → reqA
            sampling_metadata.output_token_ids: reqB → [201, 202, -1]   reqA → [101, 102, -1]
            
            执行 update_async_output_token_ids:
            sampled_token_ids[1] = 17   [201, 202, -1] → [201, 202, 17]
            sampled_token_ids[0] = 42  [101, 102, -1] → [101, 102, 42]
        """
        
        output_token_ids = self.sampling_metadata.output_token_ids                                              #eg: [[101,102,-1],[201,202,-1]]
        
        if self.sampled_token_ids_cpu is None or not output_token_ids:                                          #如果没有异步采样结果(说明不是async模式) 当前batch不需要output_token_ids(比如没有penalty/logits processor)
            # Output token ids not needed or not async scheduling.      
            return                                                      

        assert self.prev_req_id_to_index is not None
        sampled_token_ids = None                                        
        
        # ==================== 核心循环:遍历当前 batch 中的所有请求 ====================
        for index, req_id in enumerate(self.req_ids):
            prev_index = self.prev_req_id_to_index.get(req_id)                                                  #找到这个请求在上一轮的位置
            if prev_index is None:                                                                              #说明这是新请求,没有历史sampled token不需要处理
                continue                                                
            
            req_output_token_ids = output_token_ids[index]                                                      #当前request的output_token_ids(引用) 例如reqA->[101,102,-1]

            if not req_output_token_ids or req_output_token_ids[-1] != -1:
                # Final output id is not a placeholder, some tokens must have                                   #如果没有token或最后一个不是-1, 说明已经被填过了,?这个知道长度？
                # been discarded after a kv-load failure.
                continue
            
            if sampled_token_ids is None:
                assert self.async_copy_ready_event is not None
                self.async_copy_ready_event.synchronize()                                                       #等待GPU->CPU异步拷贝完成,这是唯一的阻塞点(尽量推迟)
                sampled_token_ids = self.sampled_token_ids_cpu.squeeze(-1).tolist()                             #eg.tensor([[42],[17]]) → [42,17]
            req_output_token_ids[-1] = sampled_token_ids[prev_index]                                            #原 [101, 102, -1]  sampled_token_ids[prev_index=0] = 42  → 变成: [101, 102, 42]

    @property
    def num_reqs(self) -> int:                                                                                  #当前batch中活跃request的数量
        return len(self.req_id_to_index)

    @property
    def all_greedy(self) -> bool:                                                                               #是否全是greedy解码
        return len(self.random_reqs) == 0

    @property
    def all_random(self) -> bool:
        return len(self.greedy_reqs) == 0                                                                       #是否全是随机采样

    @property
    def no_top_p(self) -> bool:
        return len(self.top_p_reqs) == 0                                                                        #是否所有请求都不适用top-p

    @property
    def no_top_k(self) -> bool:                                                                                 #是否所有请求都不用top-l
        return len(self.top_k_reqs) == 0                                                                        

    @property
    def no_penalties(self) -> bool:
        return (                                                                                                #是否所有请求都不用penalty(无repetition/frequency/presence)
            len(self.presence_penalties_reqs) == 0                                                              #用于快速跳过penalty kernel/计算
            and len(self.frequency_penalties_reqs) == 0
            and len(self.repetition_penalties_reqs) == 0
        )

    @property
    def max_num_logprobs(self) -> int | None:                                                                   #当前batch中要求的最大logprob数 例req1:5  req2:10
        return max(self.num_logprobs.values()) if self.num_logprobs else None                                   #每个request要求返回前k个候选token的logprob

    @property
    def no_allowed_token_ids(self) -> bool:
        return len(self.has_allowed_token_ids) == 0                                                             #是否所有request都没有token white-list限制
