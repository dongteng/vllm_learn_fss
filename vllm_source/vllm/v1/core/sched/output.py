# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

from typing_extensions import deprecated

from vllm._bc_linter import bc_linter_include

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import torch

    from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorMetadata
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
    from vllm.lora.request import LoRARequest
    from vllm.multimodal.inputs import MultiModalFeatureSpec
    from vllm.pooling_params import PoolingParams
    from vllm.sampling_params import SamplingParams
    from vllm.v1.request import Request
else:
    ECConnectorMetadata = object
    KVConnectorMetadata = object
    LoRARequest = object
    MultiModalFeatureSpec = object
    PoolingParams = object
    SamplingParams = object
    Request = object


@bc_linter_include
@dataclass
class NewRequestData:
    req_id: str
    prompt_token_ids: list[int] | None
    mm_features: list[MultiModalFeatureSpec]
    sampling_params: SamplingParams | None
    pooling_params: PoolingParams | None
    block_ids: tuple[list[int], ...]
    num_computed_tokens: int
    lora_request: LoRARequest | None
    prompt_embeds: "torch.Tensor | None" = None

    # Only used for v2 model runner.
    prefill_token_ids: list[int] | None = None

    @classmethod
    def from_request(
        cls,
        request: Request,
        block_ids: tuple[list[int], ...],
        prefill_token_ids: list[int] | None = None,
    ) -> "NewRequestData":
        return cls(
            req_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            mm_features=request.mm_features,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            block_ids=block_ids,
            num_computed_tokens=request.num_computed_tokens,
            lora_request=request.lora_request,
            prompt_embeds=request.prompt_embeds,
            prefill_token_ids=prefill_token_ids,
        )

    def __repr__(self) -> str:
        prompt_embeds_shape = (
            self.prompt_embeds.shape if self.prompt_embeds is not None else None
        )
        return (
            f"NewRequestData("
            f"req_id={self.req_id},"
            f"prompt_token_ids={self.prompt_token_ids},"
            f"prefill_token_ids={self.prefill_token_ids},"
            f"mm_features={self.mm_features},"
            f"sampling_params={self.sampling_params},"
            f"block_ids={self.block_ids},"
            f"num_computed_tokens={self.num_computed_tokens},"
            f"lora_request={self.lora_request},"
            f"prompt_embeds_shape={prompt_embeds_shape}"
            ")"
        )

    # Version of __repr__ with the prompt data obfuscated
    def anon_repr(self) -> str:
        prompt_token_ids_len = (
            len(self.prompt_token_ids) if self.prompt_token_ids is not None else None
        )
        prompt_embeds_shape = (
            self.prompt_embeds.shape if self.prompt_embeds is not None else None
        )
        return (
            f"NewRequestData("
            f"req_id={self.req_id},"
            f"prompt_token_ids_len={prompt_token_ids_len},"
            f"mm_features={self.mm_features},"
            f"sampling_params={self.sampling_params},"
            f"block_ids={self.block_ids},"
            f"num_computed_tokens={self.num_computed_tokens},"
            f"lora_request={self.lora_request},"
            f"prompt_embeds_shape={prompt_embeds_shape}"
            ")"
        )


@bc_linter_include
@dataclass
class CachedRequestData:
    """
    SchedulerOutput 中用于描述“已经调度过（cached）的请求”的数据容器。
    它的核心作用是：在连续调度步骤中，只传递请求的**增量变化（diff）**，而不需要每次都把请求的完整信息重新发送给 Worker。
    这大大降低了进程间通信开销，是 vLLM v1 高吞吐的重要优化之一。
    """
    req_ids: list[str]
    # For request ids not in resumed_req_ids, new_block_ids will be appended to
    # the request's block IDs. For those in the set, new_block_ids will be used as the
    # request's block IDs instead of appending to the existing block IDs.
    resumed_req_ids: set[str]           #本轮从抢占（preemption）中恢复的请求ID集合 重要区别：如果req_id不在resumed_req_ids中，new_block_ids 是追加（append）到原有 block_ids 后面（正常 decode）; 若
    # NOTE(woosuk): new_token_ids is only used for pipeline parallelism.               如果req_id 在 resumed_req_ids 中 → new_block_ids 将**替换**原有的 block_ids（因为抢占后 KV cache 被清空或移动，需要重新分配）
    # When PP is not used, new_token_ids will be empty.
    new_token_ids: list[list[int]]      #上一步采样得到的新token列表（每个请求对应一个list） 注意：仅在PP模式下有意义，当不使用PP时，这个字段为空列表，因为非最后一个rank无法直接采样
    # For requests not scheduled in the last step, propagate the token ids to the
    # connector. Won't contain requests that were scheduled in the prior step.
    all_token_ids: dict[str, list[int]] #请求的完整output token ids（主要用于异步调度asynv scheduling），特别用于那些上一轮未被调度，但本轮恢复的请求。作用：把完整的 token 序列传递给 KV Connector 或恢复请求状态，确保 input_ids 计算正确。
    new_block_ids: list[tuple[list[int], ...] | None] #本轮新分配的kv cache block ids, 每个元素对应一个请求：正常情况：是一个 tuple，里面包含该请求每个 logical block 对应的新 physical block ids；None表示该请求本轮没有新的Block分配
    num_computed_tokens: list[int]      #每个请求当前已完成前向计算的token总数（prompt + 已接受的 output tokens），这个值对构造 input_ids 和 attention metadata 至关重要
    num_output_tokens: list[int]        #每个请求当前已经生成的 output token 数量（不含 speculative tokens），用于对齐cachede states中的output_token_ids

    @property
    def num_reqs(self) -> int:
        """
        返回本轮已缓存请求的数量
        """
        return len(self.req_ids)

    @cached_property
    @deprecated("This will be removed in v0.14, use `resumed_req_ids` instead.")
    def resumed_from_preemption(self) -> list[bool]:
        """
        已废弃（将在 v0.14 移除）。
        返回每个请求是否是从抢占中恢复的布尔列表。
        请改用 `resumed_req_ids` 集合进行判断。
        """
        return [req_id in self.resumed_req_ids for req_id in self.req_ids]

    @cached_property
    @deprecated("This will be removed in v0.14, use `all_token_ids` instead.")
    def resumed_req_token_ids(self) -> list[list[int] | None]:
        """
        已废弃（将在 v0.14 移除）。
        返回从抢占中恢复的请求的完整 token ids 列表。
        请直接使用 `all_token_ids` 字典。
        """
        return [
            self.all_token_ids[req_id] if req_id in self.resumed_req_ids else None
            for req_id in self.req_ids
        ]

    @classmethod
    def make_empty(cls) -> "CachedRequestData":
        """创建一个空的 CachedRequestData 对象，用于空 batch 等场景"""
        return cls(
            req_ids=[],
            resumed_req_ids=set(),
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=[],
            num_computed_tokens=[],
            num_output_tokens=[],
        )


@bc_linter_include
@dataclass
class SchedulerOutput:
    """
    Scheduler每次调度后返回给ModelRunner/Worker的核心输出对象。它描述了本轮应该执行什么：哪些请求要跑每个请求跑多少token，是否有投机解码、那些请求已经完成等
    这个对象是 EngineCore 和 Worker 之间通信的重要桥梁。
    """
    # list of the requests that are scheduled for the first time.
    # We cache the request's data in each worker process, so that we don't
    # need to re-send it every scheduling step.
    scheduled_new_reqs: list[NewRequestData]                              #本轮第一次被调度的请求
    # list of the requests that have been scheduled before.
    # Since the request's data is already cached in the worker processes, #已经调度过的请求列表
    # we only send the diff to minimize the communication cost.           #因为请求的核心数据已经在 Worker 进程中缓存，所以这里只发送“差异部分”（如 num_computed_tokens、新分配的 block_ids、新采样的 token 等），大幅降低进程间通信开销。
    scheduled_cached_reqs: CachedRequestData

    # req_id -> num_scheduled_tokens  {req_id : num_scheduled_tokens}
    # Number of tokens scheduled for each request.
    num_scheduled_tokens: dict[str, int]                               #本轮调度中每个请求要处理的token数量，prefill阶段可能一次性几十击败，decode阶段1（投机解码是多个）
    # Total number of tokens scheduled for all requests.
    # Equal to sum(num_scheduled_tokens.values())
    total_num_scheduled_tokens: int                                    #本轮所有请求计划处理的 token 总数（即上面字典所有值的和）
    # req_id -> spec_token_ids
    # If a request does not have any spec decode tokens, it will not be
    # included in the dictionary.
    scheduled_spec_decode_tokens: dict[str, list[int]]                 #每个请求本轮携带的draft token列表，如果某个请求没有投机tokens，则不会出现在这个字典中
    # req_id -> encoder input indices that need processing.
    # E.g., if a request has [0, 1], it could mean the vision encoder needs
    # to process that the request's 0-th and 1-th images in the current step.
    scheduled_encoder_inputs: dict[str, list[int]]                      #多模态模型专用：本轮需要处理的encoder输入索引，例如一个请求有多个图片，这里记录质本轮要跑的encoder的图片序号（如第0、1张图）
    # Number of common prefix blocks for all requests in each KV cache group.
    # This can be used for cascade attention.
    num_common_prefix_blocks: list[int]                                 #所有请求共有的 prefix block 数量（y用于级联注意力优化）。

    # Request IDs that are finished in between the previous and the current
    # steps. This is used to notify the workers about the finished requests
    # so that they can free the cached states for those requests.
    finished_req_ids: set[str]                                          #在上一次调度和本次调度之间已经完成的请求 ID 集，用于通知Worker:这些请求已经结束，可以释放对应的cached statesself.requests）和 persistent batch 中的条目，回收资源。
    # list of mm_hash strings associated with the encoder outputs to be
    # freed from the encoder cache.
    free_encoder_mm_hashes: list[str]                                   #多模态encoder输出缓存需要释放的mm_hash列表，当多模态请求完成后，释放对应的 vision encoder 输出缓存，防止显存泄漏。

    # Request IDs that are preempted in this step.  本次调度中被抢占的请求ID
    # Only used for v2 model runner. 只在v2 model runner中使用
    preempted_req_ids: set[str] | None = None                           #本次调度中被抢占（preempted）的请求 ID 集合，仅在 v2 ModelRunner 中使用（v1 中主要通过 unscheduled_req_ids 处理抢占逻辑）

    # Whether the scheduled requests have all the output tokens they
    # need to perform grammar bitmask computation.
    pending_structured_output_tokens: bool = False                      #“当前这一批已经被调度（scheduled）的请求，它们生成的输出 token 是否已经足够完整，可以开始做下一阶段的 ‘grammar bitmask’ 计算了？”

    # KV Cache Connector metadata.
    kv_connector_metadata: KVConnectorMetadata | None = None            #kv cache跨节点 跨设备传输相关的元数据（用于分布式kv cache 传递场景）

    # EC Cache Connector metadata
    ec_connector_metadata: ECConnectorMetadata | None = None            #Encoder Cache Connector 元数据（多模态 encoder cache 传输相关）

    @classmethod
    def make_empty(cls) -> "SchedulerOutput":
        return cls(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[],
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
        )


@dataclass
class GrammarOutput:
    # ids of structured output requests.
    structured_output_request_ids: list[str]
    # Bitmask ordered as structured_output_request_ids.
    grammar_bitmask: "npt.NDArray[np.int32]"
