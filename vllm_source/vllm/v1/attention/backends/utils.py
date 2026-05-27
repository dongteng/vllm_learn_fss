# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import abc
import enum
import functools
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field, fields, make_dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Literal,
    Protocol,
    TypeVar,
    get_args,
)

import numpy as np
import torch
from typing_extensions import deprecated, runtime_checkable

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.utils.math_utils import cdiv

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import InputBatch

import vllm.envs as envs
from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.utils import (
    get_kv_connector_cache_layout,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.ubatch_utils import UBatchSlice

logger = init_logger(__name__)
KVCacheLayoutType = Literal["NHD", "HND"]                                                          #本质事head维和sequence/token维谁放在前面
_KV_CACHE_LAYOUT_OVERRIDE: KVCacheLayoutType | None = None

PAD_SLOT_ID = -1


def is_valid_kv_cache_layout(value: str) -> bool:
    return value in get_args(KVCacheLayoutType)


@dataclass
class CommonAttentionMetadata:
    """
    Per-batch attention metadata, shared across layers and backends.                                每个batch的注意力元数据,在各层和后端之间共享
    AttentionMetadataBuilder instances use it to construct per-layer metadata.                      AttentionMetadataBuilder实例使用它来构建每层的元数据

    For many of the tensors we keep both GPU and CPU versions.                                      对于许多张量,我们同时保留GPU和CPU版本
    """

    query_start_loc: torch.Tensor
    query_start_loc_cpu: torch.Tensor                                                               #eg 假设3个请求的长度分别是2,3,1  则query_start_loc_cpu = [0, 2, 5, 6]
    """(batch_size + 1,), the start location of each request in query Tensor"""                     #(batch+1,)每个请求在query tensor中的起始位置,用于定位每个请求的slice。+1是因为需要一个结束位置

    seq_lens: torch.Tensor
    """(batch_size,), the number of computed tokens for each request"""                             #每个请求已经计算的token数量,用于记录当前decode进度

    num_reqs: int
    """Number of requests"""
    # TODO(lucas): rename to num_tokens since it may be padded and this is misleading
    num_actual_tokens: int
    """Total number of tokens in batch"""
    max_query_len: int
    """Longest query in batch"""
    max_seq_len: int
    """Longest context length (may be an upper bound)"""

    block_table_tensor: torch.Tensor                                                                #用于记录每个request的token对应到哪些cache block
    slot_mapping: torch.Tensor                                                                      #token到kv cache slot的映射关系, attention/kernel会根据它找到token在cache中的实际存储位置

    causal: bool = True                                                                             #是否使用因果注意力, True表示只能看到当前token之前的内容,符合自回归生成

    # Needed by FastPrefillAttentionBuilder
    logits_indices_padded: torch.Tensor | None = None                                               #需要计算logits的token索引(padding版本)
    num_logits_indices: int | None = None                                                           #实际logits索引数量

    # Needed by CrossAttentionBuilder
    encoder_seq_lens: torch.Tensor | None = None                                                    #encoder侧每个request的序列长度
    encoder_seq_lens_cpu: np.ndarray | None = None

    dcp_local_seq_lens: torch.Tensor | None = None                                                  #dcp中当前rank的本地序列长度 即当前并行rank实际负责处理的sequence长度
    dcp_local_seq_lens_cpu: torch.Tensor | None = None
    """Sequence lengths of the local rank in decode context parallelism world"""

    # WARNING: Deprecated fields. Will be removed in a future release (v0.14.0)
    _seq_lens_cpu: torch.Tensor | None = None
    _num_computed_tokens_cpu: torch.Tensor | None = None

    @property
    @deprecated(
        """
    Prefer using device seq_lens directly to avoid implicit H<>D sync.
    If a CPU copy is needed, use `seq_lens.cpu()` instead.
    Will be removed in a future release (v0.14.0)
    """
    )
    def seq_lens_cpu(self) -> torch.Tensor:
        if self._seq_lens_cpu is None:
            self._seq_lens_cpu = self.seq_lens.to("cpu")
        return self._seq_lens_cpu

    @property
    @deprecated(
        """
    Prefer using device seq_lens directly to avoid implicit H<>D sync which breaks full
    async scheduling. If a CPU copy is needed, it can be derived from 
    query_start_loc_cpu and seq_lens.
    Will be removed in a future release (v0.14.0)
    """
    )
    def num_computed_tokens_cpu(self) -> torch.Tensor:
        if self._num_computed_tokens_cpu is None:
            query_seq_lens = (
                self.query_start_loc_cpu[1:] - self.query_start_loc_cpu[:-1]
            )
            self._num_computed_tokens_cpu = self.seq_lens_cpu - query_seq_lens
        return self._num_computed_tokens_cpu

    # TODO(lucas): remove once we have FULL-CG spec-decode support                                    #一旦我们完全支持full-cg spec-decode  这个方法可以删除
    def unpadded(
        self, num_actual_tokens: int, num_actual_reqs: int
    ) -> "CommonAttentionMetadata":
        """
        返回一个去掉padding后的CommonAttentionMetadata副本,用于只保留实际token和实际request的信息
        而不是包含padding的大tensor
        
        """
        
        maybe_slice_reqs = lambda x: x[:num_actual_reqs] if x is not None else None
        return CommonAttentionMetadata(
            query_start_loc=self.query_start_loc[: num_actual_reqs + 1],
            query_start_loc_cpu=self.query_start_loc_cpu[: num_actual_reqs + 1],
            seq_lens=self.seq_lens[:num_actual_reqs],
            _seq_lens_cpu=self._seq_lens_cpu[:num_actual_reqs]
            if self._seq_lens_cpu is not None
            else None,
            _num_computed_tokens_cpu=self._num_computed_tokens_cpu[:num_actual_reqs]
            if self._num_computed_tokens_cpu is not None
            else None,
            num_reqs=num_actual_reqs,
            num_actual_tokens=num_actual_tokens,
            max_query_len=self.max_query_len,
            max_seq_len=self.max_seq_len,
            block_table_tensor=self.block_table_tensor[:num_actual_reqs],
            slot_mapping=self.slot_mapping[:num_actual_tokens],
            causal=self.causal,
            logits_indices_padded=self.logits_indices_padded,
            num_logits_indices=self.num_logits_indices,
            encoder_seq_lens=maybe_slice_reqs(self.encoder_seq_lens),
            encoder_seq_lens_cpu=maybe_slice_reqs(self.encoder_seq_lens_cpu),
            dcp_local_seq_lens=maybe_slice_reqs(self.dcp_local_seq_lens),
            dcp_local_seq_lens_cpu=maybe_slice_reqs(self.dcp_local_seq_lens_cpu),
        )


def slice_query_start_locs(
    query_start_loc: torch.Tensor,
    request_slice: slice,
) -> torch.Tensor:
    """
    Creates a new query_start_loc that corresponds to the requests in                                   创建一个新的query_start_loc 它值对应request_slice指定范围的请求
    request_slice.

    Note: This function creates a new tensor to hold the new query_start_locs.                          这个函数会创建一个新的tensor来保存新的query_start_loc
    This will break cudagraph compatibility.                                                            因此会破坏cudagraph兼容性(因为cudagraph要求tensor地址固定)
    
    eg.
    原始query_start_loc=[0,2,5,9,12]  表示req0:[0:2]   req1:[2:5] req2:[5:9] req3:[9:12]
    现在request_slice= slice(1,3)   意思是取req1 req2,  则先切出来[2,5,9] 在减去起始位置2
    就得到最新的query_start_loc = [0,3,7]
    
    """
    return (
        query_start_loc[request_slice.start : request_slice.stop + 1]
        - query_start_loc[request_slice.start]
    )


def _make_metadata_with_slice(
    ubatch_slice: UBatchSlice, attn_metadata: CommonAttentionMetadata
) -> CommonAttentionMetadata:
    """
    This function creates a new CommonAttentionMetadata that corresponds to                             根据ubatch_slice从原始attn_metadata中切除一个新的CommnAttentionMetadata
    the requests included in ubatch_slice                                                               这函数主要用于 将一个大batch切成多个ubatch(micro-batch时)为每个ubatch构建对应的attention metadata
    
    """

    assert not ubatch_slice.is_empty(), f"Ubatch slice {ubatch_slice} is empty"

    request_slice = ubatch_slice.request_slice
    token_slice = ubatch_slice.token_slice

    start_locs = attn_metadata.query_start_loc_cpu
    first_req = request_slice.start
    first_tok = token_slice.start
    last_req = request_slice.stop - 1
    last_tok = token_slice.stop - 1

    assert start_locs[first_req] <= first_tok < start_locs[first_req + 1], (
        "Token slice start outside of first request"
    )
    # NOTE: last token can be outside of the last request if we have CG padding.

    # If the request is split across ubatches, we have to adjust the metadata.
    # splits_first_request: The first request in this slice is the continuation of
    #                       a request that started in a previous slice.
    # splits_last_request:  The last request in this slice continues into the
    #                       next slice.
    splits_first_request = first_tok > start_locs[first_req]
    splits_last_request = last_tok < start_locs[last_req + 1] - 1

    query_start_loc_cpu = slice_query_start_locs(start_locs, request_slice)
    query_start_loc = slice_query_start_locs(
        attn_metadata.query_start_loc, request_slice
    )

    assert len(query_start_loc) >= 2, (
        f"query_start_loc must have at least 2 elements, got {len(query_start_loc)}"
    )

    if splits_first_request:
        tokens_skipped = first_tok - start_locs[first_req]
        query_start_loc[1:] -= tokens_skipped
        query_start_loc_cpu[1:] -= tokens_skipped
    seq_lens = attn_metadata.seq_lens[request_slice]
    seq_lens_cpu = attn_metadata.seq_lens_cpu[request_slice]

    if splits_last_request:
        # NOTE: We use start_locs (the original query_start_loc_cpu) to calculate
        # the tokens skipped because query_start_loc_cpu might have been modified
        # if splits_first_request is True.
        tokens_skipped = start_locs[last_req + 1] - token_slice.stop
        query_start_loc[-1] -= tokens_skipped
        query_start_loc_cpu[-1] -= tokens_skipped

        # Make sure we don't modify the seq_lens tensors
        #  (not cudagraph compatible)
        seq_lens = seq_lens.clone()
        seq_lens_cpu = seq_lens_cpu.clone()
        seq_lens[-1] -= tokens_skipped
        seq_lens_cpu[-1] -= tokens_skipped

    max_seq_len = int(seq_lens_cpu.max())
    num_computed_tokens_cpu = attn_metadata.num_computed_tokens_cpu[request_slice]

    num_requests = request_slice.stop - request_slice.start
    num_actual_tokens = token_slice.stop - token_slice.start
    max_query_len = int(
        torch.max(torch.abs(query_start_loc_cpu[1:] - query_start_loc_cpu[:-1])).item()
    )

    # This is to account for the case where we are in a dummy
    # run and query_start_loc_cpu is full of 0s
    if max_query_len == 0:
        max_query_len = attn_metadata.max_query_len

    block_table_tensor = attn_metadata.block_table_tensor[request_slice]
    slot_mapping = attn_metadata.slot_mapping[token_slice]

    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens,
        num_reqs=num_requests,
        num_actual_tokens=num_actual_tokens,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
        _seq_lens_cpu=seq_lens_cpu,
        _num_computed_tokens_cpu=num_computed_tokens_cpu,
    )


def split_attn_metadata(
    ubatch_slices: list[UBatchSlice],
    common_attn_metadata: CommonAttentionMetadata,
) -> list[CommonAttentionMetadata]:
    """
    Creates a new CommonAttentionMetadata instance that corresponds to the                          根据ubatch_slices,为每个UBatchSlice创建对应的CommonAttentionMetadata
    requests for each UBatchSlice in ubatch_slices.                                                 也就是说把一个大batch的attention metadata切分成多个ubatch的metadata

    Note: This function does not modify common_attn_metadata                                        注意这个函数不会修改原始common_attn_metadata 而是创建新的metadata副本
    """
    results = []
    for ubatch_slice in ubatch_slices:
        results.append(_make_metadata_with_slice(ubatch_slice, common_attn_metadata))

    return results


M = TypeVar("M")


class AttentionCGSupport(enum.Enum):
    """Constants for the cudagraph support of the attention backend                                 注意力后端对cuda graph支持级别的常量定义
    Here we do not consider the cascade attention, as currently
    it is never cudagraph supported."""

    ALWAYS = 3
    """Cudagraph always supported; supports mixed-prefill-decode"""                                 #始终支持;支持混合预填充和解码
    UNIFORM_BATCH = 2
    """Cudagraph supported for batches the only contain query lengths that are                      仅在批次中所有query长度相同的情况下支持cudagraph
    the same, this can be used for spec-decode
        i.e. "decodes" are 1 + num_speculative_tokens"""
    UNIFORM_SINGLE_TOKEN_DECODE = 1
    """Cudagraph supported for batches the only contain query_len==1 decodes"""                     #仅在批次中所有解码长度为1时支持cudagraph
    NEVER = 0
    """NO cudagraph support"""


class AttentionMetadataBuilder(abc.ABC, Generic[M]):
    # Does this backend/builder support CUDA Graphs for attention (default: no).                    当前attention backend/builder 是否支持CUDA graph  ,默认不支持
    # Do not access directly. Call get_cudagraph_support() instead.                                 注意不要直接访问这个成员变量
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER
    # Does this backend/builder reorder the batch?                                                  当前backend/builder是否会对batch重新排序(reorder)
    # If not, set this to None. Otherwise set it to the query                                       如果不会,则将其设置为None  否则将其设置为 会被移动到batch最前面的query长度
    # length that will be pulled into the front of the batch.
    reorder_batch_threshold: int | None = None
    # Does this backend/builder support updating the block table in existing                        当前backend/builder是否支持:在已有metadata上更新block table
    # metadata
    supports_update_block_table: bool = False

    @abstractmethod
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.kv_cache_spec = kv_cache_spec
        self.layer_names = layer_names
        self.vllm_config = vllm_config
        self.device = device

    @classmethod
    def get_cudagraph_support(
        cls: type["AttentionMetadataBuilder"],                                                      
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        """Get the cudagraph support level of this builder class.                                   获取当前构建器类对CUDA Graph的支持级别
        """
        return cls._cudagraph_support

    def _init_reorder_batch_threshold(
        self,
        reorder_batch_threshold: int | None = 1,                                                    #可以理解成什么时候允许backend对batch做重排序的长度阈值
        supports_spec_as_decode: bool = False,
        supports_dcp_with_varlen: bool = False,
    ) -> None:
        self.reorder_batch_threshold = reorder_batch_threshold
        if self.reorder_batch_threshold is not None and supports_spec_as_decode:
            # If the backend supports spec-as-decode kernels, then we can set                       如果后端支持speculative token当作decode的kernel
            # the reorder_batch_threshold based on the number of speculative                        那么我们可以根据配置中的spec token数量 来设置reorder_batch_threshold
            # tokens from the config.
            speculative_config = self.vllm_config.speculative_config
            if (
                speculative_config is not None
                and speculative_config.num_speculative_tokens is not None
            ):
                self.reorder_batch_threshold = max(
                    self.reorder_batch_threshold,
                    1 + speculative_config.num_speculative_tokens,
                )

        if (
            self.vllm_config.parallel_config.decode_context_parallel_size > 1
            and not supports_dcp_with_varlen
        ):
            self.reorder_batch_threshold = 1

    @abstractmethod
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> M:
        """
        Central method that builds attention metadata.                                              构建attention metadata的核心方法
        Some builders (MLA) require reorder_batch to be called prior to build.                      某些builder(如MLA)要求在调用build之前 先执行reorder_batch

        Args:
            common_prefix_len: The length of the common prefix of the batch.                        batch中公共前缀(common prefix)的长度
            common_attn_metadata: The common attention metadata.                                    通用attention元数据
            fast_build: The meta-data will prioritize speed of building over                        是否优先提高metadata的构建速度,而不是执行阶段的运行速度
                then speed at execution. Can be used for spec-decode where the                      这个选项可以用于spec decode
                result of a build call may only be used for few layers/iters.                       因为build的结果可能只会被少量layer或iteration使用
        """
        raise NotImplementedError

    def update_block_table(
        self,
        metadata: M,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> M:
        """
        Update the block table for the attention metadata.                                          更新attention metadata中的block table
        Faster when theres multiple kv-cache groups that create virtually the                       当存在多个kv-cache group时 如果它们生成的metadata基本相同
        same metadata but just with different block tables.                                         只是block table不同,那么直接更新block table会更高效

        Only needs to be implemented if supports_update_block_table is True.                        只有supports_updata_block_table为True时 才需要实现则会个方法
        """
        raise NotImplementedError

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> M:
        """
        Build attention metadata for CUDA graph capture. Uses build by default.                     为CUDA GRAPH  catputre构建attention metadata
        Subclasses that override this method should call self.build or                              默认实现直接使用build方法 
        super().build_for_cudagraph_capture.                                                        如果子类需要重写该方法,应显式调用self.build或super().build_for_cudagraph_capture
        """
        return self.build(
            common_prefix_len=0, common_attn_metadata=common_attn_metadata
        )

    def build_for_drafting(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        draft_index: int,
    ) -> M:
        """
        Build attention metadata for draft model. Uses build by default.

        Args:
            common_attn_metadata: The common attention metadata.
            draft_index: The index of the current draft operation.
                When speculating a chain of tokens, this index refers to the
                draft attempt for the i-th token.
                For tree-based attention, this index instead refers to the
                draft attempt for the i-th level in the tree of tokens.
        """
        return self.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
            fast_build=True,
        )

    def use_cascade_attention(
        self,
        common_prefix_len: int,
        query_lens: np.ndarray,
        num_query_heads: int,
        num_kv_heads: int,
        use_alibi: bool,
        use_sliding_window: bool,
        use_local_attention: bool,
        num_sms: int,
        dcp_world_size: int,
    ) -> bool:
        return False


@functools.lru_cache
def get_kv_cache_layout():
    """
    确定当前系统最终使用哪一种kv cache layout
    """
    # Format specified by the code.
    global _KV_CACHE_LAYOUT_OVERRIDE

    if _KV_CACHE_LAYOUT_OVERRIDE is not None:
        cache_layout = _KV_CACHE_LAYOUT_OVERRIDE
        logger.info_once(
            "`_KV_CACHE_LAYOUT_OVERRIDE` variable detected. "
            "Setting KV cache layout to %s.",
            cache_layout,
        )
        return cache_layout

    # Format specified by the user.
    cache_layout = envs.VLLM_KV_CACHE_LAYOUT
    # When neither the user nor the override specified a layout, get default
    if cache_layout is None:
        cache_layout = get_kv_connector_cache_layout()
    else:
        assert is_valid_kv_cache_layout(cache_layout)
        logger.info_once(
            "`VLLM_KV_CACHE_LAYOUT` environment variable "
            "detected. Setting KV cache layout to %s.",
            cache_layout,
        )
    return cache_layout


def set_kv_cache_layout(cache_layout: KVCacheLayoutType):
    global _KV_CACHE_LAYOUT_OVERRIDE
    _KV_CACHE_LAYOUT_OVERRIDE = cache_layout


@dataclass
class PerLayerParameters:
    """
    Currently, FlashInfer backend only support models in which all layers share                         当前FlashInfer后端仅支持所有层layer共享相同超参数的模型
    the same values for the following hyperparameters. Should not be used for                           注意:不适用于trtllm-gen后端,因为trtllm-gen支持不同层使用不同的超参数值
    trtllm-gen backend since it supports different values for the following
    hyperparameters.
    """

    window_left: int                                                                                    #窗口注意力 向左扩展的token数量
    logits_soft_cap: float | None                                                                       #logits soft cap值,用于限制logits的数值范围 防止数值爆炸
    sm_scale: float                                                                                     #attention中softmax的缩放因子, 通常为1/sqrt(head_dim)
    has_sinks: bool = False                                                                             #是否使用sink attention(沉底注意力) 用于保持初始token的能力
    # has same params for all layers
    has_same_window_lefts: bool | None = field(default=None, compare=False)
    has_same_all_params: bool | None = field(default=None, compare=False)


def get_per_layer_parameters(
    vllm_config: VllmConfig, layer_names: list[str], cls_: type["AttentionImpl"]
) -> dict[str, PerLayerParameters]:
    """
    Scan layers in `layer_names` and determine some hyperparameters                                     从指定的layer_names中扫描所有attention层,提取每个层需要的超参数,用于后续的attention planning阶段
    to use during `plan`.                                                                               为什么需要这个函数? FlashInfer后端限制,它目前要求模型所有层使用相同的注意力参数
    """

    layers = get_layers_from_vllm_config(vllm_config, AttentionLayerBase, layer_names)
    per_layer_params: dict[str, PerLayerParameters] = {}

    for key, layer in layers.items():                                                                   #例如 key = "model.layers.0", "model.layers.1" ...
        impl = layer.impl                                                                               #拿到真正的Attention实现对象
        assert isinstance(impl, cls_)                                                                   #确保是指定类型的Attention

        # Infer hyperparameters from the attention layer
        window_size = getattr(impl, "sliding_window", None)
        window_left = window_size[0] if window_size is not None else -1
        logits_soft_cap = getattr(impl, "logits_soft_cap", None)
        sm_scale = impl.scale
        has_sinks = getattr(impl, "sinks", None) is not None

        per_layer_params[key] = PerLayerParameters(
            window_left, logits_soft_cap, sm_scale, has_sinks
        )

    return per_layer_params


def infer_global_hyperparameters(
    per_layer_params: dict[str, PerLayerParameters],
) -> PerLayerParameters:
    """
    Currently, FlashInfer backend other than trtllm-gen                                                 除了trtllm-gen之外的FlashInfer后端仅支持所有层共享相同超参数的模型
    only support models in which all layers share
    the same values for the following hyperparameters:                                                  这些超参数包括
    - `window_left`
    - `logits_soft_cap`
    - `sm_scale`

    So this function asserts that all layers share the same values for these                            因此,此函数会断言(assert)所有层再上述超参数上取值相同,并返回全局统一的参数值
    hyperparameters and returns the global values.
    """

    assert len(per_layer_params) > 0, "No attention layers found in the model."                         #确保模型中至少包含一个Attention层

    param_sets = list(per_layer_params.values())                                                        #所有层的参数对象转为列表
    global_params = param_sets[0]

    global_params.has_same_window_lefts = all(                                                          #检查所有层是否参数一致
        params.window_left == global_params.window_left for params in param_sets
    )
    global_params.has_same_all_params = all(
        params == global_params for params in param_sets
    )

    return global_params


#
# Take in `query_start_loc_np` and `seq_lens_np` and break the sequences into                           #输入query_start_loc_np和seq_lens_np 将序列拆分成多个局部注意力块(local attention block)
# local attention blocks, where each block is passed to the attention kernel                            每个块作为独立的本地(虚拟)批处理传递给注意力内核
# as an independent local ("virtual") batch item.
#
# For example, if are performing a chunked prefill a batch of 3 sequences:                             例如,当我们对一个包含3个序列的batch进行chunked prefill(分块预填充)时,
#   q_seqlens  = [4, 10, 5]                                                                            q_seqlens=[4,10,5] 
#   kv_seqlens = [6, 17, 9]                                                                            kv_seqlens=[6,17,9]
# Then normally for regular attention we would compute with an attention mask                          在常规(regular)Attention中, 对于batch中第0个序列
#  for batch idx 0 (q_seqlens = 4, kv_seqlens = 6) like:                                               (q_seqlens=4, kv_seqlens=6) 其注意力掩码(attention mask)通常如下所示
#   batch idx: 0 (q_seqlens = 4, kv_seqlens = 6)
#        k_toks >   0 1 2 3 4 5                                                                        star难道不是下三角吗?为啥对角线右上方还有?
#        q_toks v  _____________
#               0 | 1 1 1
#               1 | 1 1 1 1
#               2 | 1 1 1 1 1
#               3 | 1 1 1 1 1 1
#
# for local attention (with attn_chunk_size = 4) we would compute with an                              对于局部注意力(local attention),当attn_chunk_size=4时
#  attention mask like:                                                                                我们会采用如下注意力掩码进行计算
#   batch idx: 0  (q_seqlens = 4, kv_seqlens = 6, attn_chunk_size = 4)                                 star:attn_chunk_size=4不是指每个query固定attend4个key,真实含义是把序列拆分成多个大小为4的注意力块,每个块内部独立做attention
#        k_toks >   0 1 2 3 4 5                                                                        这是一种分块局部注意力的实现方式,而不是经典的滑动窗口注意力
#        q_toks v  _____________
#               0 | 1 1 1                                                                              Q0只attend k0,k1,k2(共3个)           
#               1 | 1 1 1 1                                                                            Q1  attend k0,k1,k2,k3 (共4个)
#               2 |         1                                                                          Q2  attend k4(1个)
#               3 |         1 1                                                                        Q3  attend k4,k5(2个)   此处看ppt
#
# We can simulate this mask using standard flash-attention by breaking the                             可以通过把序列拆成局部(虚拟)batch来 用标准flash attention实现这个mask
#  sequences into local ("virtual") batches, where each local batch item is a                          每个局部batch元素对应一个local attention block
#  local attention block, so in this case batch idx 0 would be broken up into:                         所以原batch idx 0会被拆分成
#
#   local-batch idx: 0 (q_seqlens = 2, kv_seqlens = 4)  (batch 0)
#        k_toks >   0 1 2 3
#        q_toks v  _____________
#               0 | 1 1 1
#               1 | 1 1 1 1
#   local-batch idx: 1 (q_seqlens = 2, kv_seqlens = 2) (batch 0)
#        k_toks >   4 5
#        q_toks v  _____________
#               2 | 1
#               3 | 1 1
#
# e.g. if we have:                                                                                      这个例子展示的是 有一个函数会被原始batch(多个请求)按照attn_chunk_size切成很多小的(局部虚拟batch)
#   attn_chunk_size = 4                                                                                 每个虚拟局部batch只包含同一个chunk内的query
#   query_start_loc_np = [0, 4, 14, 19] (q_seqlens = [4, 10, 5])
# Then this function would return:
#                           __b0__  ______b1______  __b2__ < orig batch indices
#   q_seqlens_local    = [   2,  2,  1,  4,  4,  1,  4,  1]                                             此处看ppt吧
#   cu_seqlens_q_local = [0, 4,  6, 10, 14, 18, 19, 23, 24] 
#   seqlens_k_local    = [   4,  2,  4,  4,  4,  1,  4,  1]                                             每个虚拟小Batch 内部，当前 Q 块到底能看到多少个 K
#   block_table_local  : shape[local_virtual_batches, pages_per_local_batch]                
def make_local_attention_virtual_batches(
    attn_chunk_size: int,
    common_attn_metadata: CommonAttentionMetadata,
    block_size: int = 0,
) -> tuple[CommonAttentionMetadata, Callable[[torch.Tensor], torch.Tensor]]:
    """
    将原始batch按照attn_chunk_size切分成多个小的 虚拟local batch
    使得每个虚拟batch得kv长度都不超过attn_chunk_size.
    从而可以使用标准FlashAttention实现chunked local attention
    """
    
    query_start_loc_np = common_attn_metadata.query_start_loc_cpu.numpy()
    seq_lens_np = common_attn_metadata.seq_lens_cpu.numpy()
    block_table = common_attn_metadata.block_table_tensor
    device = common_attn_metadata.query_start_loc.device

    q_seqlens = query_start_loc_np[1:] - query_start_loc_np[:-1]
    actual_batch_size = seq_lens_np.shape[0]

    # Handle if we are starting in the middle of a local attention block,                               处理从一个local attention block中间开始得情况
    #  we assume q_seqlens > 0 (for all elements), for each batch idx we compute                        我们假设每个请求的q_seqlens都大于0  对每个原始batch idx 我们先算出不再第一个local attention block里的query token数量
    #  the number of tokens that are not in the first local attention block and                         然后对剩余的query数量使用cdiv(向上取整除法) 就能得到总共多少个local block
    #  then we can simply use a cdiv for the rest.
    # For example if we have:
    #   attn_chunk_size = 4
    #   q_seqlens = [4, 10, 5]
    #   k_seqlens = [6, 17, 9]
    # Then we would get:
    #   new_tokens_in_first_block = [2, 1, 4]                                                           每个请求第一个local block里包含的query数量
    #   local_blocks = [2, 4, 2]                                                                        为每个请求最终拆成多少个虚拟local batch
    q_tokens_in_first_block = np.minimum(
        attn_chunk_size - ((seq_lens_np - q_seqlens) % attn_chunk_size), q_seqlens
    ).astype(np.int32)
    tokens_in_last_block = attn_chunk_size + (seq_lens_np % -attn_chunk_size)                           #计算每个请求最后一个local block的kv 长度(可能是partial)
    local_blocks = 1 + cdiv(q_seqlens - q_tokens_in_first_block, attn_chunk_size)                       #计算每个原始请求最重要被拆成多少个local virtual batch

    # Once we know the number of local blocks we can compute the request spans                          现在我们已经知道每个请求要拆成几个local block
    #  for each batch idx, we can figure out the number of "virtual" requests we                        接下来要为所有虚拟batch生成编号,方便后续构造seqlens_q_local
    #  have to make,
    # For the above example we would get:
    #   seqlens_q_local = [2, 2, 1, 4, 4, 1, 4, 1]
    #
    # First Get batched arange. (E.g., [2, 4, 2] -> [0, 1, 0, 1, 2, 3, 0, 1])
    #   (TODO: max a utility to share this code with _prepare_inputs)
    # arange step 1. [2, 4, 2] -> [2, 6, 8]
    cu_num_blocks = np.cumsum(local_blocks)
    virtual_batches = cu_num_blocks[-1]
    # arange step 2. [2, 6, 8] -> [0, 0, 2, 2, 2, 2, 6, 6]
    block_offsets = np.repeat(cu_num_blocks - local_blocks, local_blocks)
    # arange step 3. [0, 1, 0, 1, 2, 3, 0, 1]
    arange = np.arange(virtual_batches, dtype=np.int32) - block_offsets
    # also compute reverse arange (i.e. [1, 0, 3, 2, 1, 0, 1, 0])
    rarange = np.repeat(local_blocks, local_blocks) - arange - 1
    # Then we can compute the seqlens_q_local, handling the fact that the
    #  first and last blocks could be partial
    seqlens_q_local = np.repeat(q_seqlens - q_tokens_in_first_block, local_blocks)
    # set the first block since this may be a partial block
    seqlens_q_local[arange == 0] = q_tokens_in_first_block
    # set the remaining blocks
    seqlens_q_local[arange > 0] = np.minimum(
        seqlens_q_local - attn_chunk_size * (arange - 1), attn_chunk_size
    )[arange > 0]

    # convert from q_seqlens to cu_seqlens_q
    cu_seqlens_q_local = np.empty(virtual_batches + 1, dtype=np.int32)
    np.cumsum(seqlens_q_local, out=cu_seqlens_q_local[1:])
    cu_seqlens_q_local[0] = 0

    # compute the seqlens_k_local,
    #  basically a full local attention block for all but the last block in each
    #  batch
    # For our example this will be:
    #   seqlens_k_local = [4, 2, 4, 4, 4, 1, 4, 1]
    seqlens_k_local = np.full(cu_num_blocks[-1], attn_chunk_size, dtype=np.int32)
    seqlens_k_local[cu_num_blocks - 1] = tokens_in_last_block
    num_computed_tokens_local = seqlens_k_local - seqlens_q_local

    k_seqstarts_absolute = np.repeat(seq_lens_np, local_blocks) - (
        rarange * attn_chunk_size + np.repeat(tokens_in_last_block, local_blocks)
    )
    # For the example the local attention blocks start at:
    #                           _b0_  _____b1_____  _b2_
    #   k_seqstarts_absolute = [0, 4, 4, 8, 12, 16, 4, 8]
    block_starts = k_seqstarts_absolute // block_size
    assert attn_chunk_size % block_size == 0, (
        f"attn_chunk_size {attn_chunk_size} is not divisible by block_size {block_size}"
    )
    pages_per_local_batch = attn_chunk_size // block_size

    # Create a block_table for the local attention blocks
    # For out example if we have a block-table like (assuming block_size=2):
    #   block_table = [
    #     [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],  < batch 0
    #     [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],  < batch 1
    #     [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],  < batch 2
    #   ]
    # Then for the local batches we would want a block-table like
    #   block_table_local = [
    #     [  0,  1 ], < local-batch 0, (batch 0, starting from k[0])
    #     [  2,  3 ], < local-batch 1, (batch 0, starting from k[4])
    #     [ 12, 13 ], < local-batch 2, (batch 1, starting from k[4])
    #     [ 14, 15 ], < local-batch 3, (batch 1, starting from k[8])
    #     [ 16, 17 ], < local-batch 4, (batch 1, starting from k[12])
    #     [ 18, 19 ], < local-batch 5, (batch 1, starting from k[16])
    #     [ 22, 23 ], < local-batch 6, (batch 2, starting from k[4])
    #     [ 24, 25 ], < local-batch 7, (batch 2, starting from k[8])
    #   ]
    block_indices = block_starts[:, None] + np.arange(
        pages_per_local_batch, dtype=np.int32
    )
    block_indices = block_indices.reshape(-1).clip(max=block_table.shape[1] - 1)
    batch_indices = np.repeat(
        np.arange(actual_batch_size, dtype=np.int32),
        local_blocks * pages_per_local_batch,
    )

    # NOTE: https://github.com/pytorch/pytorch/pull/160256 causes performance
    # regression when using numpy arrays (batch and block indices) to index into
    # torch tensor (block_table). As a workaround, convert numpy arrays to torch
    # tensor first, which recovers perf.
    batch_indices_torch = torch.from_numpy(batch_indices)
    block_indices_torch = torch.from_numpy(block_indices)

    # Save as a lambda so we can return this for update_block_table
    make_block_table = lambda block_table: block_table[
        batch_indices_torch, block_indices_torch
    ].view(virtual_batches, -1)
    block_table_local = make_block_table(block_table)

    query_start_loc_cpu = torch.from_numpy(cu_seqlens_q_local)
    seq_lens_cpu = torch.from_numpy(seqlens_k_local)
    max_seq_len = int(seq_lens_cpu.max())

    return CommonAttentionMetadata(
        query_start_loc_cpu=query_start_loc_cpu,
        query_start_loc=query_start_loc_cpu.to(device=device, non_blocking=True),
        seq_lens=seq_lens_cpu.to(device=device, non_blocking=True),
        num_reqs=len(seq_lens_cpu),
        num_actual_tokens=common_attn_metadata.num_actual_tokens,
        max_query_len=seqlens_q_local.max(),
        max_seq_len=max_seq_len,
        block_table_tensor=block_table_local,
        slot_mapping=common_attn_metadata.slot_mapping,
        causal=True,
        _seq_lens_cpu=seq_lens_cpu,
        _num_computed_tokens_cpu=torch.from_numpy(num_computed_tokens_local),
    ), make_block_table


def make_kv_sharing_fast_prefill_common_attn_metadata(
    common_attn_metadata: CommonAttentionMetadata,
) -> CommonAttentionMetadata:
    if common_attn_metadata.max_query_len == 1:
        # All requests are decode (assume 1 token for now)
        # Skip computing fast prefill path
        return common_attn_metadata

    assert common_attn_metadata.logits_indices_padded is not None
    assert common_attn_metadata.num_logits_indices is not None

    logits_indices_padded = common_attn_metadata.logits_indices_padded
    num_logits_indices = common_attn_metadata.num_logits_indices
    # Get rid of CUDAGraph padding, if any
    logits_indices = logits_indices_padded[:num_logits_indices]
    num_reqs = common_attn_metadata.num_reqs
    query_start_loc = common_attn_metadata.query_start_loc
    # Example inputs
    # num_reqs: 3
    # generation_indices:  [14, 18, 19, 27]
    # query_start_loc: [0, 15, 20, 28]
    # seq_lens:        [41, 31, 40]

    # Find how many decode indices belong to each request
    # request_ids: [0, 1, 1, 2]
    request_ids = torch.bucketize(logits_indices, query_start_loc[1:], right=True)

    # Figure out how many tokens are in each request
    # num_decode_tokens: [1, 2, 1]
    num_decode_tokens = torch.bincount(request_ids, minlength=num_reqs)

    # Calculate new query_start_loc with tokens in generation_indices
    # decode_query_start_loc: [0, 1, 3, 4]
    decode_query_start_loc = torch.empty(
        num_reqs + 1, device=query_start_loc.device, dtype=query_start_loc.dtype
    )

    decode_query_start_loc[0] = 0
    decode_query_start_loc[1:] = torch.cumsum(num_decode_tokens, dim=0)
    decode_max_query_len = int(num_decode_tokens.max().item())
    total_num_decode_tokens = int(num_decode_tokens.sum().item())

    common_attn_metadata = CommonAttentionMetadata(
        query_start_loc=decode_query_start_loc,
        query_start_loc_cpu=decode_query_start_loc.to("cpu", non_blocking=True),
        seq_lens=common_attn_metadata.seq_lens,
        num_reqs=num_reqs,
        num_actual_tokens=total_num_decode_tokens,
        max_query_len=decode_max_query_len,
        max_seq_len=common_attn_metadata.max_seq_len,
        block_table_tensor=common_attn_metadata.block_table_tensor,
        slot_mapping=common_attn_metadata.slot_mapping,
        causal=True,
        _seq_lens_cpu=common_attn_metadata._seq_lens_cpu,
        _num_computed_tokens_cpu=common_attn_metadata._num_computed_tokens_cpu,
    )
    return common_attn_metadata


def subclass_attention_backend(
    name_prefix: str,
    attention_backend_cls: type[AttentionBackend],
    builder_cls: type[AttentionMetadataBuilder[M]],
) -> type[AttentionBackend]:
    """
    Return a new subclass where `get_builder_cls` returns `builder_cls`.
    """
    name: str = name_prefix + attention_backend_cls.__name__  # type: ignore

    return type(
        name, (attention_backend_cls,), {"get_builder_cls": lambda: builder_cls}
    )


def subclass_attention_backend_with_overrides(
    name_prefix: str,
    attention_backend_cls: type[AttentionBackend],
    overrides: dict[str, Any],
) -> type[AttentionBackend]:
    name: str = name_prefix + attention_backend_cls.__name__  # type: ignore
    return type(name, (attention_backend_cls,), overrides)


def split_decodes_prefills_and_extends(
    common_attn_metadata: CommonAttentionMetadata,
    decode_threshold: int = 1,
) -> tuple[int, int, int, int, int, int]:
    """
    Assuming a reordered batch, finds the boundary between prefill and decode
    requests.

    Args:
        common_attn_metadata: CommonAttentionMetadata object containing the
            batch metadata.
        decode_threshold: The maximum query length to be considered a decode.

    Returns:
        num_decodes: The number of decode requests.
        num_extends: The number of extend requests.
        num_prefills: The number of prefill requests.
        num_decode_tokens: The number of tokens in the decode requests.
        num_extend_tokens: The number of tokens in the extend requests.
        num_prefill_tokens: The number of tokens in the prefill requests.
    """
    max_query_len = common_attn_metadata.max_query_len
    num_reqs = common_attn_metadata.num_reqs
    num_tokens = common_attn_metadata.num_actual_tokens
    query_start_loc = common_attn_metadata.query_start_loc_cpu
    seq_lens = common_attn_metadata.seq_lens_cpu

    if max_query_len <= decode_threshold:
        return num_reqs, 0, 0, num_tokens, 0, 0

    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    is_prefill_or_extend = query_lens > decode_threshold
    is_prefill = (seq_lens == query_lens) & is_prefill_or_extend
    first_extend = is_prefill_or_extend.int().argmax(dim=-1).item()
    first_prefill = is_prefill.int().argmax(dim=-1).item()
    num_decodes = first_extend
    num_decode_tokens = query_start_loc[first_extend].item()
    if not torch.any(is_prefill_or_extend):
        return (num_decodes, 0, 0, num_decode_tokens, 0, 0)

    num_prefills_or_extends = num_reqs - num_decodes
    num_prefill_or_extend_tokens = num_tokens - num_decode_tokens
    if not torch.any(is_prefill):
        return (
            num_decodes,
            num_prefills_or_extends,
            0,
            num_decode_tokens,
            num_prefill_or_extend_tokens,
            0,
        )

    num_extends = first_prefill - num_decodes
    num_prefills = num_reqs - first_prefill

    num_prefill_tokens = num_tokens - query_start_loc[first_prefill]
    num_extend_tokens = num_prefill_or_extend_tokens - num_prefill_tokens
    return (
        num_decodes,
        num_extends,
        num_prefills,
        num_decode_tokens,
        num_extend_tokens,
        num_prefill_tokens,
    )


def split_decodes_and_prefills(
    common_attn_metadata: CommonAttentionMetadata,
    decode_threshold: int = 1,
    require_uniform: bool = False,
) -> tuple[int, int, int, int]:
    """
    Assuming a reordered batch, finds the boundary between prefill and decode                               #把一个混合了prefill和decode阶段的batch 按照query_len进行切分,找出prefill和decode的边界
    requests.

    Args:
        common_attn_metadata: CommonAttentionMetadata object containing the
            batch metadata.
        decode_threshold: The maximum query length to be considered a decode.
        require_uniform: If True, requires that all decode requests have the
            same query length. When set, some queries may be considered prefills
            even if they are <= decode_threshold, in order to ensure uniformity.

    Returns:
        num_decodes: The number of decode requests.
        num_prefills: The number of prefill requests.
        num_decode_tokens: The number of tokens in the decode requests.
        num_prefill_tokens: The number of tokens in the prefill requests.
    """
    max_query_len = common_attn_metadata.max_query_len
    num_reqs = common_attn_metadata.num_reqs
    num_tokens = common_attn_metadata.num_actual_tokens
    query_start_loc = common_attn_metadata.query_start_loc_cpu

    if max_query_len <= decode_threshold and (
        not require_uniform or decode_threshold <= 1
    ):
        return num_reqs, 0, num_tokens, 0

    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    if query_lens[0].item() > decode_threshold:
        # first request is not decode, so no decode requests
        return 0, num_reqs, 0, num_tokens

    if require_uniform:
        # check if we are in a padded uniform batch; this is used for full-CGs, some
        # requests may have a query length of 0 but since they are padding its fine
        # to treat them as decodes (ensures num_decodes matches the captured size)
        if torch.all((query_lens == query_lens[0]) | (query_lens == 0)):
            assert num_reqs * query_lens[0] == num_tokens, "tokens not padded correctly"
            return num_reqs, 0, num_tokens, 0  # all decodes
        is_prefill = query_lens != query_lens[0]
    else:
        is_prefill = query_lens > decode_threshold

    if not torch.any(is_prefill):
        return num_reqs, 0, num_tokens, 0

    first_prefill = is_prefill.int().argmax(dim=-1).item()
    assert torch.all(query_lens[:first_prefill] <= decode_threshold)
    num_decodes = first_prefill
    num_prefills = num_reqs - num_decodes
    num_decode_tokens = query_start_loc[first_prefill].item()
    num_prefill_tokens = num_tokens - num_decode_tokens
    return (num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens)


def split_prefill_chunks(
    seq_lens_cpu: torch.Tensor, workspace_size: int, request_offset: int = 0
) -> list[tuple[int, int]]:
    """
    Split the prefill requests into chunks such that the total sequence length                              把多个prefill请求按照它们的sequence length进行分组,使每一组(chunk)内所有请求的token数量(total sequence length)
    of each chunk is less than or equal to the workspace size.                                              不超过workspace_size

    Args:
        seq_lens_cpu: The sequence lengths of the prefill requests on CPU.
        workspace_size: The maximum workspace size (in tokens) per chunk.
        request_offset: The offset to add to the request indices.
    Returns:
        A list of tuples of (reqs_start, reqs_end) representing chunk boundaries.                           返回一个元组 表示把哪些请求放在同一个chunk里, 如[(0, 3), (3, 4), (4, 5)]表示0-2个请求放一个chunk, 第3个单独一个chunk
    """
    chunk_bounds = []
    i, n = 0, len(seq_lens_cpu)
    assert torch.all(seq_lens_cpu <= workspace_size).item()

    while i < n:
        start, chunk_total = i, 0
        while i < n and (chunk_total + (s := seq_lens_cpu[i].item())) <= workspace_size:
            chunk_total += s
            i += 1
        chunk_bounds.append((start + request_offset, i + request_offset))
    return chunk_bounds


def reorder_batch_to_split_decodes_and_prefills(
    input_batch: "InputBatch",
    scheduler_output: "SchedulerOutput",
    decode_threshold: int = 1,
) -> bool:
    """
    Reorders the batch to split into prefill and decode requests; places all                            该函数通过重新排列批次(Batch)将请求分为“预填充(Prefill)”和“解码(Decode)”两部分;
    requests with <= decode_threshold tokens at the front of the batch.                                 将所有调度 Token 数量 <= decode_threshold(解码阈值)的请求排在批次的最前面。
    
    Returns:
        True if the batch was modified, False otherwise.                                                #返回值: True  表示 batch 顺序发生了改变(进行了重排序);False 表示 batch 顺序已经是理想状态,无需调整
    """
    # We now want to reorder the batch into decode → extend → prefill order                             #我们希望将批次按照 解码 (Decode) → 扩展 (Extend) → 预填充 (Prefill) 的顺序排列
    # where:
    #   decode: request with num_scheduled_tokens <= decode_threshold                                   #Decode: 调度 Token 数<= 阈值的请求(通常为 1)。
    #   extend: non-decode request with existing context                                                #Extend: 调度 Token 数大于阈值,且已有历史上下文的请求。 这个分多种情况 可能长prompt分段
    #   prefill: non-decode request with no existing context                                            #Prefill: 调度 Token 数大于阈值,且没有历史上下文的请求(新请求)。
    # NOTE for now we loosely use "decode" to mean requests where attention is                          注意:目前我们粗略地用 "Decode" 代表访存密集型(Memory-bound)请求,用 "Prefill" 代表计算密集型(Compute-bound)请求
    #  likely memory-bound and "prefill" to mean requests where attention is                            #重排目的:把性质相似的请求扎堆,让底层的注意力算子(Attention Kernel)能一次性处理一类任务,从而减少 GPU 指令切换的开销。
    #  likely compute-bound,
    # 实际例子:
    #   当前 batch 有 5 个请求:
    #       A: Decode(调度 1 token)
    #       B: Prefill(新请求,调度 800 token)
    #       C: Decode(调度 1 token)
    #       D: Extend(已有上下文,调度 4 token)
    #       E: Prefill(新请求,调度 1200 token)
    #
    #   重排序前:[A, B, C, D, E]   → 混合交错,效率低
    #   重排序后:[A, C, D, B, E]   → Decode + Extend + Prefill 分组,效率更高
    num_reqs = len(input_batch.req_ids)
    
    num_scheduled_tokens = [scheduler_output.num_scheduled_tokens[id] for id in input_batch.req_ids] #获取每个请求本次调度(scheduled)的token数量
    num_scheduled_tokens_np = np.array(num_scheduled_tokens)                                         
    num_computed_tokens_np = input_batch.num_computed_tokens_cpu[:num_reqs]                          ## 获取每个请求已经计算过的 token 数量(用于区分 Extend 和 Prefill)

    # ====================== 分类三种请求类型 ======================
    is_decode = num_scheduled_tokens_np <= decode_threshold                                          ## is_decode: 本次只调度很少 token(通常是 1),属于正在生成的 Decode 请求
    is_extend = (~is_decode) & (num_computed_tokens_np > 0)                                          ## is_extend: 本次调度较多 token,但已经有历史上下文(继续生成)
    is_prefill = (~is_decode) & (num_computed_tokens_np == 0)                                        # is_prefill: 本次调度较多 token,且没有任何历史上下文(全新请求)

    # Desired order: decode → extend → prefill
    req_regions = np.zeros(is_decode.shape, dtype=np.int32)  # 0 = decode by default
    req_regions[is_extend] = 1
    req_regions[is_prefill] = 2

    num_decodes = int(is_decode.sum())
    num_extends = int(is_extend.sum())

    target_regions = np.zeros(num_reqs, dtype=np.int32)
    target_regions[num_decodes : num_decodes + num_extends] = 1
    target_regions[num_decodes + num_extends :] = 2

    needs_swap = req_regions != target_regions                                                      #判断是否要重排序,req_regiions:当前每个request所在的类别 target_regions:理想排序下每个位置该放的类别。shape =[num_reqs]

    if not needs_swap.any():
        return False                                                                                #已经是理想排序 无需调整

    # ====================== 执行重排序 ======================                                       eg.current req_regions: 0 2 0 1 2   target:0 0 1 2 2  
    # Extract indices that need swapping and sort by target region
    orig_indices = np.where(needs_swap)[0]                                                          #只拿True的  如need_swap=[1,2,3]
    sorted_order = np.argsort(req_regions[needs_swap], kind="stable")                               #req_regions[needs_swap]=[2, 0, 1] 排序结果[0,1,2](原的索引时1 2 0),所以该值是[1,2,0]
    src_indices = orig_indices[sorted_order]                                                        #[2,3,1]

    #构建源索引->目标索引的摄影社
    src_dest_map = {int(src): int(dst) for src, dst in zip(src_indices, orig_indices)}
    # 执行实际的交换操作(使用循环处理可能出现的链式交换)
    for src in src_dest_map:
        dst = src_dest_map[src]
        while src != dst:
            input_batch.swap_states(src, dst)                 ## 真正交换两个请求的所有状态
            # Mark dst as done by updating its destination to itself
            next_dst = src_dest_map.get(dst, dst)
            src_dest_map[dst] = dst
            dst = next_dst

    return True


def reshape_query_for_spec_decode(query: torch.Tensor, batch_size: int) -> torch.Tensor:
    """
    Reshapes the query tensor for the specified batch size, so that
    it has shape (batch_size, seq_len, num_heads, head_dim).
    """
    assert query.dim() == 3, f"query must be 3D, got {query.dim()}D"
    total_tokens = query.shape[0]
    num_heads = query.shape[1]
    head_dim = query.shape[2]
    assert total_tokens % batch_size == 0, (
        f"{total_tokens=} is not divisible by {batch_size=}"
    )
    seq_len = total_tokens // batch_size
    return query.view(batch_size, seq_len, num_heads, head_dim)


def reshape_attn_output_for_spec_decode(attn_output: torch.Tensor) -> torch.Tensor:
    """
    Reshapes the attention output tensor, so that
    the batch_size and seq_len dimensions are combined.
    """
    if attn_output.dim() == 3:
        # Already in the correct shape
        return attn_output
    assert attn_output.dim() == 4, f"attn_output must be 4D, got {attn_output.dim()}D"
    total_tokens = attn_output.shape[0] * attn_output.shape[1]
    return attn_output.view(total_tokens, attn_output.shape[2], attn_output.shape[3])


def subclass_attention_metadata(
    name_prefix: str,
    metadata_cls: Any,
    fields: list[tuple[str, Any, Any]],
) -> Any:
    """
    Return a new subclass of `metadata_cls` with additional fields                                      动态创建一个metadata_cls子类,为其添加额外的字段
    """
    name: str = name_prefix + metadata_cls.__name__  # type: ignore                                     #构造新类的完整名称,例如FlashAttnCommonAttentionMetadata
    Wrapped = make_dataclass(name, fields, bases=(metadata_cls,))
    return Wrapped


@runtime_checkable
class KVSharingFastPrefillMetadata(Protocol):
    logits_indices_padded: torch.Tensor | None = None
    num_logits_indices: int | None = None


def create_fast_prefill_custom_backend(
    prefix: str,
    underlying_attn_backend: AttentionBackend,
) -> type[AttentionBackend]:
    """
    创建一个自定义的Attention Backend 用于支持Fast prefill + kv sharing特性
    通过继承原有backend的Builder,并在build阶段注入自定义的metadata
    从而实现对attention计算流程的增强,而需要修改底层kernel
    """
    
    underlying_builder = underlying_attn_backend.get_builder_cls()                                      #获取底层backend的Builder类(如FlashAttentionBuilder FlashInferBuilder等)

    class FastPrefillAttentionBuilder(underlying_builder):  # type: ignore
        def build(
            self,
            common_prefix_len: int,
            common_attn_metadata: CommonAttentionMetadata,
            fast_build: bool = False,
        ) -> AttentionMetadata:
            new_common_attn_metadata = (                                                                #对common_attn_metadata做预处理(注入kv sharing相关信息)
                make_kv_sharing_fast_prefill_common_attn_metadata(common_attn_metadata)
            )
            metadata = super().build(                                                                   #调用原始builder的build方法,生成基础的attention metadata
                common_prefix_len, new_common_attn_metadata, fast_build
            )

            class KVSharingFastPrefillAttentionMetadata(                                                #动态创建自定义metadata类
                metadata.__class__,  #  type: ignore
                KVSharingFastPrefillMetadata,
            ):
                def __init__(self, metadata, common_attn_metadata):
                    # Shallow copy all fields in metadata cls
                    for _field in fields(metadata.__class__):
                        setattr(self, _field.name, getattr(metadata, _field.name))

                    self.logits_indices_padded = (
                        common_attn_metadata.logits_indices_padded
                    )
                    self.num_logits_indices = common_attn_metadata.num_logits_indices

            return KVSharingFastPrefillAttentionMetadata(metadata, common_attn_metadata)

    attn_backend = subclass_attention_backend(
        name_prefix=prefix,
        attention_backend_cls=underlying_attn_backend,
        builder_cls=FastPrefillAttentionBuilder,
    )

    return attn_backend


def compute_causal_conv1d_metadata(query_start_loc_p: torch.Tensor):
    """
    为causal_conv1d kernel准备元数据
    这个函数主要用于支持变长序列（variable sequence length）的 causal convolution 计算。
    它会把每个序列按 BLOCK_M 分块，并预先计算好每个 kernel program（线程块）应该处理
    哪个 batch 的哪一段数据，从而让 Triton/CUDA kernel 能够高效地并行执行
    """
    
    # Needed for causal_conv1d
    seqlens = query_start_loc_p.diff().to("cpu")
    nums_dict = {}  # type: ignore
    batch_ptr = None
    token_chunk_offset_ptr = None
    device = query_start_loc_p.device
    for BLOCK_M in [8]:  # cover all BLOCK_M values
        nums = -(-seqlens // BLOCK_M)
        nums_dict[BLOCK_M] = {}
        nums_dict[BLOCK_M]["nums"] = nums
        nums_dict[BLOCK_M]["tot"] = nums.sum().item()
        mlist = torch.from_numpy(np.repeat(np.arange(len(nums)), nums))
        nums_dict[BLOCK_M]["mlist"] = mlist
        mlist_len = len(nums_dict[BLOCK_M]["mlist"])
        nums_dict[BLOCK_M]["mlist_len"] = mlist_len
        MAX_NUM_PROGRAMS = max(1024, mlist_len) * 2
        offsetlist = []  # type: ignore
        for idx, num in enumerate(nums):
            offsetlist.extend(range(num))
        offsetlist = torch.tensor(offsetlist, dtype=torch.int32)
        nums_dict[BLOCK_M]["offsetlist"] = offsetlist

        if batch_ptr is None:
            # Update default value after class definition
            batch_ptr = torch.full(
                (MAX_NUM_PROGRAMS,), PAD_SLOT_ID, dtype=torch.int32, device=device
            )
            token_chunk_offset_ptr = torch.full(
                (MAX_NUM_PROGRAMS,), PAD_SLOT_ID, dtype=torch.int32, device=device
            )
        else:
            if batch_ptr.nelement() < MAX_NUM_PROGRAMS:
                batch_ptr.resize_(MAX_NUM_PROGRAMS).fill_(PAD_SLOT_ID)
                token_chunk_offset_ptr.resize_(  # type: ignore
                    MAX_NUM_PROGRAMS
                ).fill_(PAD_SLOT_ID)

        batch_ptr[0:mlist_len].copy_(mlist)
        token_chunk_offset_ptr[  # type: ignore
            0:mlist_len
        ].copy_(offsetlist)
        nums_dict[BLOCK_M]["batch_ptr"] = batch_ptr
        nums_dict[BLOCK_M]["token_chunk_offset_ptr"] = token_chunk_offset_ptr  # type: ignore

    return nums_dict, batch_ptr, token_chunk_offset_ptr


def get_dcp_local_seq_lens(
    seq_lens: torch.Tensor,
    dcp_size: int = 1,
    dcp_rank: int | None = None,
    cp_kv_cache_interleave_size: int = 1,
) -> torch.Tensor:
    """While using dcp, kv_cache size stored on each rank may be different,  在启用DCP时,每个DCP rank本地应该负责的序列长度(local seq lens)
    use this function to calculate split decode seq_lens of each dcp rank.    由于kv cache在不同rank之间是切分存储的,每个rank上实际存储的kv cache大小不同,因此需要通过这个函数计算出当前rank在decode阶段应该处理的本地序列长度
    Only consider dcp now, we can extend the case of cp based on this.        注意,目前主要支持DCP,未来可基于此扩展Context Parallel(CP)的情况
    """
    num_requests = seq_lens.size(0)
    if dcp_rank is None:
        rank_offsets = (
            torch.arange(dcp_size, dtype=torch.int32, device=seq_lens.device)
            .unsqueeze(0)
            .repeat(num_requests, 1)
        )
    else:
        rank_offsets = torch.tensor(
            [[dcp_rank]], dtype=torch.int32, device=seq_lens.device
        )
    seq_lens_tiled = (
        seq_lens.to(torch.int32).unsqueeze(-1).repeat(1, rank_offsets.shape[1])
    )
    base = (
        seq_lens_tiled
        // cp_kv_cache_interleave_size
        // dcp_size
        * cp_kv_cache_interleave_size
    )
    remainder = seq_lens_tiled - base * dcp_size
    remainder = torch.clip(
        remainder - rank_offsets * cp_kv_cache_interleave_size,
        0,
        cp_kv_cache_interleave_size,
    )
    dcp_local_seq_lens = base + remainder
    return dcp_local_seq_lens.squeeze(1)
