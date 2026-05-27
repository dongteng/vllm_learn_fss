# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Generic, Protocol, TypeVar, get_args

import torch

if TYPE_CHECKING:
    from vllm.config.cache import CacheDType
    from vllm.model_executor.layers.linear import ColumnParallelLinear
    from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey
    from vllm.platforms.interface import DeviceCapability
    from vllm.v1.attention.backends.utils import KVCacheLayoutType


class AttentionType:
    """
    Attention type.
    Use string to be compatible with `torch.compile`.
    """

    DECODER = "decoder"
    """Decoder attention between previous layer Q/K/V."""
    ENCODER = "encoder"
    """Encoder attention between previous layer Q/K/V for encoder-decoder."""
    ENCODER_ONLY = "encoder_only"
    """Encoder attention between previous layer Q/K/V."""
    ENCODER_DECODER = "encoder_decoder"
    """Attention between dec. Q and enc. K/V for encoder-decoder."""


class MultipleOf:
    base: int

    def __init__(self, base: int):
        self.base = base


class AttentionBackend(ABC):
    """Abstract class for attention backends."""                                                    #Attention后端的抽象类,vllm通过这个抽象基类来管理不同的Attention实现(FlashAttention FlashInfer Triton等)

    # For some attention backends, we allocate an output tensor before                              #对于某些attention后端，我们会在调用自定义算子(custom op)之前,先分配一个输出张量(output tensor)
    # calling the custom op. When piecewise cudagraph is enabled, this                              #当启用分段式CUDA Graph(piecewise cudagraph)时,这样可以确保输出张量是在CUDA Graph内部完成分配的
    # makes sure the output tensor is allocated inside the cudagraph.
    accept_output_buffer: bool = False
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]                 #ClassVar表示这个变量属于类本身 而不是实例对象
    supported_kv_cache_dtypes: ClassVar[list["CacheDType"]] = ["auto"]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        """
        返回当前kernel支持的block size列表  int:表示支持某个固定的block size,例如16 32 64
        MultipleOf 表示支持某个数的倍数,这里1的意思是支持任意block size
        """
        
        return [MultipleOf(1)]

    @staticmethod                                                                                   #这个函数不需要self
    @abstractmethod                                                                                 #子类必须实现这个函数
    def get_name() -> str:
        raise NotImplementedError                                                                   #返回当前attention backend的名称(字符串) 例如flash_attn或flashinfer

    @staticmethod
    @abstractmethod
    def get_impl_cls() -> type["AttentionImpl"]:
        raise NotImplementedError                                                                   #返回当前backend对应的AttentionImpl类 AttentionImpl是真正执行attention计算的实现类(包含forward方法)

    @staticmethod
    @abstractmethod
    def get_builder_cls():  # -> Type["AttentionMetadataBuilder"]:                                  返回当前 backend 对应的 AttentionMetadataBuilder 类。Builder 负责在每次 forward 前构建 AttentionMetadata。
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        raise NotImplementedError

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        """
        Get the physical (memory layout) ordering of the kv cache dimensions.                       获取kv cache各个维度在物理内存中的排列顺序(memory layout)
        e.g. if the KV cache shape is                                                               例如,假设kv cache的逻辑形状为[2, num_blocks, block_size, num_heads, head_size],
        [2, num_blocks, block_size, num_heads, head_size],
        and get_kv_cache_stride_order returns (1, 3, 0, 2, 4) then the physical                     如果get_kv_cache_stride_orider返回(1,3,0,2,4) 那么其物理维度顺序将变为
        ordering of dimensions is                                                                   [num_blocks, num_heads, 2, block_size, head_size].
        [num_blocks, num_heads, 2, block_size, head_size].

        If this function is unimplemented / raises NotImplementedError,                             如果该函数没实现 或者抛出NotImplementedError 则kv cache的物理布局与逻辑形状保持一致
        the physical layout of the KV cache will match the logical shape.                           不进行额外的维度重排

        Args:
            include_num_layers_dimension: if True, includes an additional
                num_layers dimension, which is assumed to be prepended
                to the logical KV cache shape.
                With the above example, a return value (2, 4, 0, 1, 3, 5)
                corresponds to
                [num_blocks, num_heads, num_layers, 2, block_size, head_size].

                If an additional dimension is NOT included in the returned
                tuple, the physical layout will not include a layers dimension.

        Returns:
            A tuple of ints which is a permutation of range(len(shape)).
        """
        raise NotImplementedError

    @classmethod
    def full_cls_name(cls) -> tuple[str, str]:
        return (cls.__module__, cls.__qualname__)

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return []

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        supported_head_sizes = cls.get_supported_head_sizes()
        return (not supported_head_sizes) or head_size in supported_head_sizes

    @classmethod
    def supports_dtype(cls, dtype: torch.dtype) -> bool:
        return dtype in cls.supported_dtypes

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: "CacheDType | None") -> bool:
        if kv_cache_dtype is None:
            return True
        return (not cls.supported_kv_cache_dtypes) or (
            kv_cache_dtype in cls.supported_kv_cache_dtypes
        )

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        from vllm.config.cache import BlockSize

        if block_size is None:
            return True

        valid_sizes = get_args(BlockSize)
        if block_size not in valid_sizes:
            return False

        supported_kernel_block_sizes = cls.get_supported_kernel_block_sizes()
        if not supported_kernel_block_sizes:
            return True

        for supported_size in supported_kernel_block_sizes:
            if isinstance(supported_size, MultipleOf):
                supported_size = supported_size.base
            # With hybrid_blocks feature, the framework-level block size
            # only needs to be a multiple of the kernel's requirement,
            # even if the kernel requires a fixed block_size.
            if block_size % supported_size == 0:
                return True
        return False

    @classmethod
    def is_mla(cls) -> bool:                                                                        #是否支持Mla
        return False

    @classmethod
    def supports_sink(cls) -> bool:                                                                 #是否支持attention sink(用于某些长文本优化)
        return False

    @classmethod
    def supports_mm_prefix(cls) -> bool:                                                            #是否支持多模态前缀
        return False

    @classmethod
    def is_sparse(cls) -> bool:                                                                     #当前backend是否为稀疏attention
        return False

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """Check if backend supports a given attention type.                                        检查是否支持某种attention类型

        By default, only supports decoder attention.                                                默认只支持decoder attention(即标准的自回归attention) 其他类型(如encoder cross attention)需要子类重写此方法
        Backends should override this to support other attention types.
        """
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_compute_capability(cls, capability: "DeviceCapability") -> bool:
        return True

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: "CacheDType | None",
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        device_capability: "DeviceCapability",
    ) -> str | None:
        """检查一组配置组合是否被支持 如果不支持返回一个描述原因的字符串 如果支持返回None"""
        return None

    @classmethod
    def validate_configuration(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: "CacheDType | None",
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: "DeviceCapability",
        attn_type: str,
    ) -> list[str]:
        """
        检验当前backend是否支持给定的模型配置
        返回一个字符串列表 列出所有不支持的原因
        """
        
        invalid_reasons = []
        if not cls.supports_head_size(head_size):
            invalid_reasons.append("head_size not supported")
        if not cls.supports_dtype(dtype):
            invalid_reasons.append("dtype not supported")
        if not cls.supports_kv_cache_dtype(kv_cache_dtype):
            invalid_reasons.append("kv_cache_dtype not supported")
        if not cls.supports_block_size(block_size):
            invalid_reasons.append("block_size not supported")
        if use_mm_prefix and not cls.supports_mm_prefix():
            invalid_reasons.append(
                "partial multimodal token full attention not supported"
            )
        if use_mla != cls.is_mla():
            if use_mla:
                invalid_reasons.append("MLA not supported")
            else:
                invalid_reasons.append("non-MLA not supported")
        if has_sink and not cls.supports_sink():
            invalid_reasons.append("sink setting not supported")
        if use_sparse != cls.is_sparse():
            if use_sparse:
                invalid_reasons.append("sparse not supported")
            else:
                invalid_reasons.append("non-sparse not supported")
        if not cls.supports_compute_capability(device_capability):
            invalid_reasons.append("compute capability not supported")
        if not cls.supports_attn_type(attn_type):
            invalid_reasons.append(f"attention type {attn_type} not supported")
        combination_reason = cls.supports_combination(
            head_size,
            dtype,
            kv_cache_dtype,
            block_size,
            use_mla,
            has_sink,
            use_sparse,
            device_capability,
        )
        if combination_reason is not None:
            invalid_reasons.append(combination_reason)
        return invalid_reasons

    @classmethod
    def get_required_kv_cache_layout(cls) -> "KVCacheLayoutType | None":
        return None


class AttentionMetadata:
    pass


T = TypeVar("T", bound=AttentionMetadata)


class AttentionLayer(Protocol):
    _q_scale: torch.Tensor
    _k_scale: torch.Tensor
    _v_scale: torch.Tensor
    _q_scale_float: float
    _k_scale_float: float
    _v_scale_float: float
    _prob_scale: torch.Tensor

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor: ...


class AttentionImpl(ABC, Generic[T]):
    # Whether the attention impl can return the softmax lse for decode.                         当前attention实现是否能够在decode阶段返回softmax的LSE(LogSumExp,通常用于数值稳定的softmax计算)
    # Some features like decode context parallelism require the softmax lse.
    can_return_lse_for_decode: bool = False

    # Whether the attention impl supports Prefill Context Parallelism.
    supports_pcp: bool = False
    # Whether the attention impl(or ops) supports MTP
    # when cp_kv_cache_interleave_size > 1                                                      当cp_kv_cache_interleave_size > 1时,当前attention实现(或底层算子)是否支持MTP
    supports_mtp_with_cp_non_trivial_interleave_size: bool = False

    # some attention backends might not always want to return lse
    # even if they can return lse (for efficiency reasons)
    need_to_return_lse_for_decode: bool = False

    # Whether this attention implementation supports pre-quantized query input.                 当前attention实现是否支持预量化的query输入
    # When True, the attention layer will quantize queries before passing them                  当为True attention layer会在query传递给当前backend之前 先对query量化
    # to this backend, allowing torch.compile to fuse the quantization with                     这样torch.compile就可以把query的量化操作与前面的算子融合在一起 从而提升性能
    # previous operations. This is typically supported when using FP8 KV cache                  这种能力通常出现在使用fp8 kv cache且attention kernel兼容fp8的场景中
    # with compatible attention kernels (e.g., TRT-LLM).                                        注意不是prompt是query
    # Subclasses should set this in __init__.
    # TODO add support to more backends:
    # https://github.com/vllm-project/vllm/issues/25584
    supports_quant_query_input: bool = False

    dcp_world_size: int
    dcp_rank: int

    pcp_world_size: int
    pcp_rank: int

    total_cp_world_size: int
    total_cp_rank: int

    def __new__(cls, *args, **kwargs):
        # use __new__ so that all subclasses will call this
        self = super().__new__(cls)
        try:
            from vllm.distributed.parallel_state import get_dcp_group

            self.dcp_world_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
        except AssertionError:
            # DCP might not be initialized in testing
            self.dcp_world_size = 1
            self.dcp_rank = 0
        try:
            from vllm.distributed.parallel_state import get_pcp_group

            self.pcp_world_size = get_pcp_group().world_size
            self.pcp_rank = get_pcp_group().rank_in_group
        except AssertionError:
            self.pcp_world_size = 1
            self.pcp_rank = 0
        self.total_cp_world_size = self.pcp_world_size * self.dcp_world_size
        self.total_cp_rank = self.pcp_rank * self.dcp_world_size + self.dcp_rank

        self.need_to_return_lse_for_decode = (
            self.dcp_world_size > 1 and self.can_return_lse_for_decode
        )
        return self

    @abstractmethod
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: T,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def fused_output_quant_supported(self, quant_key: "QuantKey"):
        """
        Does this attention implementation support fused output quantization.
        This is used by the AttnFusionPass to only fuse output quantization
        onto implementations that support it.

        :param quant_key: QuantKey object that describes the quantization op
        :return: is fusion supported for this type of quantization
        """
        return False

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        pass


class MLAAttentionImpl(AttentionImpl[T], Generic[T]):
    @abstractmethod
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        q_lora_rank: int | None,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        qk_head_dim: int,
        v_head_dim: int,
        kv_b_proj: "ColumnParallelLinear",
        indexer: object | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        layer: AttentionLayer,
        hidden_states_or_cq: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: T,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError


def is_quantized_kv_cache(kv_cache_dtype: str) -> bool:
    return kv_cache_dtype != "auto"
