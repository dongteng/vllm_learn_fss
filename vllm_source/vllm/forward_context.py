# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, NamedTuple

import torch

import vllm.envs as envs
from vllm.attention.backends.abstract import AttentionMetadata
from vllm.config import CUDAGraphMode, ParallelConfig, VllmConfig
from vllm.logger import init_logger
from vllm.v1.worker.dp_utils import coordinate_batch_across_dp
from vllm.v1.worker.ubatch_utils import UBatchSlices

logger = init_logger(__name__)

track_batchsize: bool = envs.VLLM_LOG_BATCHSIZE_INTERVAL >= 0
last_logging_time: float = 0
forward_start_time: float = 0
batchsize_logging_interval: float = envs.VLLM_LOG_BATCHSIZE_INTERVAL
batchsize_forward_time: defaultdict = defaultdict(list)


class BatchDescriptor(NamedTuple):
    """
    Batch descriptor for cudagraph dispatching. We should keep the num of                       这是给cudagraph调度用的描述符,要求信息尽量少(避免组合爆炸),但又能唯一标识一个batch
    items as minimal as possible to properly and uniquely describe the padded                   因为cudagraph要求shape完全一致才能复用
    batch for cudagraph.
    """

    num_tokens: int
    num_reqs: int | None = None
    """
    Number of requests in the batch. Can be None for PIECEWISE cudagraphs where
    the cudagraphs can handle any number of requests.
    """
    uniform: bool = False
    """
    True if all the requests in the batch have the same number of tokens.
    """
    has_lora: bool = False
    """
    Whether this batch has active LoRA adapters.
    """

    def relax_for_mixed_batch_cudagraphs(self) -> "BatchDescriptor":
        """
        Return a relaxed version of current batch descriptor that is still compatible            返回当前batch描述符的一个放宽版本,该版本仍然可以兼容PIECEWISE cudagraph(或混合prefill-decode的FA cudagraph)
        with PIECEWISE cudagraphs (or mixed prefill-decode FA cudagraphs).
        """
        return BatchDescriptor(
            self.num_tokens, num_reqs=None, uniform=False, has_lora=self.has_lora
        )


def _compute_sp_num_tokens(
    num_tokens_across_dp_cpu: torch.Tensor, sequence_parallel_size: int
) -> list[int]:
    """
    把各个rank的token数据 按照sp size平均分一下 然后把结果复制给对应的sp rank
    
    """
    
    sp_tokens = (
        num_tokens_across_dp_cpu + sequence_parallel_size - 1                                   #这里其实就是ceil向上取整
    ) // sequence_parallel_size

    sp_tokens = sp_tokens.repeat_interleave(sequence_parallel_size)                             #将每个dp rank对应的sp token数 复制给该dp rank下的所有sp rank
    return sp_tokens.tolist()


def _compute_chunked_local_num_tokens(
    num_tokens_across_dp_cpu: torch.Tensor,
    sequence_parallel_size: int,
    max_num_tokens: int,
    chunk_idx: int,
) -> list[int]:
    """
    例 rank0=120, rank1=80,rank2=140,rank3=100 and max_num_tokens=64
    then the tokens are processed in chunks
    rank0:[64,56]  rank1:[64,16]  rank2:[64,64,12] rank3:[64,36]
    chunk_idx=0, local_sizes=[64,64,64,64]  chunk_idx=1 local_sizes=[56,16,64,36]
    chunk_idx=2  raw result=[-8,-48,12,-28]
    最终local_sizes = [1,1,12,1]
    """
    
    sp_tokens = _compute_sp_num_tokens(num_tokens_across_dp_cpu, sequence_parallel_size)
    sp_size = len(sp_tokens)

    local_size = [-1] * sp_size
    for i in range(sp_size):
        # Take into account sharding if MoE activation is sequence parallel.
        local_size[i] = min(max_num_tokens, sp_tokens[i] - (max_num_tokens * chunk_idx))
        if local_size[i] <= 0:
            local_size[i] = 1  # ensure lockstep even if done
    return local_size


@dataclass
class DPMetadata:
    max_tokens_across_dp_cpu: torch.Tensor                                                              #所有dp rank中最大的token数
    num_tokens_across_dp_cpu: torch.Tensor                                                              #各dp rank实际拥有的token数 tensor([120, 80, 140]) 表示rank0有120...

    # NOTE: local_sizes should only be set by the chunked_sizes context manager                         #当前rank在进行chunk切分后每个chunk的大小
    local_sizes: list[int] | None = None

    @staticmethod
    def make(
        parallel_config: ParallelConfig,
        num_tokens: int,
        num_tokens_across_dp_cpu: torch.Tensor,
    ) -> "DPMetadata":
        assert num_tokens_across_dp_cpu is not None
        assert parallel_config.data_parallel_size > 1
        dp_rank = parallel_config.data_parallel_rank
        batchsize = num_tokens

        # If num_tokens_across_dp is None, it will be computed by all_reduce
        # Otherwise, num_tokens_across_dp[dp_rank] should be equal to batchsize
        assert num_tokens_across_dp_cpu[dp_rank] == batchsize, (
            f"{num_tokens_across_dp_cpu[dp_rank]} {batchsize}"
        )
        max_tokens_across_dp_cpu = torch.max(num_tokens_across_dp_cpu)
        return DPMetadata(max_tokens_across_dp_cpu, num_tokens_across_dp_cpu)

    @contextmanager
    def chunked_sizes(
        self, sequence_parallel_size: int, max_chunk_size_per_rank: int, chunk_idx: int
    ):
        """
        Context manager to compute and temporarily set the per-rank local token                         一个上下文管理器,用在chunked forward期间 计算并临时设置某个chunk对应的各个rank本地token数
        sizes for a specific chunk during chunked forward execution.

        This is necessary to ensure each DP (data parallel) rank processes its                          这样做的目的是保证所有dp rank 以同步(lockstep)的方式 处理各自负责的token 即使不同rank的token数量不一致
        designated portion of tokens in lockstep with others, even when the                             某些rank已经提前处理完自己的输入 也能保持各rank的执行步调一致
        token counts are uneven or some ranks have completed their input early.

        For chunked execution, we break up the total tokens on each rank into                           对于chunked execution  每个rank上的token会被拆分成多个chunk  不超过max_chunk_size_per_rank
        multiple chunks (of at most `max_chunk_size_per_rank`), and for a given                         对于指定的chunk_idx  该context manager会计算self.local_sizes
        `chunk_idx`, this context manager sets `self.local_sizes` to the number                         并将其设置为:当前chunk中,每个rank需要处理的token数量
        of tokens to process in that chunk on each rank.

        `self.local_sizes` is only valid inside the context.

        Args:
            sequence_parallel_size: When Attn is TP and MoE layers are EP,                              当attn使用tp 而MoE使用ep, vllm会在二者之间引入sp 以避免重复计算 这里需要参数来正确计算chunk的大小
                                    we use SP between the layers to avoid
                                    redundant ops. We need this value to
                                    compute the chunked sizes.
            max_chunk_size_per_rank: The max number of tokens each rank is                              当前chunk中,每个rank最多允许处理的token数量
                                     allowed to process in this chunk.
            chunk_idx: The index of the chunk to compute sizes for.
        """
        self.local_sizes = _compute_chunked_local_num_tokens(
            self.num_tokens_across_dp_cpu,
            sequence_parallel_size,
            max_chunk_size_per_rank,
            chunk_idx,
        )
        try:
            yield self.local_sizes
        finally:
            self.local_sizes = None

    @contextmanager
    def sp_local_sizes(self, sequence_parallel_size: int):
        """
        Context manager for setting self.local_sizes. Same as self.chunked_sizes
        but without any chunking.
        """
        self.local_sizes = _compute_sp_num_tokens(
            self.num_tokens_across_dp_cpu, sequence_parallel_size
        )
        try:
            yield self.local_sizes
        finally:
            self.local_sizes = None

    def get_chunk_sizes_across_dp_rank(self) -> list[int] | None:
        assert self.local_sizes is not None
        return self.local_sizes

    # Get the cumulative tokens across sequence parallel ranks.                                         获取sp维度上的累计token数
    # In this case the input to the MoEs will be distributed w.r.t both                                 在这种情况下 MoE的输入会同时按照dp 和tp/sp进行分布
    # DP and TP rank.
    # When sp_size==1, this is just the cummulative num tokens across DP.                               当sp_size==1 没有p结果就退化为dp各rank token数的累计和
    def cu_tokens_across_sp(self, sp_size: int) -> torch.Tensor:
        num_tokens_across_sp_cpu = (
            self.num_tokens_across_dp_cpu - 1 + sp_size
        ) // sp_size
        num_tokens_across_sp_cpu = num_tokens_across_sp_cpu.repeat_interleave(sp_size)
        return torch.cumsum(num_tokens_across_sp_cpu, dim=0)


@dataclass
class ForwardContext:                                                                                   #定义forward(一次模型前向计算)的上下文
    # copy from vllm_config.compilation_config.static_forward_context                                   可以理解为是一个结构体专门装数据
    no_compile_layers: dict[str, Any]                                                                   #哪些层不参与编译优化(比如cudagraph / torchcompile)
    attn_metadata: dict[str, AttentionMetadata] | list[dict[str, AttentionMetadata]]                    #attention的元数据 核心字段值一
    """
    Type Dict[str, AttentionMetadata] for v1, map from layer_name of each                               v1每一层attention都有对应metadata,比如kv cache位置, seq_len, block table
    attention layer to its attention metadata
    Type List[Dict[str, AttentionMetadata]] for DBO. List of size two, one                              DBO一个forward拆成多个microbatch,每个microbatch有独立attn metadata
    for each microbatch.
    Set dynamically for each forward pass
    """
    # TODO: remove after making all virtual_engines share the same kv cache                             未来计划删除这个字段
    virtual_engine: int  # set dynamically for each forward pass                                        #当前使用的虚拟引擎编号 用于多engine并行推理,不同engine可能不同kv cache
    # set dynamically for each forward pass
    dp_metadata: DPMetadata | None = None                                                               #数据并行相关信息:rank信息 shard信息  同步策略
    # determine the cudagraph style at runtime to be FULL, PIECEWISE, or NONE.
    # by default NONE, no cudagraph is used.
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE
    batch_descriptor: BatchDescriptor | None = None

    ubatch_slices: UBatchSlices | None = None

    def __post_init__(self):
        assert self.cudagraph_runtime_mode.valid_runtime_modes(), (
            f"Invalid cudagraph runtime mode: {self.cudagraph_runtime_mode}"
        )


_forward_context: ForwardContext | None = None


def get_forward_context() -> ForwardContext:
    """Get the current forward context."""
    assert _forward_context is not None, (
        "Forward context is not set. "
        "Please use `set_forward_context` to set the forward context."
    )
    return _forward_context


def is_forward_context_available() -> bool:
    return _forward_context is not None


def create_forward_context(
    attn_metadata: Any,
    vllm_config: VllmConfig,
    virtual_engine: int = 0,
    dp_metadata: DPMetadata | None = None,
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    batch_descriptor: BatchDescriptor | None = None,
    ubatch_slices: UBatchSlices | None = None,
):
    return ForwardContext(
        no_compile_layers=vllm_config.compilation_config.static_forward_context,
        virtual_engine=virtual_engine,
        attn_metadata=attn_metadata,
        dp_metadata=dp_metadata,
        cudagraph_runtime_mode=cudagraph_runtime_mode,
        batch_descriptor=batch_descriptor,
        ubatch_slices=ubatch_slices,
    )


@contextmanager
def override_forward_context(forward_context: ForwardContext | None):
    """A context manager that overrides the current forward context.
    This is used to override the forward context for a specific
    forward pass.
    """
    global _forward_context
    prev_context = _forward_context
    _forward_context = forward_context
    try:
        yield
    finally:
        _forward_context = prev_context


@contextmanager
def set_forward_context(
    attn_metadata: Any,
    vllm_config: VllmConfig,
    virtual_engine: int = 0,
    num_tokens: int | None = None,
    num_tokens_across_dp: torch.Tensor | None = None,
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    batch_descriptor: BatchDescriptor | None = None,
    ubatch_slices: UBatchSlices | None = None,
):
    """A context manager that stores the current forward context,                               一个用于保存当前forward上下文的上下文管理器
    can be attention metadata, etc.                                                             这个上下文中可以包含Attention Metadata等信息
    Here we can inject common logic for every model forward pass.                               在这里可以为每一次模型forward注入通用逻辑
    """
    global forward_start_time
    need_to_track_batchsize = track_batchsize and attn_metadata is not None                     #是否开启batch size性能统计 用于观察不同batch size的forward latency
    if need_to_track_batchsize:
        forward_start_time = time.perf_counter()

    dp_metadata: DPMetadata | None = None
    if vllm_config.parallel_config.data_parallel_size > 1 and (
        attn_metadata is not None or num_tokens is not None
    ):
        # If num_tokens_across_dp hasn't already been initialized, then
        # initialize it here. Both DP padding and Microbatching will be
        # disabled.
        if num_tokens_across_dp is None:
            assert ubatch_slices is None
            assert num_tokens is not None
            _, num_tokens_across_dp, _ = coordinate_batch_across_dp(
                num_tokens_unpadded=num_tokens,
                parallel_config=vllm_config.parallel_config,
                allow_microbatching=False,
                allow_dp_padding=False,
            )
            assert num_tokens_across_dp is not None
        dp_metadata = DPMetadata.make(
            vllm_config.parallel_config, num_tokens or 0, num_tokens_across_dp
        )

    # Convenience: if cudagraph is used and num_tokens is given, we can just
    # create a batch descriptor here if not given (there's no harm since if it
    # doesn't match in the wrapper it'll fall through).
    if cudagraph_runtime_mode != CUDAGraphMode.NONE and num_tokens is not None:
        batch_descriptor = batch_descriptor or BatchDescriptor(num_tokens=num_tokens)

    forward_context = create_forward_context(
        attn_metadata,
        vllm_config,
        virtual_engine,
        dp_metadata,
        cudagraph_runtime_mode,
        batch_descriptor,
        ubatch_slices,
    )

    try:
        with override_forward_context(forward_context):
            yield
    finally:
        global last_logging_time, batchsize_logging_interval
        if need_to_track_batchsize:
            batchsize = num_tokens
            # we use synchronous scheduling right now,
            # adding a sync point here should not affect
            # scheduling of the next batch
            from vllm.platforms import current_platform

            synchronize = current_platform.synchronize
            if synchronize is not None:
                synchronize()
            now = time.perf_counter()
            # time measurement is in milliseconds
            batchsize_forward_time[batchsize].append((now - forward_start_time) * 1000)
            if now - last_logging_time > batchsize_logging_interval:
                last_logging_time = now
                forward_stats = []
                for bs, times in batchsize_forward_time.items():
                    if len(times) <= 1:
                        # can be cudagraph / profiling run
                        continue
                    medium = torch.quantile(torch.tensor(times), q=0.5).item()
                    medium = round(medium, 2)
                    forward_stats.append((bs, len(times), medium))
                forward_stats.sort(key=lambda x: x[1], reverse=True)
                if forward_stats:
                    logger.info(
                        (
                            "Batchsize forward time stats "
                            "(batchsize, count, median_time(ms)): %s"
                        ),
                        forward_stats,
                    )
