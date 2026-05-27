# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import numpy as np
import torch
import torch.distributed as dist

from vllm.config import ParallelConfig
from vllm.distributed.parallel_state import get_dp_group
from vllm.logger import init_logger
from vllm.v1.worker.ubatch_utils import (
    check_ubatch_thresholds,
    is_last_ubatch_empty,
)

logger = init_logger(__name__)


def _get_device_and_group(parallel_config: ParallelConfig):                                     #该函数其实就一件事 决定DP同步时,到底是GPU通信还是CPU通信
    # Use the actual device assigned to the DP group, not just the device type
    device = get_dp_group().device                                                              #获取当前dp group 实际使用的device  
    group = get_dp_group().device_group                                                         #获取DP通信group  可以把group理解成那些GPU之间允许互相通信 dist.all_reduce就在这个group内通信

    # Transferring this tensor from GPU to CPU will introduce a GPU sync                        将这个tensor从GPU拷贝到CPU时, 会引入一个GPU同步点(sync point)
    # point that could adversely affect performance of vllm with asynch                         这个同步会打断原本的异步执行流程,从而可能降低vllm在异步调度下的性能
    # scheduling. This environment variable exists to quickly disable                           设置这个环境变量的目的,是为了在遇到这种性能问题时 可以快速关闭这个优化逻辑
    # this optimization if we run into this case.
    if parallel_config.disable_nccl_for_dp_synchronization:
        logger.info_once(
            "Using CPU all reduce to synchronize DP padding between ranks."
        )
        device = "cpu"
        group = get_dp_group().cpu_group
    return device, group


def _run_ar(
    should_ubatch: bool,
    should_dp_pad: bool,
    orig_num_tokens_per_ubatch: int,
    padded_num_tokens_per_ubatch: int,
    cudagraph_mode: int,
    parallel_config: ParallelConfig,
) -> torch.Tensor:
    dp_size = parallel_config.data_parallel_size                                                    
    dp_rank = parallel_config.data_parallel_rank
    device, group = _get_device_and_group(parallel_config)                                      #这个组也不是8卡就分成2个通信组,而是根据配置的
    
     # 这个 tensor 本质上是:
    #      rank0 rank1 rank2 rank3                                                              每一列代表一个dp rank  每一行代表一种状态信息  
    # row0   ?     ?     ?     ?
    # row1   ?     ?     ?     ?
    # row2   ?     ?     ?     ?
    # row3   ?     ?     ?     ?
    # row4   ?     ?     ?     ?
    tensor = torch.zeros(5, dp_size, device=device, dtype=torch.int32)
    tensor[0][dp_rank] = orig_num_tokens_per_ubatch                                             #row0 当前rank的原始token数
    tensor[1][dp_rank] = padded_num_tokens_per_ubatch                                           #row1 当前rank padding后的token数
    tensor[2][dp_rank] = 1 if should_ubatch else 0                                              #row2 当前rank是否想启用microbatch(ubatch)
    tensor[3][dp_rank] = 1 if should_dp_pad else 0                                              #row3 当前rank是否允许DP padding
    tensor[4][dp_rank] = cudagraph_mode                                                         #row4 当前rank的cudagraph mode
    dist.all_reduce(tensor, group=group)                                                        #当前tensor只有自己那一列有值,这一步就是所有GPU交换自己的状态信息 
    return tensor


def _post_process_ubatch(tensor: torch.Tensor, num_ubatches: int) -> bool:
    orig_num_tokens_tensor = tensor[0, :]                                                       #所有DP rank的原始token数 举例tensor[0] = [120,80,140,100]
    padded_num_tokens_tensor = tensor[1, :]                                                     #所有DP rank的padding后token数

    # First determine if we are going to be ubatching.                                          判断是否所有rank都同意ubatch
    should_ubatch: bool = bool(torch.all(tensor[2] == 1).item())
    if not should_ubatch:
        return False
    # If the DP ranks are planning to ubatch, make sure that                                    即使所有rank都想ubatch 也要检查拆完后会不会出现空Ubatch  
    # there are no "empty" second ubatches
    orig_min_num_tokens = int(orig_num_tokens_tensor.min().item())                              #最小rank最容易出现空ubatch
    padded_max_num_tokens = int(padded_num_tokens_tensor.max().item())
    if is_last_ubatch_empty(orig_min_num_tokens, padded_max_num_tokens, num_ubatches):
        logger.debug(
            "Aborting ubatching %s %s", orig_min_num_tokens, padded_max_num_tokens
        )
        should_ubatch = False
    return should_ubatch


def _post_process_dp_padding(tensor: torch.Tensor, should_dp_pad: bool) -> torch.Tensor:
    num_tokens_across_dp = tensor[1, :]
    if should_dp_pad:
        # If DP padding is enabled, ensure that each rank is processing the same number
        # of tokens
        max_num_tokens = int(num_tokens_across_dp.max().item())
        return torch.tensor(
            [max_num_tokens] * len(num_tokens_across_dp),
            device="cpu",
            dtype=torch.int32,
        )
    else:
        return num_tokens_across_dp.cpu()


def _post_process_cudagraph_mode(tensor: torch.Tensor) -> int:
    """
    Synchronize cudagraph_mode across DP ranks by taking the minimum.
    If any rank has NONE (0), all ranks use NONE.
    This ensures all ranks send consistent values (all padded or all unpadded).
    """
    return int(tensor[4, :].min().item())


def _synchronize_dp_ranks(
    num_tokens_unpadded: int,
    num_tokens_padded: int,
    should_attempt_ubatching: bool,
    should_attempt_dp_padding: bool,
    cudagraph_mode: int,
    parallel_config: ParallelConfig,
) -> tuple[bool, torch.Tensor | None, int]:
    """
    1. Decides if each DP rank is going to microbatch. Either all ranks                             决定每个dp rank是否用mirobatch  注意要么所有rank都启用  不允许一部分开启一部分关闭 否则会导致batch shape不一致,collective通信不匹配 cudagraph失败等
    run with microbatching or none of them do.

    2. Determines the total number of tokens that each rank will run.                               确定每个rank最终实际执行的token数 当满足以下任一条件时 使用microbatch should_attempt_dp_padding=True
    When running microbatched or if should_attempt_dp_padding is True, all                          所有rank都会被padding 从而保证所有rank使用相同token数执行 ,这样可以确保batch shape对齐  CUDA Graph可复用  collective操作安全
    ranks will be padded out so that the run with the same number of tokens

    3. Synchronizes cudagraph_mode across ranks by taking the minimum.                              在所有rank之间同步cudagraph_mode 取所有rank之间的最小值

    Returns: tuple[
        should_ubatch: Are all DP ranks going to microbatch                                         所有dp rank是否最终决定用microbatch
        num_tokens_after_padding: A tensor containing the total number of                           一个tensor 表示每个dp rank在padding后  每个microbatch的token总数   其中已经包含dp padding
        tokens per-microbatch for each DP rank including any DP padding.
        synced_cudagraph_mode: The synchronized cudagraph mode (min across ranks)                   dp ranks同步后的cudagraph mode(所有rank的最小值)
    ]

    """
    assert num_tokens_padded >= num_tokens_unpadded

    # Coordinate between the DP ranks via an All Reduce                                             通过一次all reduce在所有dp rank之间进行协调
    # to determine the total number of tokens that each rank                                        用来确定每个rank最终需要执行的token数
    # will run and if we are using ubatching or not.                                                是否启用ubatching(micro-batching/微批处理)
    tensor = _run_ar(
        should_ubatch=should_attempt_ubatching,
        should_dp_pad=should_attempt_dp_padding,
        orig_num_tokens_per_ubatch=num_tokens_unpadded,
        padded_num_tokens_per_ubatch=num_tokens_padded,
        cudagraph_mode=cudagraph_mode,
        parallel_config=parallel_config,
    )

    should_dp_pad = bool(torch.all(tensor[3] == 1).item())                                          #检查所有rank是否都开启dp padding

    # DP ranks should all have the same value for should_attempt_dp_padding.                        #所有dp rank必须对是否允许dp padding保持一致 
    assert should_attempt_dp_padding == should_dp_pad

    # Check conditions for microbatching
    should_ubatch = _post_process_ubatch(tensor, parallel_config.num_ubatches)

    if should_ubatch and not should_dp_pad:
        logger.debug_once(
            "Microbatching has been triggered and requires DP padding. "
            "Enabling DP padding even though it has been explicitly "
            "disabled.",
            scope="global",
        )
        should_dp_pad = True

    # Pad all DP ranks up to the maximum token count across ranks if                                 统一所有rank数
    # should_dp_pad is True
    num_tokens_after_padding = _post_process_dp_padding(
        tensor,
        should_dp_pad,
    )

    # Synchronize cudagraph_mode across ranks (take min)                                             所有rank取最小值
    synced_cudagraph_mode = _post_process_cudagraph_mode(tensor)

    return should_ubatch, num_tokens_after_padding, synced_cudagraph_mode                            #返回是否ubatch  padding后token数  同步后的cudagraph mode


def coordinate_batch_across_dp(
    num_tokens_unpadded: int,
    allow_microbatching: bool,
    allow_dp_padding: bool,
    parallel_config: ParallelConfig,
    num_tokens_padded: int | None = None,
    uniform_decode: bool | None = None,
    num_scheduled_tokens_per_request: np.ndarray | None = None,
    cudagraph_mode: int = 0,
) -> tuple[bool, torch.Tensor | None, int]:
    """
    Coordinates amongst all DP ranks to determine if and how the full batch                          在所有data parallel rank之间进行协调,用于决定整个batch是否需要拆分成microbatch以及如何拆分
    should be split into microbatches.

    Args:
        num_tokens_unpadded: Number of tokens without accounting for padding                         未进行任何padding时的token总数
        allow_microbatching: If microbatching should be attempted                                    是否允许尝试进行microbatch拆分
        allow_dp_padding: If all DP ranks should be padded up to the same value                      是否允许所有dp rank padding到相同token数
        parallel_config: The parallel config                                                    
        num_tokens_padded: Number of tokens including any non-DP padding (CUDA graphs,               已经包含非DP padding后的token数  这些padding可能来自cuda graph对齐  tensor parallel对齐  其他shape对齐
            TP, etc)
        uniform_decode: Only used if allow_microbatching is True. True if the batch                  仅在allow_microbatching为True时使用,如果Batch中全部是单token decode 则为True
            only contains single token decodes
        num_scheduled_tokens_per_request: Only used if allow_microbatching is True. The              仅在allow_moribatching=True使 表示每个request对应的token数
            number of tokens per request.
        cudagraph_mode: The cudagraph mode for this rank (0=NONE, 1=PIECEWISE, 2=FULL)

    Returns: tuple[
        ubatch_slices: if this is set then all DP ranks have agreed to
        microbatch
        num_tokens_after_padding: A tensor containing the total number of
        tokens per-microbatch for each DP rank including padding. Will be
        padded up to the max value across all DP ranks when allow_dp_padding
        is True.
        synced_cudagraph_mode: The synchronized cudagraph mode (min across ranks)
    ]

    """
    if parallel_config.data_parallel_size == 1:
        # Early exit.
        return False, None, cudagraph_mode

    # If the caller has explicitly enabled microbatching.
    should_attempt_ubatching = False
    if allow_microbatching:
        # Check preconditions for microbatching
        assert uniform_decode is not None
        should_attempt_ubatching = check_ubatch_thresholds(                                             #检查是否值得ubatch  不是所有batch都适合ubatch
            parallel_config,
            num_tokens_unpadded,
            uniform_decode=uniform_decode,
        )

    if num_tokens_padded is None:
        num_tokens_padded = num_tokens_unpadded

    (should_ubatch, num_tokens_after_padding, synced_cudagraph_mode) = (
        _synchronize_dp_ranks(
            num_tokens_unpadded,
            num_tokens_padded,
            should_attempt_ubatching,
            allow_dp_padding,
            cudagraph_mode,
            parallel_config,
        )
    )

    return (should_ubatch, num_tokens_after_padding, synced_cudagraph_mode)
