# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, overload

from vllm.distributed.kv_events import KVCacheEvent
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_coordinator import get_kv_cache_coordinator
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class KVCacheBlocks:
    """
    The allocation result of KVCacheManager, work as the interface between
    Scheduler and KVCacheManager, to hide KVCacheManager's internal data
    structure from the Scheduler.
    """

    blocks: tuple[Sequence[KVCacheBlock], ...]
    """
    `blocks[i][j]` refers to the i-th kv_cache_group
    and the j-th block of tokens.We don't use block of
    tokens as the outer dimension because it assumes all
    kv_cache_groups have the same number of blocks, which is true for now but
    will be broken if we want to give different block_size to different
    kv_cache_groups in the future.

    Each single type KVCacheBlocks could be represented as:
    - list[KVCacheBlock] for more than one KVCacheBlock
    - an empty tuple for requests without KVCacheBlock
      (a precomputed KVCacheBlocks is in KVCacheManager to avoid GC overhead)
    """

    def __add__(self, other: "KVCacheBlocks") -> "KVCacheBlocks":
        """Adds two KVCacheBlocks instances."""
        return KVCacheBlocks(
            tuple(
                list(itertools.chain(blk1, blk2))
                for blk1, blk2 in zip(self.blocks, other.blocks)
            )
        )

    @overload
    def get_block_ids(
        self,
        allow_none: Literal[False] = False,
    ) -> tuple[list[int], ...]: ...

    @overload
    def get_block_ids(
        self,
        allow_none: Literal[True] = True,
    ) -> tuple[list[int], ...] | None: ...

    def get_block_ids(
        self,
        allow_none: bool = False,
    ) -> tuple[list[int], ...] | None:
        """
        Converts the KVCacheBlocks instance to block_ids.

        Returns:
            tuple[list[int], ...]: A tuple of lists where:
                - the outer tuple corresponds to KV cache groups
                - each inner list contains the block_ids of the blocks in that
                  group
        """
        if allow_none and all(len(group) == 0 for group in self.blocks):
            return None
        return tuple([blk.block_id for blk in group] for group in self.blocks)

    def get_unhashed_block_ids(self) -> list[int]:
        """Get block_ids of unhashed blocks from KVCacheBlocks instance."""
        assert len(self.blocks) == 1, "Only one group is supported"
        return [block.block_id for block in self.blocks[0] if block.block_hash is None]

    def new_empty(self) -> "KVCacheBlocks":
        """
        Creates a new KVCacheBlocks instance with no blocks.
        """
        return KVCacheBlocks(tuple(() for _ in range(len(self.blocks))))


class KVCacheManager:
    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        hash_block_size: int,
        enable_caching: bool = True,
        use_eagle: bool = False,
        log_stats: bool = False,
        enable_kv_cache_events: bool = False,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ) -> None:
        self.max_model_len = max_model_len

        self.enable_caching = enable_caching
        self.use_eagle = use_eagle
        self.log_stats = log_stats
        self.metrics_collector = metrics_collector
        # FIXME: make prefix cache stats conditional on log_stats. We still need
        # this comment because when the log stats is enabled there are still
        # potential configs we could expose in the future.
        self.prefix_cache_stats = PrefixCacheStats() if log_stats else None

        self.coordinator = get_kv_cache_coordinator(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            use_eagle=self.use_eagle,
            enable_caching=self.enable_caching,
            enable_kv_cache_events=enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            metrics_collector=self.metrics_collector,
        )
        self.num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)
        self.block_pool = self.coordinator.block_pool
        self.kv_cache_config = kv_cache_config

        # Pre-constructed KVCacheBlocks with no blocks, callers should use this
        # via create_kv_cache_blocks instead of creating new ones to avoid GC
        # overhead.
        #
        # We use nested tuples to ensure the empty KVCacheBlocks is immutable.
        self.empty_kv_cache_blocks = KVCacheBlocks(
            tuple(() for _ in range(self.num_kv_cache_groups))
        )

    @property
    def usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).
        """
        return self.block_pool.get_usage()

    def make_prefix_cache_stats(self) -> PrefixCacheStats | None:
        """Get (and reset) the prefix cache stats.

        Returns:
            The current prefix caching stats, or None if logging is disabled.
        """
        if not self.log_stats:
            return None
        stats = self.prefix_cache_stats
        self.prefix_cache_stats = PrefixCacheStats()
        return stats

    def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int]:
        """Get the computed (cached) blocks for the request.                       获取该请求可以复用的已计算(cached) kv cache blocks
        Note that the computed blocks must be full.                                这是prefic caching机制的核心函数之一。它的作用是：判断当前请求的prompt前缀，有多少内容已经存在于系统的kv cache中
                                                                                   从而避免重复计算相同的prompt前缀，节省显存和计算量
        Args:                                                                   
            request: The request to get the computed blocks.

        Returns:                                                                   返回值：
            A tuple containing:                                                    tuple[KVCacheBlocks, int]:
                - A list of blocks that are computed for the request.                   - 第一项：可以直接复用的 KV Cache block 列表
                - The number of computed tokens.                                        - 本次可以命中缓存的token数量(num_new_computed_tokens)
        """
        # We skip finding the prefix cache hit when prefix caching is              # ====================== 1. 跳过前缀缓存检查的情况 ======================
        # disabled or the request is marked as skipping kv cache read              如果关闭了前缀缓存，或者请求明确标记不需要前缀缓存(如需要prompt logprobs，或者pooling模型一次性处理所有token)
        # (which happens when the request requires prompt logprobs                 则直接返回空结果
        # or calls a pooling model with all pooling).
        if not self.enable_caching or request.skip_reading_prefix_cache:
            return self.empty_kv_cache_blocks, 0

        # NOTE: When all tokens hit the cache, we must recompute the last token    # ====================== 2. 计算最大允许缓存命中长度 ======================
        # to obtain logits. Thus, set max_cache_hit_length to prompt_length - 1.   重要说明：即使前缀全部命中，我们也必须重新计算最后一个token，因为需要它来生成logits(下一个token的概率分布)
        # This can trigger recomputation of an entire block, rather than just      因此把这里最大缓存命中长度设置为prompt长度-1
        # the single last token, because allocate_slots() requires                 注意：由于block是按块对齐的，这可能会导致整个block会重新计算，而不是只重新计算最后一个token
        # num_computed_tokens to be block-size aligned. Removing this limitation   这是目前的一个局限，未来可能优化
        # could slightly improve performance in the future.
        max_cache_hit_length = request.num_tokens - 1
        
        
        computed_blocks, num_new_computed_tokens = (                                # ====================== 3. 查找最长前缀缓存命中 ======================
            self.coordinator.find_longest_cache_hit(                                # 通过coordinator(前缀缓存协调器)查找当前请求的block_hashes
                request.block_hashes, max_cache_hit_length                          #在已有的kv cache中能命中最长的前缀
            )                                                                       #这里还好奇prefill之前request.block_hashes的值是哪来的, 是在request构造时就有了,它描述的是如果这些token被分成block,会长什么样
        )

        if self.log_stats:                                                          # ====================== 4. 记录前缀缓存命中统计信息 ======================
            assert self.prefix_cache_stats is not None                              #用于监控前缀缓存的命中率，帮助分析系统性能
            self.prefix_cache_stats.record(
                num_tokens=request.num_tokens,                                      #请求总token数
                num_hits=num_new_computed_tokens,                                   #本次命中缓存的token数
                preempted=request.num_preemptions > 0,                              #是否发生过抢占
            )

        return self.create_kv_cache_blocks(computed_blocks), num_new_computed_tokens #把找到的block id列表包装成KVCacheBlocks对象返回

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
        num_lookahead_tokens: int = 0,
        num_external_computed_tokens: int = 0,
        delay_cache_blocks: bool = False,
        num_encoder_tokens: int = 0,
    ) -> KVCacheBlocks | None:
        """Add slots for a request with new tokens to append.                       为一个请求分配用于追加新token的kv cache槽位(slots)

        Args:
            request: The request to allocate slots.                                 需要分配槽位的请求
            num_new_tokens: The number of new tokens to be allocated and computed.  需要为其分配并计算新的token数量
            num_new_computed_tokens: The number of new computed tokens just         本次新命中prefix cache的token数量(不包含external tokens)
                hitting the prefix caching, excluding external tokens.
            new_computed_blocks: The cached blocks for the above new computed       上述命中 prefix cache 的 token 对应的缓存 block(按 KV cache group 分组)
                tokens, grouped as a tuple by kv cache groups.
            num_lookahead_tokens: The number of speculative tokens to allocate.     需要分配的“预读 token”(speculative decoding 使用，比如 eagle)
                This is used by spec decode proposers with kv-cache such
                as eagle.
            num_external_computed_tokens: The number of tokens that their           这些 token 的 KV cache 不在 vLLM 内部，而是在外部(通过 connector 提供)
                KV caches are not cached by vLLM but cached by the connector.
            delay_cache_blocks: Whether to skip caching the blocks. This is         是否延迟对 block 进行缓存(用于 KV 传输场景，例如 P/D 流程中，数据还没到)
                used by P/D when allocating blocks used in a KV transfer
                which will complete in a future step.
            num_encoder_tokens: The number of encoder tokens to allocate for        用于 encoder-decoder 模型(如 Whisper)中 cross-attention 的 encoder token 数量;
                cross-attention in encoder-decoder models(e.g., Whisper).
                For decoder-only models, this should be 0.

        Blocks layout:
        ```
         已经算的   新命中prefix cache    外部kv     需要新计算   预分配
        ----------------------------------------------------------------------
        | < comp > | < new_comp > | < ext_comp >  | < new >  | < lookahead > |
        ----------------------------------------------------------------------
                                                  |   < to be computed >     |
        ----------------------------------------------------------------------
                                  |            < to be allocated >           |
        ----------------------------------------------------------------------
                                  | < to be cached (roughly, |
                                  | details below)>          |
        ----------------------------------------------------------------------
        | Prefix-cached tokens from either vLLM   |
        | or connector. Can be safely removed if  |
        | they are outside sliding window.        |
        ----------------------------------------------------------------------
        |   < cached by vLLM >    | not cached by |
                                  | vLLM, but     |
        | ref_cnt  | ref_cnt not  | cached by     |
        | increased| increased yet| connector     |
        ----------------------------------------------------------------------
        ```

        Abbrivations:

        ```
        comp      = request.num_computed_tokens
        new_comp  = num_new_computed_tokens
                  = len(new_computed_blocks) * block_size
        ext_comp  = num_external_computed_tokens, cached by the connector
        new       = num_new_tokens, including unverified draft tokens
        lookahead = num_lookahead_tokens
        ```

        NOTE: for new tokens which include both verified and unverified draft
        tokens, we only cache the verified tokens (by capping the number at
        `request.num_tokens`).

        The allocation has three stages:
        - Free unnecessary blocks in `comp` and check
           if we have sufficient free blocks (return None if not).
        - Handle prefix tokens (`comp + new_comp + ext_comp`):
            - Free unnecessary blocks (e.g. outside sliding window)
            - Allocate new blocks for `ext_comp` tokens inside
              sliding window
        - Allocate new blocks for tokens to be computed (`new + lookahead`)

        Returns:
            A list of new allocated blocks.
        """
        # When loading KV data asynchronously, we may have zero new tokens to       当异步方式加载kv 数据，可能出现这种情况虽然没有新的 token 需要计算(num_new_tokens = 0)，
        # compute while still allocating slots for externally computed tokens.      但仍然需要为“来自外部、已计算好的 token”分配 KV cache 的槽位(blocks)。
        if num_new_tokens == 0 and num_external_computed_tokens == 0:
            raise ValueError(
                "num_new_tokens must be greater than 0 when there are no "
                "external computed tokens"
            )

        if new_computed_blocks is not None:                                         #获取本次新命中前缀缓存的block列表
            new_computed_block_list = new_computed_blocks.blocks
        else:
            new_computed_block_list = self.empty_kv_cache_blocks.blocks

        # The number of computed tokens is the number of computed tokens plus       # 本地已有的 + 本次新命中 prefix cache 的数量
        # the new prefix caching hits
        num_local_computed_tokens = (
            request.num_computed_tokens + num_new_computed_tokens
        )
        total_computed_tokens = min(                                                # 总已计算数量(本地 + 外部)，受限于模型最大长度
            num_local_computed_tokens + num_external_computed_tokens,
            self.max_model_len,
        )
        num_tokens_need_slot = min(                                                 # 最终总共需要占用槽位的 token 数量(已有的 + 本次要算的 + 预读的)
            total_computed_tokens + num_new_tokens + num_lookahead_tokens,
            self.max_model_len,
        )

        # Free the blocks that are skipped during the attention computation         #释放不再需要的block (sliding window机制)
        # (e.g., tokens outside the sliding window).                                在分配新块前先移除超出滑动窗口的块，以腾出空间，减少因空间不足导致的驱逐
        # We can do this even if we cannot schedule this request due to
        # insufficient free blocks.
        # Should call this function before allocating new blocks to reduce
        # the number of evicted blocks.
        self.coordinator.remove_skipped_blocks(
            request.request_id, total_computed_tokens
        )

        num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(       #向协调器询问：要达到目标token数量，还需要往外从物理池分配多少个block
            request_id=request.request_id,
            num_tokens=num_tokens_need_slot,
            new_computed_blocks=new_computed_block_list,
            num_encoder_tokens=num_encoder_tokens,
            total_computed_tokens=num_local_computed_tokens
            + num_external_computed_tokens,
        )

        if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():          #如果空闲块不足，返回None，触发调度器的抢占或等待逻辑
            # Cannot allocate new blocks
            return None

        if (
            new_computed_block_list is not self.empty_kv_cache_blocks.blocks        #
            or num_external_computed_tokens > 0
        ):
            # Append the new computed blocks to the request blocks until now to     #将命中的已计算的kv blocks先追加到当前request已有的block列表中，以避免在后续新block分配失败时丢失这些已命中的kv
            # avoid the case where the new blocks cannot be allocated.
            self.coordinator.allocate_new_computed_blocks(
                request_id=request.request_id,
                new_computed_blocks=new_computed_block_list,
                num_local_computed_tokens=num_local_computed_tokens,
                num_external_computed_tokens=num_external_computed_tokens,
            )

        new_blocks = self.coordinator.allocate_new_blocks(                          #给这个request申请一批新的kv存储空间
            request.request_id, num_tokens_need_slot, num_encoder_tokens
        )

        # P/D: delay caching blocks if we have to recv from                         缓存提交，如果是PD分离场景且设置了延迟缓存，直接返回分配结果
        # remote. Update state for locally cached blocks.
        if not self.enable_caching or delay_cache_blocks:
            return self.create_kv_cache_blocks(new_blocks)

        # NOTE(woosuk): We want to commit (cache) up to num_local_computed_tokens   
        # + num_external_computed_tokens + num_new_tokens, but must exclude
        # "non-committable" tokens (e.g., draft tokens that could be rejected).
        # Therefore, we cap the number at `request.num_tokens`, ensuring only
        # "finalized" tokens are cached.
        num_tokens_to_cache = min(                                                 #确定哪些token是可以被固化到缓存池中供后续请求共享的
            total_computed_tokens + num_new_tokens,                                #注意：这里会限制在request.num_tokens内，防止把不稳定的投入token存入缓存
            request.num_tokens,
        )
        self.coordinator.cache_blocks(request, num_tokens_to_cache)

        return self.create_kv_cache_blocks(new_blocks)                             #返回本次分配新产生的block信息(供算子使用)

    def free(self, request: Request) -> None:
        """Free the blocks allocated for the request.
        We free the blocks in reverse order so that the tail blocks are evicted
        first when caching is enabled.

        Args:
            request: The request to free the blocks.
        """
        self.coordinator.free(request.request_id)

    def remove_skipped_blocks(
        self, request_id: str, total_computed_tokens: int
    ) -> None:
        """Remove the blocks that are no longer needed from `blocks` and replace
        the removed blocks with null_block.

        Args:
            request_id: The request ID.
            total_computed_tokens: The total number of computed tokens, including
                local computed tokens and external computed tokens.
        """
        self.coordinator.remove_skipped_blocks(request_id, total_computed_tokens)

    def evict_blocks(self, block_ids: set[int]) -> None:
        """evict blocks from the prefix cache by their block IDs.

        Args:
            block_ids: Set of block IDs to evict from cache.
        """
        self.block_pool.evict_blocks(block_ids)

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF                        重置prefix cache.该函数可用于rlhf场景(当模型权重更新后，需要使已有的prefix cache失效)
        flows to invalidate prefix caching after the weights are updated,               或用于基准测试时重置prefix cache状态
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """
        if not self.block_pool.reset_prefix_cache():
            return False
        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.reset = True
        return True

    def get_num_common_prefix_blocks(self, running_request_id: str) -> list[int]:
        """Calculate the number of common prefix blocks for each kv cache group.        计算每个kv cache组中的公共前缀block数量

        The function selects a running request and iterates through its blocks.         这个函数会选取一个正在运行的请求，并遍历它的block
        A block is considered a common prefix block if ALL requests with                如果一个block被认为是 公共前缀block,则必须满足
        allocated KV cache share it (i.e., ref_cnt equals the number of entries         所有已经分配了kv cache的请求都共享这个block
        in req_to_blocks).

        NOTE(woosuk): The number of requests with allocated KV cache is **greater       已经分配了kv cache的请求数量大于或等于当前step被调度直行的请求数量。
        than or equal to** the number of requests scheduled in the current step.        这是因为，拥有kv cache只意味着
        This is because having allocated KV cache only indicates that:              
        1. The request has not yet finished, and                                        1.该请求还没结束  2该请求的block还没被释放
        2. The request holds its blocks unfreed.

        While all scheduled requests must have allocated KV cache, the inverse          虽然所有被调度的请求一定已经分配了kv cache，但反过来不成立：可能存在一些已经分配了kv cache 
        is not necessarily true. There may be requests with allocated KV cache          但当前step没被调度执行的请求
        that are not scheduled in the current step.

        This can result in an edge case where the number of common prefix blocks        这会导致一个边界情况
        is 0, even though all scheduled requests share a common prefix. This            即使所有当前被调度的请求都有一个共同前缀 函数仍然可能返回0个公共前缀
        occurs because there may be unscheduled requests that do not share the          原因是：可能存在一些未被调度的请求，它们不共享这个前缀，从而破坏了所有请求共享的条件
        common prefix. Currently, this case cannot be easily detected, so the           当前视线中，这种情况无法轻易检测，因此函数在这种情况下会返回0
        function returns 0 in such cases.

        Args:
            running_request_id: The request ID of any running request, used to          任意一个正在运行的请求id，用作参考来查找公共前缀
                identify the common prefix blocks.

        Returns:
            list[int]: The number of common prefix blocks for each kv cache             list[int]：每个 KV cache group 的公共前缀 block 数量
            group.
        """
        return self.coordinator.get_num_common_prefix_blocks(running_request_id)

    def take_events(self) -> list[KVCacheEvent]:
        """Take the KV cache events from the block pool.

        Returns:
            A list of KV cache events.
        """
        return self.block_pool.take_events()

    def get_blocks(self, request_id: str) -> KVCacheBlocks:
        """Get the blocks of a request."""
        return self.create_kv_cache_blocks(self.coordinator.get_blocks(request_id))

    def get_block_ids(self, request_id: str) -> tuple[list[int], ...]:
        """Get the block ids of a request."""
        return self.get_blocks(request_id).get_block_ids()

    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        """Cache the blocks for the request, if enabled.                                如果开启了缓存功能，则将该请求的block加入缓存

        Args:
            request: The request to cache the blocks.                                   需缓存block的请求
            num_computed_tokens: The number of computed tokens, including tokens        num_computed_tokens: 已计算的 token 数量
                that are already cached and tokens to be cached.
        """
        if self.enable_caching:
            self.coordinator.cache_blocks(request, num_computed_tokens)

    def create_kv_cache_blocks(
        self, blocks: tuple[list[KVCacheBlock], ...]
    ) -> KVCacheBlocks:
        """
        根据传入的block列表创建一个kvcacheblocks对象
        如果所有block是空的，则返回一个预先定义好的空对象
        """
        
        # Only create new KVCacheBlocks for non-empty blocks
        return KVCacheBlocks(blocks) if any(blocks) else self.empty_kv_cache_blocks
