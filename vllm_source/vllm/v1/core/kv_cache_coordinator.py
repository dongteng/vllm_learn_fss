# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import Sequence
from math import lcm

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashList,
    BlockHashListWithBlockSize,
    KVCacheBlock,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    CrossAttentionManager,
    FullAttentionManager,
    get_manager_for_kv_cache_spec,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
)
from vllm.v1.request import Request


class KVCacheCoordinator(ABC):
    """
    Coordinate the KV cache of different KV cache groups.
    KV CACHE协调器的抽象基类
    
    它是vllm前缀缓存系统的核心管理组件,负责：
     -管理不同类型的kv cache(普通的Attention cross-attention等)
     -协调kv cache block的分配 复用 释放
     -支持前缀缓存的查找和命中
     -处理多组kv cache的情况(如不同精度、不同层)
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        use_eagle: bool,
        enable_caching: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        hash_block_size: int,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ):
        self.kv_cache_config = kv_cache_config
        self.max_model_len = max_model_len
        self.enable_caching = enable_caching
        # BlockPool 是所有 KV Cache block 的统一池子,负责 block 的分配和回收
        self.block_pool = BlockPool(                
            kv_cache_config.num_blocks,
            enable_caching,
            hash_block_size,
            enable_kv_cache_events,
            metrics_collector,
        )

        # Needs special handling for find_longest_cache_hit if eagle is enabled . 是否使用 Eagle(推测解码的一种实现),对 find_longest_cache_hit 有特殊处理
        self.use_eagle = use_eagle
        
        #为魅族kv cache创建对应的manager(单类型管理器),例如普通attention的manager,cross-attention的manager等
        self.single_type_managers = tuple(
            get_manager_for_kv_cache_spec(
                kv_cache_spec=kv_cache_group.kv_cache_spec,
                block_pool=self.block_pool,
                enable_caching=enable_caching,
                kv_cache_group_id=i,
                dcp_world_size=dcp_world_size,
                pcp_world_size=pcp_world_size,
            )
            for i, kv_cache_group in enumerate(self.kv_cache_config.kv_cache_groups)
        )

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        num_encoder_tokens: int,
        total_computed_tokens: int,
    ) -> int:
        """
        Get the number of blocks needed to be allocated for the request.      计算该请求还需要额外分配多少个kv cache block   
                                                                              会分别询问每个single_type_manager,然后把它们需要的block数量累加返回
        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including 需要分配 slot 的 token 总数(包括已经分配过的 token)
                tokens that are already allocated).
            new_computed_blocks: The new computed blocks just hitting the      刚刚通过 prefix caching 命中的、新计算得到的 block
                prefix caching.
            num_encoder_tokens: The number of encoder tokens for allocating    encoder 侧的 token 数量(用于为 cross-attention 分配 block)
                blocks for cross-attention.
            total_computed_tokens: Include both local and external tokens.     已经计算过的 token 总数(包含本地计算 + 外部缓存命中)

        Returns:
            The number of blocks to allocate.
        """
        num_blocks_to_allocate = 0
        for i, manager in enumerate(self.single_type_managers):
            if isinstance(manager, CrossAttentionManager):
                # For cross-attention, we issue a single static allocation
                # of blocks based on the number of encoder input tokens.
                num_blocks_to_allocate += manager.get_num_blocks_to_allocate(  # Cross-Attention(编码器-解码器注意力)需要特殊处理# 根据 encoder 输入 token 数量进行一次性静态分配
                    request_id, num_encoder_tokens, [], 0
                )
            else:
                num_blocks_to_allocate += manager.get_num_blocks_to_allocate(   # 普通 Attention 的 block 分配
                    request_id,
                    num_tokens,
                    new_computed_blocks[i],
                    total_computed_tokens,
                )
        return num_blocks_to_allocate

    def allocate_new_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        """
        Add the new computed blocks to the request. Optionally allocate new       将prefix cache命中的物理块挂载到该请求,并为外部的数据预留空间
            blocks for external computed tokens (if any).                         是将block从逻辑上的命中转为物理上的持有

        Args:
            request_id: The request ID.
            new_computed_blocks: The new computed blocks just hitting the           
                prefix cache.
            num_local_computed_tokens: The number of local computed tokens.         
            num_external_computed_tokens: The number of external computed tokens. 外部 KV Connector (如远程 LMCache) 命中的 Token 总数
        """
        for i, manager in enumerate(self.single_type_managers):
            manager.allocate_new_computed_blocks(
                request_id,
                new_computed_blocks[i],
                num_local_computed_tokens,
                num_external_computed_tokens,
            )

    def allocate_new_blocks(
        self,
        request_id: str,
        num_tokens: int,
        num_encoder_tokens: int = 0,
    ) -> tuple[list[KVCacheBlock], ...]:
        """
        Allocate new blocks for the request to give it at least `num_tokens`   为请求分配新的 KV Cache block,以容纳至少 num_tokens 个 token。
        token slots.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including
                tokens that are already allocated).
            num_encoder_tokens: The number of encoder tokens for allocating
                blocks for cross-attention.

        Returns:
            The new allocated blocks.
        """
        return tuple(
            manager.allocate_new_blocks(
                request_id,
                num_encoder_tokens
                if isinstance(manager, CrossAttentionManager)
                else num_tokens,
            )
            for manager in self.single_type_managers
        )

    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        """
        Cache the blocks for the request.     将请求的 block 标记为已缓存(用于后续的前缀缓存命中)

        Args:
            request: The request.
            num_computed_tokens: The total number of tokens
                that need to be cached
                (including tokens that are already cached).
        """
        for manager in self.single_type_managers:
            manager.cache_blocks(request, num_computed_tokens)

    def free(self, request_id: str) -> None:
        """
        Free the blocks for the request.  释放该请求占用的所有 KV Cache block

        Args:
            request_id: The request ID.
        """
        for manager in self.single_type_managers:
            manager.free(request_id)

    def get_num_common_prefix_blocks(self, running_request_id: str) -> list[int]:
        """
        Get the number of common prefix blocks for all requests with allocated    获取每个 KV Cache 组的共同前缀 block 数量(用于调度决策)
        KV cache for each kv cache group.

        Args:
            running_request_id: The request ID of any running request, used to
                identify the common prefix blocks.

        Returns:
            list[int]: The number of common prefix blocks for each kv cache group.
        """
        return [
            manager.get_num_common_prefix_blocks(running_request_id)
            for manager in self.single_type_managers
        ]

    def remove_skipped_blocks(
        self, request_id: str, total_computed_tokens: int
    ) -> None:
        """
        Remove the blocks that are no longer needed from `blocks` and replace    移除请求中不再需要的 block(例如因为 chunked prefill 或跳过部分 token)
        the removed blocks with null_block.                                      并用 null_block 替换

        Args:
            request_id: The request ID.
            total_computed_tokens: The total number of computed tokens, including
                local computed tokens and external computed tokens.
        """
        for manager in self.single_type_managers:
            manager.remove_skipped_blocks(request_id, total_computed_tokens)

    def get_blocks(self, request_id: str) -> tuple[list[KVCacheBlock], ...]:
        """
        Get the blocks for the request. 获取该请求当前持有的所有 KV Cache block。
        """
        return tuple(
            manager.req_to_blocks.get(request_id) or []
            for manager in self.single_type_managers
        )

    @abstractmethod
    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        """
        抽象方法：查找最长的前缀缓存命中。
        不同子类会根据是否启用前缀缓存实现不同的逻辑。
        """
        pass


class KVCacheCoordinatorNoPrefixCache(KVCacheCoordinator):
    """
    KV cache coordinator to use if prefix caching is disabled or unsupported.
    In contrast to UnitaryKVCacheCoordinator and HybridKVCacheCoordinator,
    supports arbitrary numbers of KV cache groups (including 0 groups).
    Does not implement any features related to prefix caching.
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        use_eagle: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        hash_block_size: int,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ):
        """
        当禁用或不支持前缀缓存使用的KV Cache协调器。与 UnitaryKVCacheCoordinator 和 HybridKVCacheCoordinator 不同,
        该类支持任意数量的KV Cache组(包括0组)。不实现与前缀缓存相关的任何功能。
        
        """
        
        super().__init__(
            kv_cache_config,
            max_model_len,
            use_eagle,
            False,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
        )
        self.num_single_type_manager = len(self.single_type_managers)

    def get_num_common_prefix_blocks(self, running_request_id: str) -> list[int]:
        return [0] * self.num_single_type_manager

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [] for _ in range(self.num_single_type_manager)
        )
        return blocks, 0


class UnitaryKVCacheCoordinator(KVCacheCoordinator):
    """
    KV cache coordinator for models with only one KV cache group. This is the           专门处理只有一种kv cache类型的模型
    case for models with only one KV cache type, e.g., all attention layers use         所有层都是full attention(标准transformer),或所有层都是sliding window attention
    full attention or all attention layers use sliding window attention.                整个模型只有一种kv cache结构,所以调度逻辑可以简化
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,                                                 #
        max_model_len: int,                                                             #
        use_eagle: bool,
        enable_caching: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        hash_block_size: int,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ):
        super().__init__(
            kv_cache_config,
            max_model_len,
            use_eagle,
            enable_caching,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
        )
        self.kv_cache_spec = self.kv_cache_config.kv_cache_groups[0].kv_cache_spec      #前模型的 KV cache 描述(shape、block_size 等)
        self.block_size = self.kv_cache_spec.block_size                                 # vLLM 的基本粒度：block(通常是16 tokens)
        self.dcp_world_size = dcp_world_size                                            #并行信息(Distributed Context Parallel / Pipeline Context Parallel)
        self.pcp_world_size = pcp_world_size
        if dcp_world_size > 1:                                                          #block_size 在并行时会“放大”
            self.block_size *= dcp_world_size
        if pcp_world_size > 1:
            self.block_size *= pcp_world_size
        # For models using only Mamba, block_size is set to max_model_len when          关键约束：hash_block_size 必须等于 block_size,
        # prefix caching is disabled, and hash_block_size validation is skipped.
        assert not enable_caching or (hash_block_size == self.block_size), (
            "UnitaryKVCacheCoordinator assumes hash_block_size == block_size"
        )
        assert len(self.kv_cache_config.kv_cache_groups) == 1, (
            "UnitaryKVCacheCoordinator assumes only one kv cache group"
        )

    def find_longest_cache_hit(                                                         #找到最长可复用kv cache前缀,输入：block_hashes:每个block(16 tokens)的hash,  max_cache_hit_length:最多允许命中的token数
        self,   
        block_hashes: list[BlockHash],                                                  # 每个 block(例如16个token,默认就是16)对应的 hash 列表
        max_cache_hit_length: int,                                                      ## 最多允许命中的 token 数(限制前缀长度)
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        hit_blocks = self.single_type_managers[0].find_longest_cache_hit(               #调用底层manager去找最长命中的cache block前缀,底层manager可能为FullAttentionManager
            block_hashes=block_hashes,
            max_length=max_cache_hit_length,
            kv_cache_group_ids=[0],                                                     # 指定使用哪个 cache group(这里只用第0组)
            block_pool=self.block_pool,
            kv_cache_spec=self.kv_cache_spec,
            use_eagle=self.use_eagle,
            alignment_tokens=self.block_size,
            dcp_world_size=self.dcp_world_size,
            pcp_world_size=self.pcp_world_size,
        )
        return hit_blocks, len(hit_blocks[0]) * self.block_size


class HybridKVCacheCoordinator(KVCacheCoordinator):
    """
    KV cache coordinator for hybrid models with multiple KV cache types, and
    thus multiple kv cache groups.
    To simplify `find_longest_cache_hit`, it only supports the combination of
    two types of KV cache groups, and one of them must be full attention.
    May extend to more general cases in the future.
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        use_eagle: bool,
        enable_caching: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        hash_block_size: int,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ):
        super().__init__(
            kv_cache_config,
            max_model_len,
            use_eagle,
            enable_caching,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
        )
        # hash_block_size: the block size used to compute block hashes.
        # The actual block size usually equals hash_block_size, but in cases where
        # different KV cache groups have different block sizes, the actual block size
        # can be a multiple of hash_block_size.
        self.hash_block_size = hash_block_size
        assert all(
            g.kv_cache_spec.block_size % hash_block_size == 0
            for g in kv_cache_config.kv_cache_groups
        ), "block_size must be divisible by hash_block_size"
        assert dcp_world_size == 1, "DCP not support hybrid attn now."
        assert pcp_world_size == 1, "PCP not support hybrid attn now."
        self.verify_and_split_kv_cache_groups()

    def verify_and_split_kv_cache_groups(self) -> None:
        """
        Verifies that the model has exactly two types of KV cache groups, and
        one of them is full attention. Then, split the kv cache groups into full
        attention groups and other groups.
        """
        full_attention_spec: FullAttentionSpec | None = None
        other_spec: KVCacheSpec | None = None
        self.full_attention_group_ids: list[int] = []
        self.other_group_ids: list[int] = []
        for i, g in enumerate(self.kv_cache_config.kv_cache_groups):
            if isinstance(g.kv_cache_spec, FullAttentionSpec):
                if full_attention_spec is None:
                    full_attention_spec = g.kv_cache_spec
                else:
                    assert full_attention_spec == g.kv_cache_spec, (
                        "HybridKVCacheCoordinator assumes exactly one type of "
                        "full attention groups now."
                    )
                self.full_attention_group_ids.append(i)
            else:
                if other_spec is None:
                    other_spec = g.kv_cache_spec
                else:
                    assert other_spec == g.kv_cache_spec, (
                        "HybridKVCacheCoordinator assumes "
                        "exactly one other type of groups now."
                    )
                self.other_group_ids.append(i)

        assert full_attention_spec is not None, (
            "HybridKVCacheCoordinator assumes exactly one type of full "
            "attention groups now."
        )
        assert other_spec is not None, (
            "HybridKVCacheCoordinator assumes exactly one type of other groups now."
        )

        self.full_attention_manager_cls = FullAttentionManager
        self.other_attention_cls = self.single_type_managers[
            self.other_group_ids[0]
        ].__class__
        self.full_attention_spec = full_attention_spec
        self.other_spec = other_spec
        self.full_attention_block_size = self.full_attention_spec.block_size
        self.other_block_size = self.other_spec.block_size
        # The LCM of the block sizes of full attention and other attention.
        # The cache hit length must be a multiple of the LCM of the block sizes
        # to make sure the cache hit length is a multiple of the block size of
        # each attention type. Requiring this because we don't support partial
        # block cache hit yet.
        self.lcm_block_size = lcm(self.full_attention_block_size, self.other_block_size)

        if max(self.full_attention_group_ids) < min(self.other_group_ids):
            self.full_attn_first = True
        elif max(self.other_group_ids) < min(self.full_attention_group_ids):
            self.full_attn_first = False
        else:
            raise ValueError(
                "HybridKVCacheCoordinator assumes the full "
                "attention group ids and other attention group ids "
                "do not interleave, either full attention group ids "
                "are before other attention group ids or vice versa."
                "This is for simplifying merging hit_blocks_full_attn and "
                "hit_blocks_other_attn to hit_blocks."
            )

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        """
        Find the longest cache hit for the request.

        Args:
            block_hashes: The block hashes of the request.
            max_cache_hit_length: The maximum length of the cache hit.

        Returns:
            A tuple containing:
                - A list of the cache hit blocks for each single type manager.
                - The number of tokens of the longest cache hit.
        """
        # First, find the longest cache hit for full attention.
        if self.full_attention_spec.block_size == self.hash_block_size:
            # Common case.
            full_attention_block_hashes: BlockHashList = block_hashes
        else:
            # block_size is a multiple of hash_block_size. This happens when different
            # KV cache groups have different block sizes. In this case, we need to
            # recalculate block_hashes at the granularity of block_size, using the
            # original block_hashes (at the granularity of hash_block_size).
            full_attention_block_hashes = BlockHashListWithBlockSize(
                block_hashes, self.hash_block_size, self.full_attention_spec.block_size
            )
        hit_blocks_full_attn = self.full_attention_manager_cls.find_longest_cache_hit(
            block_hashes=full_attention_block_hashes,
            max_length=max_cache_hit_length,
            kv_cache_group_ids=self.full_attention_group_ids,
            block_pool=self.block_pool,
            kv_cache_spec=self.full_attention_spec,
            use_eagle=self.use_eagle,
            alignment_tokens=self.lcm_block_size,
        )
        hit_length = len(hit_blocks_full_attn[0]) * self.full_attention_block_size

        # Next, find the cache hit for the other attention WITHIN
        # the cache hit of full attention.
        if self.other_spec.block_size == self.hash_block_size:
            # Common case.
            other_block_hashes: BlockHashList = block_hashes
        else:
            # Similar to the full attention case, here we need to recalculate
            # block_hashes at the granularity of block_size, using the original
            # block_hashes (at the granularity of hash_block_size).
            other_block_hashes = BlockHashListWithBlockSize(
                block_hashes, self.hash_block_size, self.other_spec.block_size
            )
        hit_blocks_other_attn = self.other_attention_cls.find_longest_cache_hit(
            block_hashes=other_block_hashes,
            max_length=hit_length,
            kv_cache_group_ids=self.other_group_ids,
            block_pool=self.block_pool,
            kv_cache_spec=self.other_spec,
            use_eagle=self.use_eagle,
            alignment_tokens=self.lcm_block_size,
        )
        hit_length = len(hit_blocks_other_attn[0]) * self.other_block_size

        # NOTE: the prefix cache hit length must be a multiple of block_size as
        # we don't support partial block cache hit yet. The cache hit length
        # of other attention is ensured to be a multiple of the block size of
        # full attention layers in current implementation, because hit_length is
        # a multiple of other attention's block size, and other attention's
        # block size is a multiple of full attention's block size (verified in
        # `verify_and_split_kv_cache_groups`).
        assert hit_length % self.full_attention_block_size == 0

        # Truncate the full attention cache hit to the length of the
        # cache hit of the other attention.
        for group_hit_blocks in hit_blocks_full_attn:
            del group_hit_blocks[hit_length // self.full_attention_block_size :]

        # Merge the hit blocks of full attention and other attention.
        if self.full_attn_first:
            hit_blocks = hit_blocks_full_attn + hit_blocks_other_attn
        else:
            hit_blocks = hit_blocks_other_attn + hit_blocks_full_attn
        return hit_blocks, hit_length


def get_kv_cache_coordinator(
    kv_cache_config: KVCacheConfig,
    max_model_len: int,
    use_eagle: bool,
    enable_caching: bool,
    enable_kv_cache_events: bool,
    dcp_world_size: int,
    pcp_world_size: int,
    hash_block_size: int,
    metrics_collector: KVCacheMetricsCollector | None = None,
) -> KVCacheCoordinator:
    """
    根据配置创建合适的KVCacheCoordinator(KV Cache协调器)
    KVCacheCoordinator是vLLM 前缀缓存(prefix caching)系统的核心管理组件,负责管理所有KV Cache的存储、查找、命中、分配和回收等操作
    这个函数是一个工厂函数,会根据不同情况返回不同类型的Coordinator实现。
    """
    # ====================== 情况 1：未启用前缀缓存 ======================
    # 如果用户关闭了 prefix caching(--disable-prefix-caching),
    # 则使用最简单的 NoPrefixCache 版本,不进行任何前缀缓存查找和复用
    if not enable_caching:
        return KVCacheCoordinatorNoPrefixCache(
            kv_cache_config,
            max_model_len,
            use_eagle,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
        )
    # ====================== 情况 2：普通单组 KV Cache(最常见情况) ======================
    # 当 kv_cache_config 中只有一个 kv_cache_group 时(通常是普通模型),
    # 使用 UnitaryKVCacheCoordinator(单一协调器)
    if len(kv_cache_config.kv_cache_groups) == 1:
        return UnitaryKVCacheCoordinator(
            kv_cache_config,
            max_model_len,
            use_eagle,
            enable_caching,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
        )
    # ====================== 情况 3：混合 KV Cache(高级场景) ======================
    # 当存在多个 kv_cache_group 时(比如同时支持不同精度、不同层、Encoder-Decoder 模型等),
    # 使用 HybridKVCacheCoordinator 来处理更复杂的多组 KV Cache 管理
    return HybridKVCacheCoordinator(
        kv_cache_config,
        max_model_len,
        use_eagle,
        enable_caching,
        enable_kv_cache_events,
        dcp_world_size=dcp_world_size,
        pcp_world_size=pcp_world_size,
        hash_block_size=hash_block_size,
        metrics_collector=metrics_collector,
    )
