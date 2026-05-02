# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable, Sequence
from typing import Any

from vllm.distributed.kv_events import (
    MEDIUM_GPU,
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVCacheEvent,
)
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashList,
    BlockHashListWithBlockSize,
    BlockHashWithGroupId,
    ExternalBlockHash,
    FreeKVCacheBlockQueue,
    KVCacheBlock,
    get_block_hash,
    make_block_hash_with_group_id,
    maybe_convert_block_hash,
)
from vllm.v1.request import Request

logger = init_logger(__name__)


class BlockHashToBlockMap:
    """
    Cache of blocks that are used for prefix caching. It caches blocks     用于prefix cache的核心缓存映射表
    from hash directly to a block or multiple blocks                       它的核心结构是{block_hash: KVCacheBlock 或 多个 KVCacheBlock}
    (i.e. {block_hash: KVCacheBlocks})                                     大多数情况下,一个hash只对应一个 KVCacheBlock
    - Mostly block_hash maps to a single KVCacheBlock, and KVCacheBlocks   少数情况下,一个hash对应多个block(用dict存)
        would simply be a KVCacheBlock.
    - Otherwise, KVCacheBlocks is a dict from {block_id: KVCacheBlock}

    A cached block is a full block with a block hash that can be used     一个cached block是一个完整block,可用于prefix cache复用
    for prefix caching.
    The cached block may be used by running requests or in the
    free_block_queue that could potentially be evicted.

    NOTE #1: We currently don't de-duplicate the blocks in the cache,     1.不做去重(即不检查是否有完全相同的block) 原因:block_id需要保持稳定(append-only)
    meaning that if a block becomes full and is cached, we don't check
    if there is already an identical block in the cache. This is because
    we want to make sure the allocated block IDs won't change so that
    block tables are append-only.
    NOTE #2: The union type is introduced in order to reduce GC costs     2.使用union(单个or dict)是为了减少gc开销
    from the inner dict.
    """

    def __init__(self):
        self._cache: dict[
            BlockHashWithGroupId, KVCacheBlock | dict[int, KVCacheBlock]  #key是BlockHashWithGroupId,而value有2种可能,KVCacheBlock 或者 一个 {block_id: KVCacheBlock} 的字典
        ] = {}

    def get_one_block(self, key: BlockHashWithGroupId) -> KVCacheBlock | None:
        """
        Gets any block with the given block hash key.                     #根据key获取任意一个block
        """
        blocks = self._cache.get(key)
        if blocks is not None:
            if isinstance(blocks, KVCacheBlock):                          #情况1,只有1个block
                return blocks
            if isinstance(blocks, dict):                                  #情况2, 有多个block
                return next(iter(blocks.values()))                        #返回一个 随便一个
            self._unexpected_blocks_type(blocks)                          #防御性检查 类型不对直接报错
        return None

    def insert(self, key: BlockHashWithGroupId, block: KVCacheBlock) -> None:
        """
        Inserts the KVCacheBlock to the cache 插入一个KVCacheBlock到cache
        """
        blocks = self._cache.get(key)                                     #先看这个key是否存在
        if blocks is None:
            # When key is not found, attach a single block to the key     #已经有一个block,升级为dict 边
            self._cache[key] = block
        elif isinstance(blocks, KVCacheBlock):
            # If there's a block with the same key, merge the original block  #已经有一个block,升级为dict 变多个block
            # and the new block into a dict
            self._cache[key] = {blocks.block_id: blocks, block.block_id: block}
        elif isinstance(blocks, dict):
            # If it's already a dict, simply insert the block             #如果已经是多个block直接加入
            blocks[block.block_id] = block
        else:
            self._unexpected_blocks_type(blocks)

    def pop(self, key: BlockHashWithGroupId, block_id: int) -> KVCacheBlock | None:
        """
        Checks if block_hash exists and pop block_id from the cache        #从 cache 中删除指定 block_id,并返回该 block
        """
        blocks = self._cache.pop(key, None)
        if blocks is None:
            # block_hash not found in the cache
            return None
        # TODO(Jialin): If key is found, block_id should always present
        # in blocks. We currently keep the original behaviour for safety.
        #
        # Will add block_id == blocks.block_id assertion and
        # use del blocks[block_id] instead as followup.
        if isinstance(blocks, KVCacheBlock):
            if blocks.block_id == block_id:
                return blocks
            # If the single block ID doesn't match, we should put the
            # block back (it should happen rarely)
            self._cache[key] = blocks
            return None
        if isinstance(blocks, dict):
            # Try to pop block_id from the block dict, and if dict still
            # contain blocks, put back to the cache.
            block = blocks.pop(block_id, None)
            if len(blocks) > 0:
                self._cache[key] = blocks
            return block
        self._unexpected_blocks_type(blocks)
        return None

    def __len__(self) -> int:
        return len(self._cache)

    def _unexpected_blocks_type(self, blocks: Any) -> None:
        raise AssertionError(f"Invalid KV cache block type {type(blocks)}")


class BlockPool:
    """BlockPool that manages KVCacheBlocks.
    It provides methods to allocate, free and cache the kv cache blocks. The     简单理解:
    free_block_queue stores the free blocks in eviction order to enable             -所有kv cache使用的内存块 都由它统一管理
    allocation, free, and cache eviction. The cached_block_hash_to_block            -负责 block 的分配(allocate)、释放(free)、缓存(cache)和驱逐(evict)。
    maps between block hash and cached block to support finding cached blocks
    by their block hash.

    Args:
        num_gpu_blocks: The number of blocks in the pool.
        enable_caching: Whether to enable prefix caching.
        hash_block_size: The block size of which the block hashes are computed.
            The actual block size usually equals hash_block_size, but in cases
            where different KV cache groups have different block sizes, the
            actual block size can be a multiple of hash_block_size.
        enable_kv_cache_events: Whether to enable kv cache events.
        metrics_collector: Optional metrics collector for tracking block residency.
    """

    def __init__(
        self,
        num_gpu_blocks: int,                                                    #GPU上总共可以使用的block水浪
        enable_caching: bool,
        hash_block_size: int,                                                   #计算 block hash 时使用的 block 大小
        enable_kv_cache_events: bool = False,                                   #是否开启kv cache事件记录
        metrics_collector: KVCacheMetricsCollector | None = None,               #可选的指标收集器,用于统计block使用情况
    ):
        assert isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0
        self.num_gpu_blocks = num_gpu_blocks
        self.enable_caching = enable_caching
        self.hash_block_size = hash_block_size
        # All kv-cache blocks.                                                   #创建所有kv cache block
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_gpu_blocks)
        ]
        # Free block queue that constructs and manipulates a doubly linked
        # list of free blocks (including eviction candidates when caching is
        # enabled).
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)               #空闲队列,它维护一个双向链表,用于快速分配和按照驱逐优先级管理空闲 block

        # Cache for block lookup                                                 #缓存映射,通过block_hash 快速找到对应的 cached block,# 这是前缀缓存(Prefix Caching)的核心数据结构
        self.cached_block_hash_to_block: BlockHashToBlockMap = BlockHashToBlockMap() 

        # To represent a placeholder block with block_id=0.                       # null_block:一个特殊的占位 block(block_id = 0)
        # The ref_cnt of null_block is not maintained, needs special care to      #用于滑动窗口等场景中跳过某些token时间占位,注意:它的 ref_cnt 不参与正常计数,需要特殊处理
        # avoid freeing it.
        self.null_block = self.free_block_queue.popleft()                                       
        self.null_block.is_null = True

        self.enable_kv_cache_events = enable_kv_cache_events                      ## 用于记录 KV Cache 事件
        self.kv_event_queue: list[KVCacheEvent] = []

        self.metrics_collector = metrics_collector

    def get_cached_block(
        self, block_hash: BlockHash, kv_cache_group_ids: list[int]
    ) -> list[KVCacheBlock] | None:
        """Get the cached block by the block hash for each group in                #根据block hash和kv cache group ids查找是否命中缓存
        `kv_cache_group_ids`, or None if cache miss for any group.
        If there are duplicated blocks, we return the first block in the cache.    如果所有组都命中,则返回对应的 block 列表；只要有一个组没命中,就返回 None。

        Args:
            block_hash: The hash value of the block.
            kv_cache_group_ids: The ids of the KV cache groups.

        Returns:
            The cached blocks if exists, or None.
        """
        cached_blocks = []
        for group_id in kv_cache_group_ids:
            block_hash_with_group_id = make_block_hash_with_group_id(               # 为每个 group 生成带 group_id 的 hash(不同 group 可能有不同配置)
                block_hash, group_id
            )
            block = self.cached_block_hash_to_block.get_one_block(
                block_hash_with_group_id
            )
            if not block:
                return None                                                          # 只要有一个 group 没有命中,就返回 None
            cached_blocks.append(block)
        return cached_blocks

    def cache_full_blocks(
        self,
        request: Request,
        blocks: list[KVCacheBlock],
        num_cached_blocks: int,
        num_full_blocks: int,
        block_size: int,
        kv_cache_group_id: int,
    ) -> None:
        """
        可以想象:每个 block 就像一个装满 token 的箱子,而这个函数的任务就是——把新装满的箱子贴上标签(hash),然后放进仓库(cache)。
        Cache a list of full blocks for prefix caching.                         缓存一组完整的block,用于前缀缓存
        This function takes a list of blocks that will have their block hash    这个函数接收一组block,这些block的block hash元数据会被更新并缓存
        metadata to be updated and cached. Given a request, it updates the      对于一个请求,它会更新每个block的元数据,并把它们缓存到 `cached_block_hash_to_block` 中。block hash 的值是在 Request 对象被创建时以及当新的 token 被追加时计算的。
        metadata for each block and caching it in the
        `cached_block_hash_to_block`.
        The block hashes values are computed by the Request object immediately  block hash 的值是在 Request 对象被创建时以及当新的 token 被追加时计算的。
        when it is created and when new tokens are appended.

        Args:
            request: The request to cache the blocks.                            要进行缓存的请求
            blocks: All blocks in the request.                                   该请求包含的所有block
            num_cached_blocks: The number of blocks that are already cached.     已经缓存过的block数量
            num_full_blocks: The number of blocks that are full and should       已经填满的block数量(这些block在本函数执行后应该被缓存)
                be cached after this function.
            block_size: Number of tokens in each block.                          每个block里包含的token数量
            kv_cache_group_id: The id of the KV cache group.
        
        Example:
            假设 request 的 token 为:[t1, t2, t3, t4,  t5, t6, t7, t8,  t9, t10]  block_size = 4,则会切分为:
        block0 = [t1 t2 t3 t4]   (full)
        block1 = [t5 t6 t7 t8]   (full)
        block2 = [t9 t10]        (not full)
        blocks = [block0, block1, block2]
        假设:num_cached_blocks = 1   # block0 已经缓存
             num_full_blocks   = 2   # block0, block1 是满块
        那么本函数只会处理:new_full_blocks = blocks[1:2] = [block1]
        request.block_hashes:h0 -> block0;h1 -> block1
        本次会:
        1. 取出 block1 对应的 hash = h1
        2. 结合 kv_cache_group_id 生成唯一 key: (h1, group_id)
        3. 写入缓存:cached_block_hash_to_block[(h1, group_id)] = block1
        总结:该函数只缓存“新变满但尚未缓存”的 block(这里是 block1)
        """
        if num_cached_blocks >= num_full_blocks:                              #如果full block都已经缓存过了,那啥也不用干了,直接退出。假设num_full_blocks   = 2   # block0, block1  ,num_cached_blocks = 2   # 两个都缓存了 
            return                                                  
        new_full_blocks = blocks[num_cached_blocks:num_full_blocks]           #找出新满的块,只处理变满的块
        assert len(request.block_hashes) >= num_full_blocks                   ## 断言:请求中提供的 block_hashes 数量必须足够(至少等于需要处理的 full blocks 数量);hash 的生成是上游责任, 这函数只负责用已有hash做缓存映射
        if block_size == self.hash_block_size:                                  
            # Common case.
            block_hashes: BlockHashList = request.block_hashes
        else:
            # 当 block_size 与 hash_block_size 不一致时(通常 block_size 更大),
            # 需要将原始按小粒度计算的 block_hashes 重新聚合为大粒度的 block hash。
            # 即:多个小 block 的 hash → 合成为一个大 block 的 hash,
            # 以保证缓存 key 与当前 block 切分方式一致。
            # block_size is a multiple of hash_block_size. This happens when  
            # different KV cache groups have different block sizes.
            assert block_size % self.hash_block_size == 0
            # Recalculate block_hashes at the granularity of block_size, using
            # the original block_hashes (at the granularity of hash_block_size).
            block_hashes = BlockHashListWithBlockSize(                         #使用包装类,重新按block_size的粒度重新计算block_hashes
                request.block_hashes, self.hash_block_size, block_size
            )

        new_block_hashes = block_hashes[num_cached_blocks:]
        new_hashes: list[ExternalBlockHash] | None = (                         ## 如果启用了 KV Cache 事件记录,则准备一个列表来收集新块的 hash；否则设为 None
            [] if self.enable_kv_cache_events else None         
        )
        for i, blk in enumerate(new_full_blocks):
            # Some blocks may be null blocks when enabling sparse attention like 某些块可能是空块,例如开启了slifing window attention等稀疏注意力时
            # sliding window attention. We skip null blocks here.
            if blk.is_null:                                                     #跳过空块,不进行缓存
                continue
            assert blk.block_hash is None                                       ## 断言:该块之前应该还没有被设置过 block_hash
            block_hash = new_block_hashes[i]                                    #获取当前块对应的hash值

            # ====================== 核心缓存逻辑 ======================
            # Update and added the full block to the cache.
            block_hash_with_group_id = make_block_hash_with_group_id(           ##为block_hash 加上 group_id(因为不同 KV Cache Group可能有相同hash),
                block_hash, kv_cache_group_id
            )
            blk.block_hash = block_hash_with_group_id                           ## 把计算好的 hash 存到 block 对象上
            self.cached_block_hash_to_block.insert(block_hash_with_group_id, blk)
            if new_hashes is not None:                                          #如果需要记录事件,则把 hash(可能经过转换)加入 new_hashes 列表
                new_hashes.append(maybe_convert_block_hash(block_hash))
        
        # ====================== KV Cache 事件处理 ======================
        # 只有启用了 kv_cache_events 时,才会往事件队列中加入 BlockStored 事件
        if self.enable_kv_cache_events:
            if num_cached_blocks == 0:
                parent_block_hash: ExternalBlockHash | None = None
            else:
                parent_block_hash = maybe_convert_block_hash(
                    block_hashes[num_cached_blocks - 1]
                )

            self.kv_event_queue.append(
                BlockStored(
                    block_hashes=new_hashes,
                    parent_block_hash=parent_block_hash,
                    token_ids=request.all_token_ids[
                        num_cached_blocks * block_size : num_full_blocks * block_size
                    ],
                    block_size=block_size,
                    lora_id=request.lora_request.adapter_id
                    if request.lora_request
                    else None,
                    medium=MEDIUM_GPU,
                )
            )

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        """Get new blocks from the free block pool.             
        Note that we do not check block cache in this function.         只从空闲块池中直接分配新块,不会检查block cache(已经缓存的块)
        Args:
            num_blocks: The number of blocks to allocate.               需要分配的block数量
        Returns:
            A list of new block.                                        返回分配到的KVCacheBlock列表
        """
        if num_blocks > self.get_num_free_blocks():                     #先检查空闲块是否足够
            raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")

        ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks) #从空闲块队列(free_block_queue)头部一次性取出num_blocks个空闲块

        # In order to only iterate the list once, we duplicated code a bit
        if self.enable_caching:                                         #根据是否启用前缀缓存,执行不同的逻辑
            for block in ret:                                           #开启前缀缓存的情况
                self._maybe_evict_cached_block(block)                   #如果这个block之前因为缓存而被引用,这里需要先尝试它从缓存中移除
                assert block.ref_cnt == 0                               #断言,此时该block的引用计数必须为0(不应该被任何请求持有)
                block.ref_cnt += 1
                if self.metrics_collector:                              #如果开启了指标收集器,记录block被分配的事件(用于监控)
                    self.metrics_collector.on_block_allocated(block)
        else:
            for block in ret:                                           #未启用缓存的简单情况
                assert block.ref_cnt == 0                               #没有缓存逻辑,直接检查引用计数
                block.ref_cnt += 1
                if self.metrics_collector:                              #记录分配指标
                    self.metrics_collector.on_block_allocated(block)
        return ret                                                      #返回分配好的block列表

    def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
        """
        If a block is cached in `cached_block_hash_to_block`, we reset its hash  如果一个block当前正被缓存在cached_block_hash_to_block中,
        metadata and evict it from the cache.                                    则清楚它的hash数据,并把它从缓存中移除(evict)
        Args:
            block: The block to evict.                                           需要尝试清理的block对象
        Returns:
            True if the block is evicted, False otherwise.                       如果成功从缓存中清楚则返回True,否则返回False
        """
        # Clean up metrics tracking first to prevent leaks                      # ====================== 第一步:清理监控指标 ====================== 
        if self.metrics_collector:                                              #先清理指标收集器,防止内存泄漏(metrics里可能持有对block的引用)
            self.metrics_collector.on_block_evicted(block)
                                                                                # ====================== 第二步:检查是否需要 eviction ======================
        block_hash = block.block_hash                                           ## 获取当前 block 上保存的 hash(带 group_id 的版本)                                        
        if block_hash is None:                                                  #如果这个block没设置过hash,说明它之前就没被缓存过
            # The block doesn't have hash, eviction is not needed
            return False                                                        #无需驱逐,直接返回False
                                                                                # ====================== 第三步:从缓存字典中移除 ======================
        if self.cached_block_hash_to_block.pop(block_hash, block.block_id) is None: #尝试从 cached_block_hash_to_block 这个字典中删除该 block_hash
            # block not found in cached_block_hash_to_block,                    #如果pop返回None,说明这个hash之前根本不存在字典里
            # eviction is not needed
            return False

        block.reset_hash()                                                      #重置block的hash元数据,成功从缓存中移除后,需要把block上的hash信息清除,从而这个block就变成干净的状态,可以被重新分配给其他请求

        if self.enable_kv_cache_events:                                         # ====================== 第五步:如果启用事件,记录 BlockRemoved 事件 ======================
            # FIXME (Chen): Not sure whether we should return `hash_value`      当前的 FIXME:不确定这里应该传纯 hash 还是 (hash, group_id)
            # or `(hash_value, group_id)` here. But it's fine now because       但目前没问题,因为开启 kv cache event 时,混合 KV Cache Manager 被禁用了,只有一个 group。
            # we disable hybrid kv cache manager when kv cache event is
            # enabled, so there is only one group.
            self.kv_event_queue.append(
                BlockRemoved(
                    block_hashes=[maybe_convert_block_hash(get_block_hash(block_hash))],
                    medium=MEDIUM_GPU,
                )
            )
        return True

    def touch(self, blocks: Sequence[KVCacheBlock]) -> None:
        """
        某些block又用到了,把它们从待回收状态拉回正在使用状态
        Touch a block increases its reference count by 1, and may remove         触摸/命中一个block,会把它的引用计数+1
        the block from the free queue. This is used when a block is hit by       并且如果它之前在空闲队列中,会把它从空闲队列中移除
        another request with the same prefix.
                                                                                 这个函数主要用于:当一个新请求命中了之前的prefix(缓存命中)时
        Args:                                                                    需要把这些已经分配的Block 重新标记为正在使用
            blocks: A list of blocks to touch.                                   # 需要 touch(刷新使用状态)的 block 列表
        """
        for block in blocks:
            # ref_cnt=0 means this block is in the free list (i.e. eviction      如果ref_count==0说明这个block当前在free_block_queue(属于可被回收的候选块)
            # candidate), so remove it.
            if block.ref_cnt == 0 and not block.is_null:
                self.free_block_queue.remove(block)                              ## 把这个 block 从空闲队列中移除,防止它被后续的 eviction(回收)操作拿走
            block.ref_cnt += 1                                                   #增加引用计数
            if self.metrics_collector:                                           #如果开启了指标收集器,记录这次block被访问的事件
                self.metrics_collector.on_block_accessed(block)

    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        """Free a list of blocks. The blocks should be ordered by their          释放(释放引用)。传入的blocks必须按照被回收优先级从高到低(即最先应该被evict的block排在最前面)
        eviction priority, where the first block will be evicted first.

        Args:
            ordered_blocks: A list of blocks to free ordered by their eviction   需要释放的block列表,已按
                priority.
        """
        # Materialize the iterable to allow multiple passes.
        # ====================== 第一步:物化 iterable ======================
        # 因为后面需要对同一个列表进行两次遍历(一次减引用,一次筛选),
        # 而 Iterable 可能只能遍历一次,所以先转为 list 进行物化(materialize)
        # Materialize the iterable to allow multiple passes.
        blocks_list = list(ordered_blocks)
        
        # ====================== 第二步:减少所有 block 的引用计数 ======================
        # 对每个 block 的引用计数 ref_cnt 减 1,表示释放一次引用
        for block in blocks_list:
            block.ref_cnt -= 1
            
        # ====================== 第三步:把真正可以回收的 block 加入空闲队列 ======================
        # 筛选出那些 ref_cnt 已经降为 0 且不是 null block 的块,
        # 把它们一次性批量加入到 free_block_queue 中,供后续分配使用
        self.free_block_queue.append_n(
            [block for block in blocks_list if block.ref_cnt == 0 and not block.is_null]
        )

    def evict_blocks(self, block_ids: set[int]) -> None:
        """evict blocks from the prefix cache by their block IDs.           

        only evicts blocks that are currently cached (have a hash). blocks     只从前缀缓存中驱逐指定的blocks,只会驱逐当前整备缓存的block(即拥有hash的block)
        with ref_cnt > 0 are not freed from the block pool, only evicted       注意:即使被驱逐,ref_cnt>0的 block 也不会从block pool中真正释放,
        from the prefix cache hash table.                                      只会从prefix cache的哈希表中移除

        Args:
            block_ids: Set of block IDs to evict from cache.                    需要从前缀缓存中驱逐的block ID集合
        """
        for block_id in block_ids:
            assert block_id < len(self.blocks), (
                f"Invalid block_id {block_id} >= {len(self.blocks)}. "
                f"This indicates a bug in the KV connector - workers should "
                f"only report block IDs that were allocated by the scheduler."
            )
            # 根据 block_id 从 blocks 数组中取出对应的 block 对象
            block = self.blocks[block_id]
            
            # 调用之前注释过的函数,尝试从 prefix cache 中移除该 block
            #(如果该 block 当前有 hash 且在缓存中,才会真正执行 eviction)
            self._maybe_evict_cached_block(block)

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF                    重置前缀缓存 该函数主要用于以下场景:
        flows to invalid prefix caching after the weights are updated,              1.在强化学习流程中,当模型权重更新后,需要之前的prefix caching失效
        or used for resetting prefix caching status for benchmarking.               2.在性能基准测试benchmark中,重置prefix caching状态以确保公平对比

        Returns:
            bool: True if the prefix cache is successfully reset,                   如果前缀缓存成功重置则返回True,否则返回False
            False otherwise.
        """
        #计算当前已被占用的GPU块数量 self.num_gpu_blocks 包含所有可用的块,get_num_free_blocks() 返回空闲块数
        num_used_blocks = self.num_gpu_blocks - self.get_num_free_blocks()
        
        if num_used_blocks != 1:  # The null block is always marked as used         如果占用块数量不等于1,说明还有其他kv cache块正在被用
                                                                                    #系统始终会保留一个null block作为占位,所以1个块是正常的重置条件
            logger.warning(
                "Failed to reset prefix cache because some "
                "blocks (%d) are not freed yet",
                num_used_blocks - 1,
            )
            return False

        # ==================== 执行重置操作 ====================
        # Remove all hashes so that no new blocks will hit. 清空哈希到块的映射表,防止后续有任何请求命中旧的prefix cache
        self.cached_block_hash_to_block = BlockHashToBlockMap()

        # Remove all hashes from all blocks.                                          遍历所有快,清除每个块中保存的哈希值
        for block in self.blocks:                                                     #这样可以确保所有块都不会再被识别为已缓存的前缀块
            block.reset_hash()

        if self.metrics_collector:                                                    #如果启用了指标收集器,则重置相关统计指标
            self.metrics_collector.reset()

        logger.info("Successfully reset prefix cache")

        if self.enable_kv_cache_events:                                                #如果启用了kv cache事件记录,则向事件队列中添加“所有块已清空”事件
            self.kv_event_queue.append(AllBlocksCleared())

        return True

    def get_num_free_blocks(self) -> int:
        """Get the number of free blocks in the pool.
        获取当前kv cache块池中空闲块的数量。
        该方法用于查询还有多少GPU内存块处于空闲状态,可供新的请求分配使用
        常用于:
            1.判断是否可以接受新的请求 2.计算已占用块数量 3.监控缓存使用率

        Returns:
            The number of free blocks. 当前空闲的块数量
        """
        return self.free_block_queue.num_free_blocks

    def get_usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).
        """

        # Subtract 1 to account for null block.
        total_gpu_blocks = self.num_gpu_blocks - 1
        if not total_gpu_blocks:
            return 0
        return 1.0 - (self.get_num_free_blocks() / total_gpu_blocks)

    def take_events(self) -> list[KVCacheEvent]:
        """Atomically takes all events and clears the queue. 原子性取出所有kv cache事件,并清空事件对了
        该方法用于获取上次调用以来产生的所有kv cache相关事件,通常需要记录或监控缓存行为(如块分配、释放、前缀命中等)时使用。

        设计为“一次性取出并清空”,避免事件重复处理或队列无限增长。
        
        注意:
            - 如果未启用 KV Cache 事件功能(enable_kv_cache_events=False),
              则直接返回空列表,不进行任何操作。
            - 操作具有原子性(在单线程或正确加锁的上下文中),确保事件不会丢失或重复。
        Returns:
            A list of KV cache events.
        """
        if not self.enable_kv_cache_events:
            return []
        # 清空事件队列(重新创建一个空列表)
        # 注意:这里采用“替换列表”的方式,而不是调用 clear(),
        # 可能是为了避免在多线程环境下产生问题,或保持代码简单
        events = self.kv_event_queue
        self.kv_event_queue = []
        return events
