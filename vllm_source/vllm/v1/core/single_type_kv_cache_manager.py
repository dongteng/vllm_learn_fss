# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Sequence

from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHashList, KVCacheBlock
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    CrossAttentionSpec,
    FullAttentionSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
)
from vllm.v1.request import Request


class SingleTypeKVCacheManager(ABC):
    """
    An abstract base class for a manager that handle the kv cache management
    logic of one specific type of attention layer.
    单类型kv cache管理器的抽象基类(Single Type KV Cache Manager)
    作用：负责管理某一种特定类型的Attention层的KV Cache逻辑
    
    例如：普通自注意力的kv cache; Cross-Attention(编码器-解码器注意力)的 KV Cache
    
    不同Attention类型可能有不同的block分配、缓存、跳过逻辑,所以用这个基类来统一管理
    """

    def __init__(
        self,
        kv_cache_spec: KVCacheSpec,
        block_pool: BlockPool,
        enable_caching: bool,
        kv_cache_group_id: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> None:
        """
        Initializes the SingleTypeKVCacheManager.                            初始化
        Args:
            kv_cache_spec: The kv_cache_spec for this manager.               当前这种 Attention 类型的 KV Cache 配置(block 大小、数据类型等)
            block_pool: The block pool.                                      全局的 block 池子,所有 KV Cache block 都从这里分配和回收
            kv_cache_group_id: The id of the kv cache group of this manager. 当前 KV Cache 组的编号(用于区分不同类型的组)
        """
        self.block_size = kv_cache_spec.block_size                          #每个block能容纳多少token
        self.dcp_world_size = dcp_world_size                                #dp pp相关参数
        self.pcp_world_size = pcp_world_size
        if dcp_world_size * pcp_world_size > 1:                             #如果存在并行(比如tp/context parallel) 一个逻辑block会被拆到多个卡上
            self.block_size *= dcp_world_size * pcp_world_size              #实际block size要扩大
        self.kv_cache_spec = kv_cache_spec
        self.block_pool = block_pool                                        #全局block分配器(类似内存池)
        self.enable_caching = enable_caching                                #

        # Mapping from request ID to blocks to track the blocks allocated   请求到block的映射,结构是{req_id: [block1, block2, ...]}
        # for each request, so that we can free the blocks when the request 记录每个请求占用了哪些block
        # is finished.
        self.req_to_blocks: defaultdict[str, list[KVCacheBlock]] = defaultdict(list)

        # {req_id: The number of cached blocks for this given request}      每个请求有多少block是cache命中的
        # This is used to track the number of cached blocks for each request.
        # This is only used to track the RUNNING requests, we do not track the  这里只记录正在运行中的请求,被抢占的请求不会记录
        # data for preempted ones.
        self.num_cached_block: dict[str, int] = {}

        self.kv_cache_group_id = kv_cache_group_id                           #cache分组id,用来区分不同kv cache空间,举例：group 0 → self-attention cache     group 1 → cross-attention cache 
        self._null_block = block_pool.null_block                             #空block哨兵,

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
    ) -> int:
        """
        Get the number of blocks needed to be allocated for the request.      #获取该请求需要分配的物理块数量,不是在算新分配多少,而是算这次操作

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including 需要槽位的总token数
                tokens that are already allocated).
            new_computed_blocks: The new computed blocks just hitting the      这次新请求刚刚命中prefix cache的那些block
                prefix caching.
            total_computed_tokens: Include both local and external computed    已计算的总token数(包含本地和外部获取的),这个请求历史上已经算过的token总数
                tokens.

        Returns:
            The number of blocks to allocate.                                  需要新分配的块数量
        """

        num_required_blocks = cdiv(num_tokens, self.block_size)                 #计算总共需要的块数：使用向上取整除法(cdiv),即总 tokens / 每个块的大小
        num_req_blocks = len(self.req_to_blocks.get(request_id, ()))            #获取该请求目前已经持有的块数量

        if request_id in self.num_cached_block:                                 #如果请求已经再运行中(存在于num_cached_block中)
            # Fast-path: a running request won't have any new prefix-cache hits.#运行中的请求不会再发生新prefix命中,prefix只会再请求刚进入系统时发生一次
            assert len(new_computed_blocks) == 0
            # NOTE: With speculative decoding, request's blocks may be allocated
            # for draft tokens which are later rejected. In this case,
            # num_required_blocks may be smaller than num_req_blocks.
            return max(num_required_blocks - num_req_blocks, 0)

        num_skipped_tokens = self.get_num_skipped_tokens(total_computed_tokens)  #处理滑动窗口情况：获取被滑动窗口跳过的Token数
        num_local_computed_blocks = len(new_computed_blocks) + num_req_blocks    #计算当前逻辑已有的块：前缀缓存命中的块+该请求已有的块
        # Number of whole blocks that are skipped by the attention window.      
        # If nothing is skipped, this is 0.
        num_skipped_blocks = num_skipped_tokens // self.block_size               #计算因为滑动窗口而被整块跳过的块数
        # We need blocks for the non-skipped suffix. If there are still
        # local-computed blocks inside the window, they contribute to the
        # required capacity; otherwise, skipped blocks dominate.
        num_new_blocks = max(
            num_required_blocks - max(num_skipped_blocks, num_local_computed_blocks),#计算真正需要新分配的物理块数。总需求减去(跳过的块和已有块中的较大者)。如果已有块都在窗口内,则减去已有块；如果跳过的块更多,说明部分已有块已失效,按跳过的算。
            0,
        )

        # Among the `new_computed_blocks`, the first `num_skipped_blocks` worth   处理前缀缓存中需要排除的部分
        # of blocks are skipped; `num_req_blocks` of those may already be in      在new_computed_blocks中,前num_skipped_blocks个块可能被跳过
        # `req_to_blocks`, so only skip the remainder from `new_computed_blocks`. 但其中 num_req_blocks 可能已经处理过,这里计算真正需要从 new_computed_blocks 中剔除的跳过块数。
        num_skipped_new_computed_blocks = max(0, num_skipped_blocks - num_req_blocks)

        # If a computed block is an eviction candidate (in the free queue and     处理可驱逐块的重新激活：
        # ref_cnt == 0), it will be removed from the free queue when touched by   如果一个已计算的块(在Prefix cache中)当前引用计数为0,且在空闲队列中
        # the allocated request, so we must count it in the free-capacity check.  当它被这个请求重新使用时,它会从空闲队列中移除(不再空闲),为了保证内存安全,在检查剩余空间时,必须把这些即将被占用的“伪空闲”块也算作分配需求。
        num_evictable_blocks = sum(
            blk.ref_cnt == 0 and not blk.is_null
            for blk in new_computed_blocks[num_skipped_new_computed_blocks:]
        )
        return num_new_blocks + num_evictable_blocks                              #最终结果 = 纯新分配的块 + 从空闲状态转为占用状态的缓存块

    def allocate_new_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        """
        Add the new computed blocks to the request. This involves three steps:      将新计算得到的 block(包括 prefix cache 命中的 block)加入到当前 request 中
        1. Touch the computed blocks to make sure they won't be evicted.            1.先触碰这些block 确保不会回收
        1.5. (Optional) For sliding window, skip blocks are padded with null blocks.1.5如果使用滑动窗口(sliding window),对于被跳过的位置,用空 block(null block)进行填充。
        2. Add the remaining computed blocks.                                       2将剩余有效的计算 block 加入到 request 中。
        3. (Optional) For KV connectors, allocate new blocks for external computed  3(可选)如果使用 KV connector,对于来自外部的已计算 token,需要额外分配新的 block。
            tokens (if any).                                                          外部已经算过了为什么还要再分配block? 算过 不等于已经有本地kv block,当前这个request的kv cache里还没有位置来存它们

        Args:
            request_id: The request ID.
            new_computed_blocks: The new computed blocks just hitting the           本次请求中新命中prefix cache的block(即可以直接复用的KV cache block)
                prefix cache.
            num_local_computed_tokens: The number of local computed tokens.         本地计算得到的token数量(当前模型实际计算出来的)
            num_external_computed_tokens: The number of external computed tokens.   外部来源的已计算的token数量(例如通过kv connector /检索/远端缓存获得的)
        """

        if request_id in self.num_cached_block:                                     #如果是running阶段的request,那么running阶段不会再发生prefix cache命中,因为prefix cache只在刚进入系统时发生
            # Fast-path: a running request won't have any new prefix-cache hits.
            # It should not have any new computed blocks.
            assert len(new_computed_blocks) == 0
            return                                                                  #直接退出 不需要再分配处理

        # A new request.
        req_blocks = self.req_to_blocks[request_id]                                 #新请求初始化
        assert len(req_blocks) == 0                                                 #新请求还没有任何block
        num_total_computed_tokens = (                                               #当前已经算过的token数 本地+外部
            num_local_computed_tokens + num_external_computed_tokens
        )
        num_skipped_tokens = self.get_num_skipped_tokens(num_total_computed_tokens) #被窗口裁掉的token数(超出attention window)
        num_skipped_blocks = num_skipped_tokens // self.block_size                  #转为block数(整块被裁掉的)
        if num_skipped_blocks > 0:
            # It is possible that all new computed blocks are skipped when          #缓存命中的block也可能被窗口裁掉 
            # num_skipped_blocks > len(new_computed_blocks).                        #cache是按分块的,index索引小的
            new_computed_blocks = new_computed_blocks[num_skipped_blocks:]          
            # Some external computed tokens may be skipped too.                     #external tokens也可能被裁掉 (只保留窗口部分)
            num_external_computed_tokens = min(
                num_total_computed_tokens - num_skipped_tokens,                     #窗口内token数
                num_external_computed_tokens,                                       #原external token数
            )

        # Touch the computed blocks to make sure they won't be evicted.
        if self.enable_caching:
            self.block_pool.touch(new_computed_blocks)                              #标记这些block为则正在使用,否则它们可能在free queue中被回收,本质就是增加引用计数
        else:
            assert not any(new_computed_blocks), (
                "Computed blocks should be empty when prefix caching is disabled"   #如果关闭了prefix cache , 不应该有任何命中
            )

        # Skip blocks are padded with null blocks.                                  #填充被skip的block(null block),举例skip了2个block, [NULL, NULL, real_block1, real_block2]
        req_blocks.extend([self._null_block] * num_skipped_blocks)
        # Add the remaining computed blocks.                                        #加入有效block(prefix命中/新计算)
        req_blocks.extend(new_computed_blocks)
        # All cached hits (including skipped nulls) are already cached; mark        #标记这些block经过处理(包括null padding) 后续cache_blocks()不会重复cache
        # them so cache_blocks() will not try to re-cache blocks that already       
        # have a block_hash set.
        self.num_cached_block[request_id] = len(req_blocks)                         

        if num_external_computed_tokens > 0:                                        #处理external tokens(重点),核心原因：算过不等于已经在当前request的kv cache里；这段代码只是在分配本地位置,并没有把远程kv 数据真正拷出来
            # Allocate new blocks for external computed tokens.                     #external kv 可能在远端 CPU 其他缓存  ,但当前request需要的是 一段连续的、本地的kv block布局,所以必须给这些token分配本地存储位置
            allocated_blocks = self.block_pool.get_new_blocks(
                cdiv(num_total_computed_tokens, self.block_size) - len(req_blocks)
            )
            req_blocks.extend(allocated_blocks)

    def allocate_new_blocks(
        self, request_id: str, num_tokens: int
    ) -> list[KVCacheBlock]:
        """
        Allocate new blocks for the request to give it at least `num_tokens`        为该请求分配新的block,使其至少拥有num_tokens个token的存储槽位
        token slots.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including      需要分配槽位的token总数(包含已经分配过的token)
                tokens that are already allocated).

        Returns:    
            The new allocated blocks.                                               新分配的block列表
        """
        req_blocks = self.req_to_blocks[request_id]                                 #已经拥有的block列表
        num_required_blocks = cdiv(num_tokens, self.block_size)                     #计算总共需要多少block 
        num_new_blocks = num_required_blocks - len(req_blocks)                      #计算还缺多少block
        if num_new_blocks <= 0:
            return []
        else:
            new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
            req_blocks.extend(new_blocks)
            return new_blocks

    def cache_blocks(self, request: Request, num_tokens: int) -> None:
        """
        Cache the blocks for the request.                                           #为该request的block进行缓存(加入prefix cache)

        Args:
            request: The request.
            num_tokens: The total number of tokens that need to be cached           当前请求
                (including tokens that are already cached).                         需要缓存的token总数(包含已经缓存的)
        """
        num_cached_blocks = self.num_cached_block.get(request.request_id, 0)        #当前已经cache了多少block,表示当前这个request已经有多少block被放进prefix cache
        num_full_blocks = num_tokens // self.block_size                             #当前可形成多少完整block

        if num_cached_blocks >= num_full_blocks:                                    #如果cache够了 直接返回 
            return

        self.block_pool.cache_full_blocks(                                          #新完成的完整block加入prefix cache
            request=request,
            blocks=self.req_to_blocks[request.request_id],                          #这个request当前的所有block
            num_cached_blocks=num_cached_blocks,                                    #从第几个block开始cache(避免重复)
            num_full_blocks=num_full_blocks,                                        #cache到第几个block为止
            block_size=self.block_size,                                             #
            kv_cache_group_id=self.kv_cache_group_id,                               #用于区分不同kv cache类型
        )

        self.num_cached_block[request.request_id] = num_full_blocks

    def free(self, request_id: str) -> None:
        """
        Free the blocks for the request.                                            #释放这个request占用的所有kv cache block (回收到Block pool)

        Args:
            request_id: The request ID.
        """
        # Default to [] in case a request is freed (aborted) before alloc.          #取出该request对应的block列表
        req_blocks = self.req_to_blocks.pop(request_id, [])

        # Free blocks in reverse order so that the tail blocks are                  #逆序后[tail_block, ..., prefix_block]
        # freed first.                                                              #目的是优先释放尾部block(最新的最不可能被复用的)
        ordered_blocks = reversed(req_blocks)

        self.block_pool.free_blocks(ordered_blocks)                                 #归还block到block_pool
        self.num_cached_block.pop(request_id, None)                                 #

    @abstractmethod
    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        """
        Get the number of common prefix blocks for all requests with allocated     #获取当前所已经分配了kv cache的请求之间的公共前缀block数量
        KV cache.

        Args:
            running_request_id: The request ID.

        Returns:
            The number of common prefix blocks for all requests with allocated
            KV cache.
        """

        raise NotImplementedError

    @classmethod
    @abstractmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        """
        Get the longest cache hit prefix of the blocks that is not longer than     获取一段最长的cache命中前缀,其长度不超过max_length
        `max_length`. The prefix should be a common prefix hit for all the         这个前缀必须在 kv_cache_group_ids指定的所有kv cache组中都是共同命中的前缀
        kv cache groups in `kv_cache_group_ids`. If no cache hit is found,         如果没有任何cache命中,则返回空列表
        return an empty list.
        If eagle is enabled, drop the last matched block to force recompute the    如果启用了eagle,则会丢弃最后一个已经匹配的block,以强制重新计算最后一个Block,
        last block to get the required hidden states for eagle drafting head.      从而获取eagle drafting head所需的hidden states
        Need to be customized for each attention type.

        Args:
            block_hashes: The block hashes of the request.                         当前请求对应的block 哈希列表
            max_length: The maximum length of the cache hit prefix.                允许返回的cache命中前缀的最大长度
            kv_cache_group_ids: The ids of the kv cache groups.                    kv cache组的id列表(可能有多组,比如不同attention类型)
            block_pool: The block pool.
            kv_cache_spec: The kv cache spec.                                      kv cache配置
            use_eagle: Whether to use eagle.                                       是否启用eagle
            alignment_tokens: The returned cache hit length (in tokens) should     返回命中的长度(以token计)必须是该值的整数倍
                be a multiple of this value (in tokens). By default, it should
                be set to the block_size.
            dcp_world_size: The world size of decode context parallelism.          decode context parallel的并行规模
            pcp_world_size: The world size of prefill context parallelism.         prefill context parallel的并行规模

        Returns:
            A list of cached blocks with skipped blocks replaced by null block     返回一个列表,表示每个kv cache组对应的命中block列表
            for each kv cache group in `kv_cache_group_ids`.                       返回列表长度=len(kv_cache_group_ids)
            Return a list of length `len(kv_cache_group_ids)`, where the i-th      第i个元素：第i个kv cache group的命中block列表
            element is a list of cached blocks for the i-th kv cache group
            in `kv_cache_group_ids`.
            For example, sliding window manager should return a list like
            ([NULL, NULL, KVCacheBlock(7), KVCacheBlock(8)]) for block size 4
            and sliding window 8 and len(kv_cache_group_ids) = 1.
        """

        raise NotImplementedError

    def remove_skipped_blocks(
        self, request_id: str, total_computed_tokens: int
    ) -> None:
        """
        Remove and free the blocks that are no longer needed for attention computation.  移除并释放注意力计算中不需要的block
        The removed blocks should be replaced by null_block.                             被移除的block会被替换为null_block

        This function depends on `get_num_skipped_tokens`, which need to be implemented  这函数依赖get_num_skipped_tokens
        differently for each attention type.                                             这个函数需要根据不同的attention类型分别实现

        Args:
            request_id: The request ID.                                                 请求ID
            total_computed_tokens: The total number of computed tokens, including       已经计算过的token总数
                local computed tokens and external computed tokens.
        """
        # Remove the blocks that will be skipped during attention computation.          有多少token已经滑出窗口
        num_skipped_tokens = self.get_num_skipped_tokens(total_computed_tokens)
        if num_skipped_tokens <= 0:                                                     #如果没有token被裁掉
            # This indicates that ALL tokens are inside attention window.               #全部token都还在attention window内,不需要释放任何block
            # Thus we do not need to free any blocks outside attention window.
            # A typical case is full attention that we never free any token
            # before the request is finished.
            return
        blocks = self.req_to_blocks[request_id]                                         #当前request的block列表
        num_skipped_blocks = num_skipped_tokens // self.block_size
        # `num_skipped_tokens` may include tokens that haven't been allocated yet
        # (e.g., when the attention window moves into the external computed tokens
        # range), so we must cap to the number of blocks that currently exist for
        # this request.
        num_skipped_blocks = min(num_skipped_blocks, len(blocks))
        removed_blocks: list[KVCacheBlock] = []
        # Because the block starts from index 0, the num_skipped_block-th block
        # corresponds to index num_skipped_blocks - 1.
        for i in range(num_skipped_blocks - 1, -1, -1):
            if blocks[i] == self._null_block:
                # If the block is already a null block, the blocks before it
                # should also have been set to null blocks by the previous calls
                # to this function.
                break
            removed_blocks.append(blocks[i])
            blocks[i] = self._null_block
        self.block_pool.free_blocks(removed_blocks)

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        """
        Get the number of tokens that will be skipped for attention computation.

        Args:
            num_computed_tokens: The number of tokens that have been computed.

        Returns:
            The number of tokens that will be skipped for attention computation.
        """
        # The default behavior is to not skip any tokens.
        return 0


class FullAttentionManager(SingleTypeKVCacheManager):
    """
    管理full attention 和chunked local attention的kv cache
    """
    
    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        """
        查找请求与已缓存kv cache中最长的公共前缀块,该方法通过block hash链,尝试从block pool中查找已缓存的块,返回每个kv cache group对应的最长命中块列表
        参数:
            block_hashes: 当前请求生成的 block hash 序列(从前到后)
            max_length: 当前请求允许查找的最大长度(token 数)
            kv_cache_group_ids: 需要查找的 KV Cache Group ID 列表
            block_pool: 全局 KV Cache 块池
            kv_cache_spec: 当前 attention 类型的配置规格
            use_eagle: 是否启用 Eagle 解码(需要特殊处理最后一块)
            alignment_tokens: token 对齐要求(某些硬件/优化需要对齐)
            dcp_world_size: 数据并行世界大小(用于调整 block_size)
            pcp_world_size: 流水并行世界大小
        返回:
            tuple[list[KVCacheBlock], ...]: 对每个 kv_cache_group_id 返回对应的已命中块列表
        example:  block_size = 16 tokens  当前请求 tokens = 50 → 分块：Block0(0-15), Block1(16-31), Block2(32-47), Block3(48-49),对应 block_hashes = [H0, H1, H2, H3]. cache中已有H0,H1存在, H2不存在
        执行逻辑：H0,H1命中,  
        """
        
        assert isinstance(
            kv_cache_spec, FullAttentionSpec | ChunkedLocalAttentionSpec
        ), (
            "FullAttentionManager can only be used for full attention "
            "and chunked local attention groups"
        )
        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(                    #为每个kv_cache_group初始化一个命中block列表
            [] for _ in range(len(kv_cache_group_ids))
        )
        block_size = kv_cache_spec.block_size
        if dcp_world_size * pcp_world_size > 1:                                     #如果开启并行,逻辑block会被放大
            block_size *= dcp_world_size * pcp_world_size
        max_num_blocks = max_length // block_size                                   #最多能匹配多少个block(受max_length限制)
        for block_hash in itertools.islice(block_hashes, max_num_blocks):           #核心逻辑,从前往后匹配block hash(寻找最长连续前缀)
            # block_hashes is a chain of block hashes. If a block hash is not       #block hashes是链式结构(前缀依赖)   一旦某个Block不存在,后面的肯定也没算出来,直接break
            # in the cached_block_hash_to_id, the following block hashes are
            # not computed yet for sure.
            if cached_block := block_pool.get_cached_block(                         
                block_hash, kv_cache_group_ids
            ):
                for computed, cached in zip(computed_blocks, cached_block):         #对每个kv_cache_group,把命中的block加进去
                    computed.append(cached)
            else:
                break
        if use_eagle and computed_blocks[0]:
            # Need to drop the last matched block if eagle is enabled.
            for computed in computed_blocks:
                computed.pop()
        while (
            block_size != alignment_tokens  # Faster for common case.               #对齐修正：要求命中的tokens数必须能被alignment_tokens整除,不满足就从尾部不断剪裁block
            and len(computed_blocks[0]) * block_size % alignment_tokens != 0
        ):
            for computed in computed_blocks:
                computed.pop()
        return computed_blocks

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        """
        获取当前正在运行的请求与系统中所有其他请求共享的公共前缀块数量。

        用于判断该请求的 prefix 有多大程度被其他请求复用,常用于调度决策。
        """
        blocks = self.req_to_blocks[running_request_id]
        num_common_blocks = 0
        for block in blocks:
            if block.ref_cnt == len(self.req_to_blocks):
                num_common_blocks += 1
            else:
                break
        return num_common_blocks


class SlidingWindowManager(SingleTypeKVCacheManager):
    def __init__(
        self, kv_cache_spec: SlidingWindowSpec, block_pool: BlockPool, **kwargs
    ) -> None:
        super().__init__(kv_cache_spec, block_pool, **kwargs)
        self.sliding_window = kv_cache_spec.sliding_window
        self._null_block = block_pool.null_block

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        assert isinstance(kv_cache_spec, SlidingWindowSpec), (
            "SlidingWindowManager can only be used for sliding window groups"
        )
        assert dcp_world_size == 1, "DCP not support sliding window attn now."
        assert pcp_world_size == 1, "PCP not support sliding window attn now."

        # The number of contiguous blocks needed for prefix cache hit.
        # -1 since the input token itself is also included in the window
        sliding_window_contiguous_blocks = cdiv(
            kv_cache_spec.sliding_window - 1, kv_cache_spec.block_size
        )
        if use_eagle:
            # Need to drop the last matched block if eagle is enabled. For
            # sliding window layer, we achieve this by increasing the number of
            # contiguous blocks needed for prefix cache hit by one and dropping
            # the last matched block.
            sliding_window_contiguous_blocks += 1

        # TODO: reduce i by sliding_window_contiguous_blocks when cache miss, to
        # optimize the time complexity from O(max_num_blocks) to
        # O(max_num_blocks / sliding_window_contiguous_blocks +
        # sliding_window_contiguous_blocks),
        # which is good for low cache hit rate scenarios.
        max_num_blocks = max_length // kv_cache_spec.block_size
        computed_blocks = tuple(
            [block_pool.null_block] * max_num_blocks
            for _ in range(len(kv_cache_group_ids))
        )
        block_size = kv_cache_spec.block_size
        num_contiguous_blocks = 0
        match_found = False
        # Search from right to left and early stop when a match is found.
        for i in range(max_num_blocks - 1, -1, -1):
            if cached_block := block_pool.get_cached_block(
                block_hashes[i], kv_cache_group_ids
            ):
                # Skip prefix matching check if the block is not aligned with
                # `alignment_tokens`.
                if (
                    num_contiguous_blocks == 0
                    and block_size != alignment_tokens  # Faster for common case.
                    and (i + 1) * block_size % alignment_tokens != 0
                ):
                    continue
                # Add the cached block to the computed blocks.
                for computed, cached in zip(computed_blocks, cached_block):
                    computed[i] = cached
                num_contiguous_blocks += 1
                if num_contiguous_blocks >= sliding_window_contiguous_blocks:
                    # Trim the trailing blocks.
                    # E.g., [NULL, NULL, 8, 3, NULL, 9] -> [NULL, NULL, 8, 3]
                    # when sliding_window_contiguous_blocks=2.
                    for computed in computed_blocks:
                        del computed[i + num_contiguous_blocks :]
                    match_found = True
                    break
            else:
                num_contiguous_blocks = 0
        if not match_found:
            # The first `num_contiguous_blocks` is a cache hit even if
            # `num_contiguous_blocks < sliding_window_contiguous_blocks`.
            for computed in computed_blocks:
                del computed[num_contiguous_blocks:]
            while (
                block_size != alignment_tokens  # Faster for common case.
                and len(computed_blocks[0]) * block_size % alignment_tokens != 0
            ):
                for computed in computed_blocks:
                    computed.pop()
        if use_eagle and computed_blocks[0]:
            assert kv_cache_spec.block_size == alignment_tokens, (
                "aligned_length is not compatible with eagle now"
            )
            for computed in computed_blocks:
                computed.pop()
        return computed_blocks

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        """
        Get the number of tokens that will be skipped for attention computation.

        For sliding window, this corresponds to the tokens that are prior to
        the current sliding window.

        Example:
        sliding_window=4, num_computed_tokens=7

        Tokens:   [ 0  1  2  3  4  5  6  7 ]
                  | ---- computed -----|
                                         ^ next token to be computed
                               |-----------| sliding window for next token
                  |--skipped---|

        The current window contains tokens 4~7. Tokens 0~3 will be skipped for
        attention computation since they are outside the sliding window.
        Thus, get_num_skipped_tokens(7) == 4.

        Args:
            num_computed_tokens: The number of tokens that have been computed.

        Returns:
            The number of tokens that will be skipped for attention computation.
        """
        return max(0, num_computed_tokens - self.sliding_window + 1)

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        """
        NOTE(Chen): The prefix blocks are null blocks for sliding window layers.
        So it's not correct to count ref_cnt like FullAttentionManager. Return
        0 here for correctness. Need to support cascade attention + sliding
        window in the future.
        """
        return 0


class ChunkedLocalAttentionManager(SingleTypeKVCacheManager):
    def __init__(
        self, kv_cache_spec: ChunkedLocalAttentionSpec, block_pool: BlockPool, **kwargs
    ) -> None:
        super().__init__(kv_cache_spec, block_pool, **kwargs)
        self.attention_chunk_size = kv_cache_spec.attention_chunk_size
        self._null_block = block_pool.null_block

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        """
        For chunked local attention, we need to find the longest cache hit
        prefix of the blocks that is not longer than `max_length`. The prefix
        should be a common prefix hit for all the kv cache groups in
        `kv_cache_group_ids`. If no cache hit is found, return an empty list.
        note we mark as computed if the whole block is outside of the local
        window, and set the block as null. Examples:

        1. Attention chunk size of 8, block size of 4, max length of 15
        for next token at 15th (zero-indexed), 8th - 14th tokens are in
        the window(needs lookup), 0th - 7th are not in the window,
        so they are already marked as computed. We check the complete
        block3 (8th - 11th tokens), Assume block 3 is hit, we will return
        [null, null, block 3], otherwise, we return [null, null]

        2. Attention chunk size of 8, block size of 4, max length of 16
        for next token at 16th (zero-indexed), 0th - 15th tokens are not
        in the window, so they are already marked as computed.
        we return 4 blocks[null, null, null, null]

        Args:
            block_hashes: The block hashes of the request.
            max_length: The maximum length of the cache hit prefix.
            kv_cache_group_ids: The ids of the kv cache groups.
            block_pool: The block pool.
            kv_cache_spec: The kv cache spec.
            use_eagle: Whether to use eagle.
            dcp_world_size: The world size of decode context parallelism.
            pcp_world_size: The world size of prefill context parallelism.
            alignment_tokens: The returned cache hit length (in tokens) should
                be a multiple of this value (in tokens).

        Returns:
            A list of cached blocks
        """
        assert isinstance(kv_cache_spec, ChunkedLocalAttentionSpec), (
            "ChunkedLocalAttentionManager can only be used for "
            + "chunked local attention groups"
        )
        assert use_eagle is False, (
            "Hybrid KV cache is not supported for " + "eagle + chunked local attention."
        )
        assert dcp_world_size == 1, "DCP not support chunked local attn now."
        assert pcp_world_size == 1, "PCP not support chunked local attn now."
        assert kv_cache_spec.block_size == alignment_tokens, (
            "KV cache groups with different block sizes are not compatible with "
            "chunked local attention now"
        )
        max_num_blocks = max_length // kv_cache_spec.block_size
        if max_length > 0:
            local_attention_start_idx = (
                max_length
                // kv_cache_spec.attention_chunk_size
                * kv_cache_spec.attention_chunk_size
            )
        else:
            local_attention_start_idx = 0
        # we marked blocks out of window as computed
        # with null blocks, and blocks inside window based on cache lookup
        # result [null] [null] ... [null] [hit block 1 (1st block contain
        # last window)] [hit block 2] ... [hit block x]
        local_attention_start_block_idx = (
            local_attention_start_idx // kv_cache_spec.block_size
        )
        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [block_pool.null_block] * local_attention_start_block_idx
            for _ in range(len(kv_cache_group_ids))
        )
        for i in range(local_attention_start_block_idx, max_num_blocks):
            block_hash = block_hashes[i]
            if cached_block := block_pool.get_cached_block(
                block_hash, kv_cache_group_ids
            ):
                for computed, cached in zip(computed_blocks, cached_block):
                    computed.append(cached)
            else:
                break
        return computed_blocks

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        """
        Get the number of tokens that will be skipped for attention computation.

        For chunked local attention, this corresponds to the tokens that are on
        the left side of the current chunk.

        Example 1:
        chunk size = 8, num_computed_tokens = 13
        Tokens:  [ 0 1 2 3 4 5 6 7 | 8 9 10 11 12 13 14 15 ] ...
                 | ----- computed ---------------|
                                                  ^^ next token to be computed
                                   |----------------| <-- attention window for
                                                          next token
                 |--- skipped -----|
        Output: get_num_skipped_tokens(13) == 8

        Example 2:
        chunk size = 8, num_computed_tokens = 8
        Tokens:  [ 0 1 2 3 4 5 6 7 | 8 9 10 11 12 13 14 15 ] ...
                 | --- computed ---|
                                     ^ next token to be computed
                                   |--| <-- attention window for next token
                 | --- skipped ----|
        Output: get_num_skipped_tokens(8) == 8

        Example 3:
        chunk size = 8, num_computed_tokens = 7
        Tokens:  [ 0 1 2 3 4 5 6 7 | 8 9 10 11 12 13 14 15 ] ...
                 |---computed---|
                                 ^ next token to be computed
                 |-----------------| <-- attention window for next token
                 no token should be skipped.
        Output: get_num_skipped_tokens(7) == 0

        Args:
            num_computed_tokens: The number of tokens that have been computed.

        Returns:
            The number of tokens that will be skipped for attention computation.
        """
        num_skipped_tokens = (
            num_computed_tokens // self.attention_chunk_size
        ) * self.attention_chunk_size
        return num_skipped_tokens

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        """
        cascade attention is not supported by chunked local attention.
        """
        return 0


class MambaManager(SingleTypeKVCacheManager):
    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        assert isinstance(kv_cache_spec, MambaSpec), (
            "MambaManager can only be used for mamba groups"
        )
        assert dcp_world_size == 1, "DCP not support mamba now."
        assert pcp_world_size == 1, "PCP not support mamba now."
        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [] for _ in range(len(kv_cache_group_ids))
        )

        block_size = kv_cache_spec.block_size
        max_num_blocks = max_length // block_size
        # Search from right to left and early stop when a match is found.
        for i in range(max_num_blocks - 1, -1, -1):
            if cached_block := block_pool.get_cached_block(
                block_hashes[i], kv_cache_group_ids
            ):
                # When enable Mamba prefix caching, `block_size` will be aligned
                # across full attention layers and Mamba layers to ensure the
                # prefix hit length aligned at block
                if (
                    block_size != alignment_tokens  # Faster for common case.
                    and (i + 1) * block_size % alignment_tokens != 0
                ):
                    continue
                for computed, cached in zip(computed_blocks, cached_block):
                    # the hit length logic later assumes:
                    #  hit_length = len(hit_blocks_other_attn[0])
                    #               * self.other_block_size
                    # so we insert dummy blocks at the beginning:
                    computed.extend([block_pool.null_block] * i)
                    computed.append(cached)
                break  # we just need the last match - early stopping

        return computed_blocks

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        """
        cascade attention is not supported by mamba
        """
        return 0

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
    ) -> int:
        # Allocate extra `num_speculative_blocks` blocks for
        # speculative decoding (MTP/EAGLE) with linear attention.
        assert isinstance(self.kv_cache_spec, MambaSpec)
        if self.kv_cache_spec.num_speculative_blocks > 0:
            num_tokens += (
                self.kv_cache_spec.block_size
                * self.kv_cache_spec.num_speculative_blocks
            )
        return super().get_num_blocks_to_allocate(
            request_id, num_tokens, new_computed_blocks, total_computed_tokens
        )

    def allocate_new_blocks(
        self, request_id: str, num_tokens: int
    ) -> list[KVCacheBlock]:
        # Allocate extra `num_speculative_blocks` blocks for
        # speculative decoding (MTP/EAGLE) with linear attention.
        assert isinstance(self.kv_cache_spec, MambaSpec)
        if self.kv_cache_spec.num_speculative_blocks > 0:
            num_tokens += (
                self.kv_cache_spec.block_size
                * self.kv_cache_spec.num_speculative_blocks
            )
        return super().allocate_new_blocks(request_id, num_tokens)

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        """
        Get the number of tokens whose mamba state are not needed anymore. Mamba only
        need to keep the state of the last computed token, so we return
        num_computed_tokens - 1.
        """
        return num_computed_tokens - 1


class CrossAttentionManager(SingleTypeKVCacheManager):
    """Manager for cross-attention KV cache in encoder-decoder models."""

    def allocate_new_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        # We do not cache blocks for cross-attention to be shared between
        # requests, so  `new_computed_blocks` should always be empty.
        assert len(new_computed_blocks) == 0

    def cache_blocks(self, request: Request, num_tokens: int) -> None:
        # We do not cache blocks for cross-attention to be shared between
        # requests, so this method is not relevant.
        raise ValueError("Should not be called as prefix caching is disabled.")

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        # Cross-attention blocks contain request-specific encoder states
        # and are not shared between different requests
        return 0

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        assert isinstance(kv_cache_spec, CrossAttentionSpec), (
            "CrossAttentionManager can only be used for cross-attention groups"
        )
        # Cross-attention does not benefit from prefix caching since:
        # 1. Encoder states are unique per request (different audio/image
        #    inputs)
        # 2. Encoder states are computed once per request, not incrementally
        # 3. No reusable prefix exists between different multimodal inputs
        # Return empty blocks to indicate no cache hits
        raise NotImplementedError("CrossAttentionManager does not support caching")


spec_manager_map: dict[type[KVCacheSpec], type[SingleTypeKVCacheManager]] = {
    FullAttentionSpec: FullAttentionManager,
    MLAAttentionSpec: FullAttentionManager,
    SlidingWindowSpec: SlidingWindowManager,
    ChunkedLocalAttentionSpec: ChunkedLocalAttentionManager,
    MambaSpec: MambaManager,
    CrossAttentionSpec: CrossAttentionManager,
}


def get_manager_for_kv_cache_spec(
    kv_cache_spec: KVCacheSpec, **kwargs
) -> SingleTypeKVCacheManager:
    manager_class = spec_manager_map[type(kv_cache_spec)]
    manager = manager_class(kv_cache_spec, **kwargs)
    return manager
