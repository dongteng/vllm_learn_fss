# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import torch

from vllm.distributed import get_dcp_group, get_pcp_group
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.utils import CpuGpuBuffer

logger = init_logger(__name__)


class BlockTable: #vLLM管理kv cache的核心数据结构之一，可以把它理解为kv cache的页表
    """
    BlockTable负责记录每个请求当前使用了哪些Block，以及如何映射到逻辑位置
    """
    
    def __init__(
        self,
        block_size: int,
        max_num_reqs: int,
        max_num_blocks_per_req: int,
        max_num_batched_tokens: int,
        pin_memory: bool,
        device: torch.device,
        kernel_block_size: int,
        cp_kv_cache_interleave_size: int,
    ):
        """
        
        Args:
            block_size: Block size used for KV cache memory allocation          内存分配时的块大小,例如16 32
            max_num_reqs: Maximum number of concurrent requests supported.      系统支持的最大并发请求数
            max_num_blocks_per_req: Maximum number of blocks per request.       每个请求最多分配的block数量
            max_num_batched_tokens: Maximum number of tokens in a batch.        单次batch中允许的最大token数量
            pin_memory: Whether to pin memory for faster GPU transfers.         是否用锁页内存 pinnded memory  加速CPU-GPU数据传输
            device: Target device for the block table.                          BlockTable 主要放在哪个设备上（通常是 CPU，因为需要频繁读写）
            kernel_block_size: The block_size of underlying attention kernel.   底层 Attention Kernel（如 FlashAttention、MLA 等）实际使用的 block 大小。可能与 block_size 不一致（比如内存按 32 分配，但 kernel 按 16 计算）
                Will be the same as `block_size` if `block_size` is supported
                by the attention kernel.
            cp_kv_cache_interleave_size                                         context parallel相关的kv cache交错大小
        """
        self.max_num_reqs = max_num_reqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.pin_memory = pin_memory
        self.device = device
        
        # ====================== 处理 block_size 与 kernel_block_size 的差异 ======================
        # vLLM 支持 “混合 Block” 模式：内存分配的 block_size 与 Attention Kernel 使用的 block_size 可以不同
        if kernel_block_size == block_size:
            # Standard case: allocation and computation use same block size      最常见的情况：内存分配和kernel计算使用相同的block大小
            # No block splitting needed, direct mapping                          无需拆分，一一映射，性能最好
            self.block_size = block_size
            self.blocks_per_kv_block = 1
            self.use_hybrid_blocks = False
        else:
            # Hybrid case: allocation block size differs from kernel block size   混合模式，例如内存以32-token为单位分配，但 Attention Kernel 只支持 16-token 的 block
            # Memory blocks are subdivided to match kernel requirements           此时每个内存block需要被拆分成多个kernel block
            # Example: 32-token memory blocks with 16-token kernel blocks
            # → Each memory block corresponds to 2 kernel blocks
            if block_size % kernel_block_size != 0:
                raise ValueError(
                    f"kernel_block_size {kernel_block_size} must divide "
                    f"kv_manager_block_size size {block_size} evenly"
                )

            self.block_size = kernel_block_size                                   #以kernel的block大小为准
            self.blocks_per_kv_block = block_size // kernel_block_size            #一个内存block 包含几个kernel block
            self.use_hybrid_blocks = True

        self.max_num_blocks_per_req = max_num_blocks_per_req * self.blocks_per_kv_block #最终每个请求允许的最大block数量（考虑hybrid拆分）

        # ====================== 核心数据结构 ======================
        # block_table: 形状为 [max_num_reqs, max_num_blocks_per_req] 的 int32 tensor
        # 含义：第 i 个请求使用的第 j 个物理 block 的编号（Block ID）
        self.block_table = self._make_buffer(
            self.max_num_reqs, self.max_num_blocks_per_req, dtype=torch.int32
        )
        #记录每个请求当前实际使用的block数量
        self.num_blocks_per_row = np.zeros(max_num_reqs, dtype=np.int32)            #
        
        #slot_mapping 非常重要的映射表
        #把batch中 每个token的位置映射到具体kv cache的物理slot位置  形状通常为[max_num_batched_tokens]
        self.slot_mapping = self._make_buffer(
            self.max_num_batched_tokens, dtype=torch.int64
        )

        #hybrid block模式下需要用到的辅助数组
        if self.use_hybrid_blocks:
            self._kernel_block_arange = np.arange(0, self.blocks_per_kv_block).reshape(
                1, -1
            )
        else:
            self._kernel_block_arange = None

        # ====================== Context Parallel (CP) / Distributed 相关配置 ======================
        # PCP = Pipeline Context Parallel？ DCP = Data Context Parallel？
        # 用于支持多卡上下文并行时的 KV Cache 管理
        try:
            self.pcp_world_size = get_pcp_group().world_size
            self.pcp_rank = get_pcp_group().rank_in_group
        except AssertionError:
            # PCP might not be initialized in testing 测试环境或未初始化时使用默认值
            self.pcp_world_size = 1
            self.pcp_rank = 0
        try:
            self.dcp_world_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
        except AssertionError:
            # DCP might not be initialized in testing
            self.dcp_world_size = 1
            self.dcp_rank = 0
        self.cp_kv_cache_interleave_size = cp_kv_cache_interleave_size

    def append_row(
        self,
        block_ids: list[int],
        row_idx: int,
    ) -> None:
        """
        为执行请求（row）追加新的KV Cachae Block，实现上下文长度的动态增长。
        这个方法主要在以下场景中被调用：
        - 请求在生成过程中需要更多KV Cache时（即上下文长度增长）
        - 第一次为新请求分配Block后，后续生成token时继续追加
        
        Args:
        -block_ids: 要追加的物理 Block ID 列表（由 KV Cache Manager 分配）
        -row_idx:   该请求在 BlockTable 中的行索引（即请求在 batch 中的位置） 
        """
        if not block_ids:   #如果没有新的block，直接返回
            return

        # ====================== Hybrid Block 处理 ======================
        # 如果当前使用的是混合 Block 模式（内存 block_size 与 kernel block_size 不一致）
        # 需要把内存层分配的 block_ids 转换为 Attention Kernel 实际需要的 kernel block ids
        if self.use_hybrid_blocks:
            block_ids = self.map_to_kernel_blocks(
                np.array(block_ids), self.blocks_per_kv_block, self._kernel_block_arange
            )
        # ====================== 执行追加操作 ======================
        # 计算当前该请求已经使用的 block 数量（即当前行已填充到第几列）
        num_blocks = len(block_ids)
        start = self.num_blocks_per_row[row_idx]
        # 更新该请求使用的 block 总数
        self.num_blocks_per_row[row_idx] += num_blocks
        # 把新的 block_ids 写入 block_table 的对应行
        # block_table 的形状为 [max_num_reqs, max_num_blocks_per_req]
        # 例如：把新 block 写入第 row_idx 行的 [start : start+num_blocks] 位置
        self.block_table.np[row_idx, start : start + num_blocks] = block_ids

    def add_row(self, block_ids: list[int], row_idx: int) -> None:
        """
        为指定请求（row_idx）首次添加kv cache block
        这个方法通常在以下2种场景被调用
         -请求刚完成prefill阶段，准备进入decode阶段（首次分配block）
         -新请求被重新调度或从waiting队列移入running队列时
         
         与append_row()的区别
          -add_row()是初始化操作，先把该请求的block计数清零
          -append_row() 是「追加」操作：用于请求生成过程中继续扩展上下文长度
        
        Args:
            block_ids: KV Cache Manager 新分配给该请求的物理 Block ID 列表
            row_idx:   该请求在 BlockTable 中的行索引（对应 batch 中的位置）
        """
        #先将该请求当前的block数量重置为0  因为这是首次添加，所以必须清空之前可能残留的数据 
        self.num_blocks_per_row[row_idx] = 0
        #调用append_row执行实际的Block添加操作
        #append_row() 会处理 Hybrid Block 转换、写入 block_table 等逻辑
        self.append_row(block_ids, row_idx)

    def move_row(self, src: int, tgt: int) -> None:
        """
        将src行的block信息完整移动到tgt行 ，主要在condense操作中被调用，用于压缩时调整请求的位置
        由于condense会把有效请求向前滑动，原来位置靠后的请求需要被移动到前面较小的索引处，此时就需要该函数来搬运其kv cache的映射关系
        
        注意：
        - 这是一个“移动”操作，而不是复制。移动完成后，源行（src）的 Block 信息会被视为无效。
        - 只移动 Block 映射，不释放或分配新的 Block。
        
        Args:
            src: 源请求在 block_table 中的行索引（要被移动的请求）
            tgt: 目标请求在 block_table 中的行索引（移动后的新位置，通常比 src 小）
        """
        #获取源请求当前使用的block数量  我们只需要移动实际使用的block，而不需要移动整行（因为后面可能有很多空位）
        num_blocks = self.num_blocks_per_row[src]
        #获取blcok_table的numpy视图（为了高效操作）
        block_table_np = self.block_table.np
        #执行实际的block映射移动
        #把源请求使用的所有Block ID 从 src 行复制到 tgt 行的前 num_blocks 个位置，# 例如：把 block_table[src, 0:5] 的内容复制到 block_table[tgt, 0:5]
        block_table_np[tgt, :num_blocks] = block_table_np[src, :num_blocks]
        # 更新目标行的 Block 数量，使其与源行一致
        self.num_blocks_per_row[tgt] = num_blocks

    def swap_row(self, src: int, tgt: int) -> None:
        src_tgt, tgt_src = [src, tgt], [tgt, src]
        self.num_blocks_per_row[src_tgt] = self.num_blocks_per_row[tgt_src]
        self.block_table.np[src_tgt] = self.block_table.np[tgt_src]

    def compute_slot_mapping(
        self, req_indices: np.ndarray, positions: np.ndarray
    ) -> None:
        """
        args:
            req_indices: oken 级别的请求索引。例如 [0,0,1,1,1,2,2] 表示前两个 token 属于请求0，接下来三个属于请求1
            positions: token 级别的位置。每个 token 在其请求序列中的绝对位置（position ids）。就是前面我们算的 positions_np
            两者长度相同，都等于total_num_scheduled_tokens
        """
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 0, K, K, K + 1, K + 1, K + 2, 2 * K, 2 * K, 2 * K + 1]
        #上边这一串是什么意思呢？如下所示
        # token序号, 属于请求, position, position // block_size, block_offsets, 最终 slot_mapping
        # 0,         0,         10,         5,                  0,               5 * 2 + 0 = 10
        # 1,         0,         11,         5,                  1,              5 * 2 + 1 = 11
        # 2,         1,         25,         12,                 1,              12 * 2 + 1 = 25
        # 3,        1,          26,         13,                 0,              13 * 2 + 0 = 26
        # 4,         1,         27,         13,                 1,              13 * 2 + 1 = 27
        # 5,        1,          28,         14,                 0,              14 * 2 + 0 = 28
        # 6,        1,          29,         14,                 1,              14 * 2 + 1 = 29
        # 7,        2,          8,          4,                   0,              4 * 2 + 0 = 8
        # 8,        2,           9,          4,                 1,              4 * 2 + 1 = 9
        # 9,        2,          10,          5,                 0,               5 * 2 + 0 = 10
        #所以最终的slot mapping是  [10, 11, 25, 26, 27, 28, 29, 8, 9, 10]
        # where K is the max_num_blocks_per_req and the block size is 2.
        # NOTE(woosuk): We can't simply use `token_indices // block_size`       不能简单用token全局索引//block_size来计算
        # here because M (max_model_len) is not necessarily divisible by        因为max_model_len不一定能被block_size整除，会导致计算错误
        # block_size.

        #=====处理Context Parallel（CP）+ Data Parallel（DP）分布式场景 ===
        total_cp_world_size = self.pcp_world_size * self.dcp_world_size
        total_cp_rank = self.pcp_rank * self.dcp_world_size + self.dcp_rank
        if total_cp_world_size > 1:
            #=====分布式kv cache情况 较为复杂====
            # Note(hc): The DCP implement store kvcache with an interleave          DCP 使用交错（interleave）方式存储 KV Cach
            # style, the kvcache for the token whose token_idx is i is              第 i 个 token 的 KV Cache 总是存储在 dcp_rank = i % cp_world_size 的 GPU 上。
            # always stored on the GPU whose dcp_rank equals i % cp_world_size:

            # Use a "virtual block" which equals to world_size * block_size         使用 "虚拟 block" 来计算 block_table 索引
            # for block_table_indices calculation.                                  虚拟 block 大小 = 真实 block_size × 总 CP 世界大小
            virtual_block_size = self.block_size * total_cp_world_size
            block_table_indices = (
                req_indices * self.max_num_blocks_per_req
                + positions // virtual_block_size
            )
            #从block_table中取出实际的物理block编号
            block_numbers = self.block_table.np.ravel()[block_table_indices]
            # Use virtual_block_size for mask calculation, which marks local
            # tokens. 计算虚拟block内的便宜
            virtual_block_offsets = positions % virtual_block_size
            #mask 用于判断当前token的kv 是否应该存储在本GPU上
            mask = (
                virtual_block_offsets
                // self.cp_kv_cache_interleave_size
                % total_cp_world_size
                == total_cp_rank
            )
            # Calculate local block_offsets 计算本地 block offset（去掉交错部分）
            block_offsets = (
                virtual_block_offsets
                // (total_cp_world_size * self.cp_kv_cache_interleave_size)
                * self.cp_kv_cache_interleave_size
                + virtual_block_offsets % self.cp_kv_cache_interleave_size
            )
            # Calculate slot_mapping 计算最终的slot mapping
            slot_mapping = block_numbers * self.block_size + block_offsets
            # Write final slots, use -1 for not-local  把结果写入slot_mapping，对于不属于本GPU的token，填-1
            self.slot_mapping.np[: req_indices.shape[0]] = np.where(
                mask, slot_mapping, -1
            )
        else:
            #===============单机/非CP场景 （常规情况）====================
            #最常见的计算方式
            #计算每个token对应的block在block_table中的索引
            block_table_indices = (
                req_indices * self.max_num_blocks_per_req + positions // self.block_size
            )
            #取出对应的物理block编号
            block_numbers = self.block_table.np.ravel()[block_table_indices]
            #计算该token在block内部的偏移量
            block_offsets = positions % self.block_size
            # slot = block_number * block_size + block_offset
            np.add(
                block_numbers * self.block_size,
                block_offsets,
                out=self.slot_mapping.np[: req_indices.shape[0]],
            )

    def commit_block_table(self, num_reqs: int) -> None:#block_table是什么？是非常核心的数据结构，记录了每个请求的token位置对应到哪个屋里KV CACHE block。举例请求A的100-127个token存在物理block#23里
        self.block_table.copy_to_gpu(num_reqs)          #只有拷贝到GPU内存之后，GPU上的attention kernel才能使用这些映射信息来找到正确的kv caches

    def commit_slot_mapping(self, num_tokens: int) -> None:
        self.slot_mapping.copy_to_gpu(num_tokens)

    def clear(self) -> None:
        self.block_table.gpu.fill_(0)
        self.block_table.cpu.fill_(0)

    @staticmethod
    def map_to_kernel_blocks(
        kv_manager_block_ids: np.ndarray,
        blocks_per_kv_block: int,
        kernel_block_arange: np.ndarray,
    ) -> np.ndarray:
        """Convert kv_manager_block_id IDs to kernel block IDs. 将KV Manager的block id 转换为attention kernel(计算曾)实际所需的Kernel Block ID
        这是blocktable中处理hybrid block模式的核心转换函数
        Example:
            # kv_manager_block_ids: 32 tokens,                      内存分配block_size=32 tokens
            # Kernel block size: 16 tokens                          kernel_block_size = 16 tokens
            # blocks_per_kv_block = 2                               即一个内存block对应2个kernel block
            >>> kv_manager_block_ids = np.array([0, 1, 2])          输入：kv_manager_block_ids = [0, 1, 2]
            >>> Result: [0, 1, 2, 3, 4, 5]                          输出： [0, 1, 2, 3, 4, 5]

            # Each kv_manager_block_id maps to 2 kernel block id:   转换过程：
            # kv_manager_block_id 0 → kernel block id [0, 1]        Block 0 → Kernel Block [0, 1]
            # kv_manager_block_id 1 → kernel block id [2, 3]        Block 1 → Kernel Block [2, 3]
            # kv_manager_block_id 2 → kernel block id [4, 5]        Block 2 → Kernel Block [4, 5]
        """
        ## 如果 blocks_per_kv_block == 1，说明内存 block 大小和 kernel block 大小一致
        # 无需任何转换，直接返回原来的 block ids（最常见的情况）
        if blocks_per_kv_block == 1:
            return kv_manager_block_ids

        kernel_block_ids = (
            kv_manager_block_ids.reshape(-1, 1) * blocks_per_kv_block
            + kernel_block_arange
        )

        return kernel_block_ids.reshape(-1)

    def get_device_tensor(self, num_reqs: int) -> torch.Tensor:
        """Returns the device tensor of the block table."""
        return self.block_table.gpu[:num_reqs]

    def get_cpu_tensor(self) -> torch.Tensor:
        """Returns the CPU tensor of the block table."""
        return self.block_table.cpu

    def get_numpy_array(self) -> np.ndarray:
        """Returns the numpy array of the block table."""
        return self.block_table.np

    def _make_buffer(
        self, *size: int | torch.SymInt, dtype: torch.dtype
    ) -> CpuGpuBuffer:
        return CpuGpuBuffer(
            *size, dtype=dtype, device=self.device, pin_memory=self.pin_memory
        )


class MultiGroupBlockTable:
    """The BlockTables for each KV cache group."""

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        pin_memory: bool,
        device: torch.device,
        block_sizes: list[int],
        kernel_block_sizes: list[int],
        num_speculative_tokens: int = 0,
        cp_kv_cache_interleave_size: int = 1,
    ) -> None:
        # Note(hc): each dcp rank only store
        # (max_model_len//dcp_world_size) tokens in kvcache,
        # so the block_size which used for calc max_num_blocks_per_req
        # must be multiplied by dcp_world_size.
        try:
            pcp_world_size = get_pcp_group().world_size
        except AssertionError:
            # PCP might not be initialized in testing
            pcp_world_size = 1
        try:
            dcp_world_size = get_dcp_group().world_size
        except AssertionError:
            # DCP might not be initialized in testing
            dcp_world_size = 1

        if len(kernel_block_sizes) != len(block_sizes):
            raise ValueError(
                f"kernel_block_sizes length ({len(kernel_block_sizes)}) "
                f"must match block_sizes length ({len(block_sizes)})"
            )

        total_cp_world_size = dcp_world_size * pcp_world_size

        self.block_tables = [
            BlockTable(
                block_size,
                max_num_reqs,
                max(
                    cdiv(max_model_len, block_size * total_cp_world_size),
                    1 + num_speculative_tokens,
                ),
                max_num_batched_tokens,
                pin_memory,
                device,
                kernel_block_size,
                cp_kv_cache_interleave_size,
            )
            for block_size, kernel_block_size in zip(block_sizes, kernel_block_sizes)
        ]

    def append_row(self, block_ids: tuple[list[int], ...], row_idx: int) -> None:
        for i, block_table in enumerate(self.block_tables):
            block_table.append_row(block_ids[i], row_idx)

    def add_row(self, block_ids: tuple[list[int], ...], row_idx: int) -> None:
        for i, block_table in enumerate(self.block_tables):
            block_table.add_row(block_ids[i], row_idx)

    def move_row(self, src: int, tgt: int) -> None:
        for block_table in self.block_tables:
            block_table.move_row(src, tgt)

    def swap_row(self, src: int, tgt: int) -> None:
        for block_table in self.block_tables:
            block_table.swap_row(src, tgt)

    def compute_slot_mapping(
        self, req_indices: np.ndarray, positions: np.ndarray
    ) -> None:
        for block_table in self.block_tables:
            block_table.compute_slot_mapping(req_indices, positions)

    def commit_block_table(self, num_reqs: int) -> None:
        for block_table in self.block_tables:
            block_table.commit_block_table(num_reqs)

    def commit_slot_mapping(self, num_tokens: int) -> None:
        for block_table in self.block_tables:
            block_table.commit_slot_mapping(num_tokens)

    def clear(self) -> None:
        for block_table in self.block_tables:
            block_table.clear()

    def __getitem__(self, idx: int) -> "BlockTable":
        """Returns the BlockTable for the i-th KV cache group."""
        return self.block_tables[idx]
