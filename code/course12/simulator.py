import time
import threading
from typing import List, Dict, Any, Optional, Tuple

# ---- 模拟 vLLM v1 的核心组件 ---
class SimModelConfig:
    """模拟模型配置"""
    def __init__(self, hidden_size=1024, num_attention_heads=8, vocab_size=32000):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.vocab_size = vocab_size  # 新增: 用于模拟 TP 中的 vocab 分片

class SimSequence:
    """模拟一个正在处理的序列（请求）"""
    def __init__(self, seq_id: int, prompt: str, max_tokens: int):
        self.seq_id = seq_id
        self.prompt = prompt
        # 模拟 "Tokenization"
        self.prompt_tokens = [f"PromptToken_{t}" for t in prompt.split()]
        self.generated_tokens: List[str] = []
        self.max_tokens = max_tokens
        self.is_finished = False
        self.is_prefill = True  # 新增: 区分 prefill 和 decode 阶段

    def append_token(self, token: str):
        if self.is_prefill:
            self.is_prefill = False
        self.generated_tokens.append(token)
        if len(self.generated_tokens) >= self.max_tokens:
            self.is_finished = True

    def get_current_length(self) -> int:
        return len(self.prompt_tokens) + len(self.generated_tokens)

    def __repr__(self):
        return f"Seq(id={self.seq_id}, tokens={self.get_current_length()}/{len(self.prompt_tokens) + self.max_tokens}, prefill={self.is_prefill})"

class SimPagedKVCache:
    """模拟 PagedAttention 的 KV 缓存，使用分页块"""
    def __init__(self, block_size: int = 16):  # 模拟块大小
        self.block_size = block_size
        self.blocks: Dict[int, List[Tuple[Any, Any]]] = {}  # seq_id -> list of blocks (each block: list of (K, V) per position)

    def allocate_for_seq(self, seq_id: int):
        if seq_id not in self.blocks:
            self.blocks[seq_id] = []

    def append_kv(self, seq_id: int, new_kv: Tuple[Any, Any]):
        if not self.blocks[seq_id] or len(self.blocks[seq_id][-1]) >= self.block_size:
            self.blocks[seq_id].append([])  # 新块
        self.blocks[seq_id][-1].append(new_kv)

    def get_cache_size(self, seq_id: int) -> int:
        if seq_id not in self.blocks:
            return 0
        return sum(len(block) for block in self.blocks[seq_id])

class SimScheduler:
    """模拟 vLLM 的调度器 (Scheduler)，更贴近真实: 考虑 KV 块分配和连续批处理"""
    def __init__(self, max_batch_size: int = 32, max_kv_blocks: int = 1024):
        self.request_queue: List[SimSequence] = []
        self.active_sequences: Dict[int, SimSequence] = {}
        self.max_batch_size = max_batch_size  # 新增: 批次大小限制
        self.max_kv_blocks = max_kv_blocks  # 新增: 总 KV 块限制 (模拟内存)
        self.used_kv_blocks = 0  # 跟踪已用块
        print("[Scheduler] 初始化完成。")

    def add_sequence(self, seq: SimSequence):
        """模拟新请求进入，并检查是否能立即调度"""
        print(f"[Scheduler] 接收到新请求: Seq {seq.seq_id}")
        self.request_queue.append(seq)
        self._try_schedule_pending()

    def _try_schedule_pending(self):
        """尝试将 pending 请求移到 active，如果有足够 KV 块"""
        while self.request_queue:
            seq = self.request_queue[0]
            prompt_len = len(seq.prompt_tokens)
            needed_blocks = (prompt_len + seq.max_tokens) // 16 + 1  # 粗略估计块数 (假设 block_size=16)
            if (len(self.active_sequences) < self.max_batch_size and
                self.used_kv_blocks + needed_blocks <= self.max_kv_blocks):
                self.request_queue.pop(0)
                self.active_sequences[seq.seq_id] = seq
                self.used_kv_blocks += needed_blocks
                print(f"[Scheduler] 调度 Seq {seq.seq_id} 到 active (使用 {needed_blocks} 块)")
            else:
                break  # 无法调度更多

    def schedule(self) -> Tuple[List[SimSequence], List[SimSequence]]:
        """
        决定当前步骤的批次: 分离 prefill 和 decode 序列 (vLLM 支持混合批处理，但这里简化分离)
        在真实 vLLM 中，使用 SwapIn/SwapOut 来管理内存。
        """
        self._try_schedule_pending()
        prefill_batch = [s for s in self.active_sequences.values() if s.is_prefill]
        decode_batch = [s for s in self.active_sequences.values() if not s.is_prefill and not s.is_finished]
        return prefill_batch, decode_batch

    def finish_sequence(self, seq_id: int):
        """当序列完成时，将其从活动池中移除并释放 KV 块"""
        if seq_id in self.active_sequences:
            seq = self.active_sequences[seq_id]
            released_blocks = (seq.get_current_length() // 16) + 1
            self.used_kv_blocks -= released_blocks
            print(f"[Scheduler] 序列 {seq_id} 已完成。释放 {released_blocks} 块。")
            del self.active_sequences[seq_id]
            self._try_schedule_pending()  # 尝试调度 pending

    def has_pending_sequences(self) -> bool:
        return len(self.active_sequences) > 0 or len(self.request_queue) > 0

class SimWorker:
    """
    模拟一个 vLLM Worker (对应一个 GPU 和一个 TP 分片)，更贴近真实: 分片 KV cache 和 partial computations
    """
    def __init__(self, rank: int, tp_size: int, config: SimModelConfig):
        self.rank = rank
        self.tp_size = tp_size
        self.config = config
        
        # 模拟模型权重分片 (TP): Attention heads 和 Vocab 分片
        heads_per_worker = config.num_attention_heads // tp_size
        self.my_head_start = rank * heads_per_worker
        self.my_head_end = (rank + 1) * heads_per_worker
        vocab_per_worker = config.vocab_size // tp_size
        self.my_vocab_start = rank * vocab_per_worker
        self.my_vocab_end = (rank + 1) * vocab_per_worker
        
        # 模拟 PagedAttention KV 缓存分片 (只存储我的 head 分片)
        self.kv_cache = SimPagedKVCache()
        
        print(f"[Worker {self.rank}] 已启动。负责 Heads: {self.my_head_start}-{self.my_head_end-1}, Vocab: {self.my_vocab_start}-{self.my_vocab_end-1}")

    def execute_step(self, batch: List[SimSequence], is_prefill: bool) -> Dict[int, str]:
        """
        模拟一个前向传播步骤: prefill 或 decode
        - Prefill: 处理整个 prompt (多 token)
        - Decode: 处理单个 token
        """
        print(f"  [Worker {self.rank}] 执行 {'Prefill' if is_prefill else 'Decode'} 批次 ({len(batch)} 个序列)...")
        partial_results = {}
        
        for seq in batch:
            seq_id = seq.seq_id
            self.kv_cache.allocate_for_seq(seq_id)
            
            # 模拟访问 Paged KV 缓存
            cached_len = self.kv_cache.get_cache_size(seq_id)
            input_len = len(seq.prompt_tokens) if is_prefill else 1  # prefill: 多 token, decode: 1

            # 模拟计算 partial logits (只计算我的 vocab slice)
            partial_logit = f"PartialLogit(Seq:{seq_id}, Worker:{self.rank}, CacheLen:{cached_len}, InputLen:{input_len})"
            
            # 存储新的 KV 分片 (为 input_len 个位置)
            for _ in range(input_len):
                new_kv_shard = (f"K_shard(Heads {self.my_head_start}-{self.my_head_end-1})", 
                                f"V_shard(Heads {self.my_head_start}-{self.my_head_end-1})")
                self.kv_cache.append_kv(seq_id, new_kv_shard)
            
            partial_results[seq_id] = partial_logit
            
        # 模拟计算延迟 (prefill 更长)
        time.sleep(0.2 if is_prefill else 0.1)
        
        print(f"  [Worker {self.rank}] 完成计算。")
        return partial_results

class SimLLMEngine:
    """
    模拟 vLLM 引擎 (LLMEngine)，负责协调: 更贴近真实 v1 (连续批处理, prefill/decode 分离)
    """
    def __init__(self, model_id: str, tensor_parallel_size: int):
        print(f"[Engine] 正在启动 vLLM 分布式模拟...")
        print(f"[Engine] 模型: {model_id}, TP 大小: {tensor_parallel_size}")
        
        self.tp_size = tensor_parallel_size
        self.config = SimModelConfig()
        self.scheduler = SimScheduler()
        self.workers: List[SimWorker] = []
        
        self.finished_sequences: Dict[int, SimSequence] = {}
        self.next_seq_id = 0
        
        self._create_workers()

    def _create_workers(self):
        """模拟创建 TP_SIZE 个工作节点 (Ray actors)"""
        print(f"[Engine] 正在创建 {self.tp_size} 个分布式 Worker...")
        for i in range(self.tp_size):
            worker = SimWorker(rank=i, tp_size=self.tp_size, config=self.config)
            self.workers.append(worker)
        print("[Engine] 所有 Worker 已准备就绪。")

    def step(self):
        """
        模拟 vLLM 引擎的单个执行步骤 (LLMEngine.step): 处理 prefill 和 decode 批次
        """
        prefill_batch, decode_batch = self.scheduler.schedule()
        if not prefill_batch and not decode_batch:
            return False

        print(f"\n--- [Engine Step] ---")
        if prefill_batch:
            print(f"[Engine] 处理 Prefill 批次: {len(prefill_batch)} 个序列 {[s.seq_id for s in prefill_batch]}")
            self._execute_batch(prefill_batch, is_prefill=True)
        if decode_batch:
            print(f"[Engine] 处理 Decode 批次: {len(decode_batch)} 个序列 {[s.seq_id for s in decode_batch]}")
            self._execute_batch(decode_batch, is_prefill=False)

        return True

    def _execute_batch(self, batch: List[SimSequence], is_prefill: bool):
        # 分布式执行: 用线程模拟并行
        all_partial_results: List[Dict[int, str]] = [{} for _ in self.workers]
        
        def _run_worker(rank: int):
            partial_result = self.workers[rank].execute_step(batch, is_prefill)
            all_partial_results[rank] = partial_result

        threads = []
        for i in range(self.tp_size):
            t = threading.Thread(target=_run_worker, args=(i,))
            threads.append(t)
            t.start()
            
        for t in threads:
            t.join()

        # 模拟 All-Gather / All-Reduce (在 rank 0 聚合)
        print(f"[Engine] 模拟 All-Gather: 从 {self.tp_size} 个 Worker 收集部分结果...")
        
        aggregated_logits: Dict[int, List[str]] = {}
        for seq in batch:
            seq_id = seq.seq_id
            aggregated_logits[seq_id] = [pr[seq_id] for pr in all_partial_results]

        # 采样 (在 rank 0)
        new_tokens = self._sample_tokens(aggregated_logits, batch, is_prefill)

        # 更新序列
        for seq in batch:
            new_token = new_tokens[seq.seq_id]  # prefill: 第一个 token, decode: 下一个
            seq.append_token(new_token)
            print(f"[Engine] 采样结果: Seq {seq.seq_id} 生成新 Token -> '{new_token}'")
            
            if seq.is_finished:
                self.scheduler.finish_sequence(seq.seq_id)
                self.finished_sequences[seq.seq_id] = seq

    def _sample_tokens(self, aggregated_logits: Dict[int, List[str]], batch: List[SimSequence], is_prefill: bool) -> Dict[int, str]:
        """模拟 Token 采样: prefill 生成第一个 token, decode 生成下一个"""
        new_tokens = {}
        for seq in batch:
            gen_idx = len(seq.generated_tokens) + (1 if is_prefill else 0)
            new_token = f"GenToken_{gen_idx}"
            new_tokens[seq.seq_id] = new_token
        return new_tokens

    def generate(self, prompts: List[str], max_tokens: int) -> List[SimSequence]:
        """
        模拟 vLLM 的高层 generate 接口: 添加到 scheduler 并运行循环
        """
        print(f"\n[Engine] 收到 {len(prompts)} 个 'generate' 请求。")
        
        for prompt in prompts:
            seq_id = self.next_seq_id
            self.next_seq_id += 1
            seq = SimSequence(seq_id=seq_id, prompt=prompt, max_tokens=max_tokens)
            self.scheduler.add_sequence(seq)

        while self.scheduler.has_pending_sequences():
            self.step()
            time.sleep(0.5)  # 观察延迟

        print("\n[Engine] 所有请求处理完毕。")
        
        return [self.finished_sequences[i] for i in range(len(prompts))]

# -------- 运行模拟 ------
if __name__ == "__main__":
    
    TP_SIZE = 4
    
    sim_engine = SimLLMEngine(
        model_id="meta-llama/Llama-2-7b-simulated",
        tensor_parallel_size=TP_SIZE
    )
    
    prompts_to_run = [
        "Hello my name is",
        "The capital of France is",
        "vLLM simulates distributed inference by",
    ]
    
    outputs = sim_engine.generate(prompts=prompts_to_run, max_tokens=5)
    
    print("\n--- [模拟完成：最终输出] ---")
    for seq in outputs:
        print(f"Prompt: '{seq.prompt}'")
        print(f"Output: {seq.prompt_tokens + seq.generated_tokens}")
        print("-" * 20)
        
    print("\n--- [内部状态检查] ---")
    worker_0_cache = sim_engine.workers[0].kv_cache
    seq_0_id = outputs[0].seq_id
    print(f"Worker 0 为 Seq {seq_0_id} 缓存的 KV 位置数量: {worker_0_cache.get_cache_size(seq_0_id)}")
    if worker_0_cache.blocks.get(seq_0_id):
        print(f"Worker 0 缓存的第一个 KV 分片 (模拟): {worker_0_cache.blocks[seq_0_id][0][0]}")
