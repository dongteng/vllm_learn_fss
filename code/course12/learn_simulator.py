"""
@Author    : zhjm
@Time      : 2026/1/29 
@File      : learn_simulator.py
@Desc      : 
"""
import time
import threading
from typing import List, Dict, Any, Optional, Tuple


# 模拟vllm v1核心组件
class SimModelConfig:
    def __init__(self,hidden_size=1024, num_attention_heads=8, vocab_size=32000):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.vocab_size = vocab_size  # 新增: 用于模拟TP中的vocab分片


class SimSequence:
    #模拟一个正在处理的序列
    def __init__(self, seq_id:int , prompt:str, max_tokens:int):
        self.seq_id = seq_id
        self.prompt = prompt
        self.prompt_tokens = [f"PromptToken_{t}" for t in prompt.split()]
        self.generated_tokens:List[str]= []
        self.max_tokens = max_tokens
        self.is_finished = False
        self.is_prefill = True #新增  区分prefill和decode阶段

    def append_token(self, token:str):
        #如果这是第一次调用append_token(也就是刚从prefill转到decode) ，把is_prefill改为False->标记预填充已经结束，现在正式生成token
        if self.is_prefill:
            self.is_prefill = False
        self.generated_tokens.append(token)
        if len(self.generated_tokens) >= self.max_tokens:
            self.is_finished = True #检查生成的token数量是否已经达到最大限制，如果是就标记
    def get_current_length(self):
        return len(self.prompt_tokens) + len(self.generated_tokens)

    def __repr__(self):
        return f"Seq(id={self.seq_id}, tokens={self.get_current_length()}/{len(self.prompt_tokens) + self.max_tokens}, prefill={self.is_prefill})"

class SimPagedKVCache:
    #模拟PagedAttention的K V缓存，使用分页块
    def __init__(self,block_size:int =16):
        self.block_size = block_size
        self.blocks: Dict[int, List[Tuple[Any,Any]]] ={}
        #self.blocks 就是整个系统的“KV 缓存仓库”，按 seq_id 分开管理，每个序列拥有自己的一组“内存块”。
        #所以更形象的内存布局大概是这样的：
        # blocks = {
        #     seq_id: [
        #         block0: [(K0,V0), (K1,V1), ..., (K15,V15)],    # 16 个 token 的 KV
        #         block1: [(K16,V16), ..., (K31,V31)],
        #         block2: [(K32,V32), ...,          ]             # 最后一个块可能不满
        #     ]
        # }
    def allocate_for_seq(self,seq_id:int):
        if seq_id not in self.blocks:
            self.blocks[seq_id] = []

    def append_kv(self,seq_id:int ,new_kv:Tuple[Any,Any]):
        #把模型刚刚为某个序列seq_id新计算出来的一对（K V）加到这个序列的KV缓存里，并自动处理当前块，块满了就先新开一个块
        if not self.blocks[seq_id] or len(self.blocks[seq_id][-1]) == self.block_size:
            self.blocks[seq_id].append([])
        self.blocks[seq_id][-1].append(new_kv)

    def get_cache_size(self,seq_id:int)->int:
        if seq_id not in self.blocks:
            return 0
        return sum(len(block) for block in self.blocks[seq_id])


class SimScheduler:
    #模拟vllm的调度器 Scheduler,更贴近真实：考虑KV 快分配和连续批处理
    def __init__(self, max_batch_size:int =32, max_kv_blocks:int =1024):
        self.request_queue : List[SimSequence] = []
        self.activate_sequences : Dict[int, SimSequence] = {}
        self.max_batch_size = max_batch_size #新增：批次大小限制
        self.max_kv_blocks = max_kv_blocks #新增：KV块数量限制
        self.used_kv_blocks = 0 #跟踪已用的KV块数量
        print("[Scheduler]初始化已经完成")
    def add_sequence(self,seq:SimSequence):
        print(f"[Scheduler]接收到新请求：Seq {seq.seq_id}")
        #有新用户请求来了，先放进排队队列，然后马上尝试调度
        self.request_queue.append(seq)
        self._try_schedule_pending()
    def _try_schedule_pending(self):
        #尝试将pending请求 移到activate ,如果有足够的kv 块
        while self.request_queue: #只要还有等待的序列
            seq = self.request_queue[0] #看队头 先进先出
            prompt_len = len(seq.prompt_tokens)

            #计算这个序列总共需要多少个KV块
            needed_blocks = (prompt_len + seq.max_tokens)//16 +1 #这里为什么+1？
            if (len(self.activate_sequences)< self.max_batch_size and self.used_kv_blocks + needed_blocks <= self.max_kv_blocks):
                self.request_queue.pop(0)
                self.activate_sequences[seq.seq_id] = seq
                self.used_kv_blocks += needed_blocks
                print(f"[Scheduler]已激活Seq {seq.seq_id} 到activate (使用{needed_blocks}块）")
            else:
                break #无法调度更多

    def schedule(self)-> Tuple[List[SimSequence],List[SimSequence]]:
        #决定当前步骤的批次，分离prefill和decode序列（vLLM支持混合批处理，但这里简化分离）
        #在真实vLLM种，使用SwapIn/SwapOut 来管理内存

        self._try_schedule_pending() #先尝试把排队的序列激活一些进来

        #从所有active序列中挑出还在prefill的
        prefill_sequences = [s for s in self.activate_sequences.values() if s.is_prefill]
        #挑出已经在decode且还没结束的
        decode_sequences = [s for s in self.activate_sequences.values() if not s.is_prefill]
        return prefill_sequences,decode_sequences

    def finish_sequence(self,seq_id:int):
        #当序列完成时，将其从活动池中移除并释放KV块
        if seq_id in self.activate_sequences:
            seq = self.activate_sequences[seq_id]
            released_blocks = (seq.get_current_length() //16) +1
            self.used_kv_blocks -= released_blocks
            print(f"[Scheduler]完成Seq {seq_id}，释放{released_blocks}块")
            del self.activate_sequences[seq_id]
            self._try_schedule_pending()

    def has_pending_sequences(self)->bool:
        #判断系统是否空闲
        #self.activate_sequences 里还有序列（正在运行中的序列 > 0）  → 也就是还有序列在 active 状态（有的在 prefill，有的在 decode 生成中）
        #self.request_queue 里还有序列在排队（等待队列 > 0） → 也就是有新请求来了，但因为资源不够（batch size 满或 KV 块不够），还没被调度激活
        return len(self.activate_sequences) > 0 or len(self.request_queue) > 0



class SimWorker:
    #模拟一个vllm worker 对应一个GPU和一个TP分片，更贴近真实：分片KV cache和partial computation
    def __init__(self,rank:int , tp_size:int , config:SimModelConfig):
        self.rank = rank #rank是分布式计算最常见的概念，在这里的意思是这个worker（这个GPU/进程）在整个TP组中的编号
        self.tp_size = tp_size
        self.config = config

        #模拟模型权重分片（TP）：Attention heads 和Vocab分片
        heads_per_worker = config.num_attention_heads // tp_size

        #举例：假设模型有 64 个 attention heads，tp_size = 4
        #worker的rank              负责的attention heads范围               解释
        #0                          [0, 16)                                0~15
        #1                          [16, 32)                               16~31
        #2                          [32, 48)                               32~47
        #3                          [48, 64)                               48~63
        self.my_head_start = rank *heads_per_worker
        self.my_head_end = (rank +1) * heads_per_worker

        vocab_per_worker = config.vocab_size // tp_size
        self.my_vocab_start = rank * vocab_per_worker
        self.my_vocab_end = (rank +1) * vocab_per_worker

        #模拟PagedAttention KV 缓存分片（只存储我的head分片)
        self.kv_cache = SimPagedKVCache()
        print(f"Worker {self.rank}已经启动。负责Heads {self.my_head_start}~{self.my_head_end-1}，Vocab {self.my_vocab_start}~{self.my_vocab_end-1}")

    def execute_step(self,batch:List[SimSequence],is_prefill:bool)->Dict[int,str]:
        #模拟一个前向传播步骤：prefill或decode
        #prefill:处理整个prompt (多token)
        #decode:处理一个token
        print(f"  [Worker {self.rank}] 执行 {'Prefill' if is_prefill else 'Decode'} 批次 ({len(batch)} 个序列)...")

        partial_results = {}
        for seq in batch:
            seq_id = seq.seq_id
            self.kv_cache.allocate_for_seq(seq_id)

            #模拟访问paged kv 缓存
            cached_len = self.kv_cache.get_cache_size(seq_id)
            input_len = len(seq.prompt_tokens) if is_prefill else 1

            #模拟计算partial logits(只计算我的vocab size)
            partial_logit = f"PartialLogit(Seq:{seq_id}, Worker:{self.rank}, CacheLen:{cached_len}, InputLen:{input_len})"

            #存储新的kv 分片（为input_len个位置）
            #为本次输入的每个token，生成并追加自己的KV 分片
            for _ in range(input_len):
                new_kv_shard = (f"K_shard(Heads {self.my_head_start}-{self.my_head_end - 1})",
                                f"V_shard(Heads {self.my_head_start}-{self.my_head_end - 1})")
                self.kv_cache.append_kv(seq_id, new_kv_shard)
            partial_results[seq_id] = partial_logit

        # 模拟计算延迟 (prefill 更长)
        time.sleep(0.2 if is_prefill else 0.1)
        print(f" [Worker {self.rank}] 执行完成")
        return partial_results


class SimLLMEngine:
    #模拟vLLM引擎（LLMEngine），负责协调：更贴近v1（连续批处理，prefill/decode分离）
    def __init__(self,model_id:str, tensor_parallel_size:int):
        print(f"[Engine] 正在启动vLLM 分布式模拟···")
        print(f"[Engine] 模型：{model_id}, TP大小：{tensor_parallel_size}")

        self.tp_size = tensor_parallel_size
        self.config = SimModelConfig()
        self.scheduler = SimScheduler()
        self.workers : List[SimWorker] = []
        self.finished_sequences : Dict[int,str] = {}
        self.next_seq_id = 0

        self._create_workers()
    def _create_workers(self):
        #模拟创建TP_SIZE个工作节点（ray workers）
        print(f"[Engine] 正在创建{self.tp_size}个分布式worker")
        for rank in range(self.tp_size):
            worker = SimWorker(rank=rank, tp_size=self.tp_size, config= self.config)
            self.workers.append(worker)
        print(f"[Engine] 所有Worker 已准备就绪")

    def step(self):
        #模拟vllm引擎的单个执行步骤（LLMEngine.step）：处理prefill和decode批次
        prefill_batch,decode_batch = self.scheduler.schedule()
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
    def _execute_batch(self,batch:List[SimSequence],is_prefill:bool):
        #分布式执行：用线程模拟并行
        all_patrial_results : List[Dict[int,str]] = [{} for _ in self.workers]
        def _run_worker(rank:int):
            partial_result = self.workers[rank].execute_step(batch,is_prefill)
            all_patrial_results[rank] = partial_result

        thread = []
        for i in range(self.tp_size):
            t = threading.Thread(target=_run_worker,args=(i,))
            thread.append(t)
            t.start()
        for t in thread:
            t.join()

        #模拟 all-gather / all-reduce (在rank 0 聚合)
        print(f"[Engine] 模拟 All-Gather: 从 {self.tp_size} 个 Worker 收集部分结果...")
        aggregated_logits : Dict[int,List[str]] = {}
        for seq in batch:
            seq_id = seq.seq_id
            aggregated_logits[seq_id] = [pr[seq_id] for pr in all_patrial_results]

        #采样 (在 rank 0)
        new_tokens = self._sample_tokens(aggregated_logits,batch,is_prefill)

        #更新序列
        for seq in batch:
            new_token = new_tokens[seq.seq_id] #prefill: 第一个token, decode: 下一个
            seq.append_token(new_token)
            print(f"[Engine] 采样结果: Seq {seq.seq_id} 生成新 Token -> '{new_token}'")
            if seq.is_finished:
                self.scheduler.finish_sequence(seq.seq_id)
                self.finished_sequences[seq.seq_id] = seq
    def _sample_tokens(self, aggregated_logits: Dict[int, List[str]], batch: List[SimSequence], is_prefill: bool) -> Dict[int, str]:
        #模拟token采样： prefill生成第一个token, decode生成下一个
        new_tokens = {}
        for seq in batch:
            gen_idx = len(seq.generated_tokens) + (1 if is_prefill else 0)
            new_token = f"GenToken_{gen_idx}"
            new_tokens[seq.seq_id] = new_token
        return new_tokens
    def generate(self, prompts: List[str], max_tokens: int) -> List[SimSequence]:
        #模拟vLLM高层generate接口：添加到scheduler并运行循环
        print(f"\n[Engine] 收到 {len(prompts)} 个 'generate' 请求。")
        for prompt in prompts:
            seq_id = self.next_seq_id
            self.next_seq_id +=1
            seq = SimSequence(seq_id,prompt,max_tokens)
            self.scheduler.add_sequence(seq)
        while self.scheduler.has_pending_sequences():
            self.step()
            time.sleep(0.5) #观察延迟
        print('\n[Engine] 所有请求处理完毕')
        return [self.finished_sequences[i] for i in range(len(prompts))]

#运行模拟
if __name__ == "__main__":

    TP_SIZE = 4

    sim_engine = SimLLMEngine(model_id= "qwen3/qwen3-1.7B",tensor_parallel_size=TP_SIZE)

    prompts_to_run = [
        "Hello my name is",
        # "The capital of France is",
        # "vLLM simulates distributed inference by",
        ]
    outputs = sim_engine.generate(prompts=prompts_to_run,max_tokens=5)

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




















