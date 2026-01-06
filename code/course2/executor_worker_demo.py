import multiprocessing as mp #python的多进程库
import time
import random
import os


def dummy_execute_model(args):
    """模拟 execute_model 计算"""
    scheduler_output, = args
    time.sleep(random.uniform(0.2, 0.5))
    return f"logits_from_rank{os.getpid()}_{scheduler_output}"


def worker(rank: int, work_q: mp.Queue, result_q: mp.Queue):

    pid = os.getpid()
    print(f"[Worker {rank}] 启动，pid={pid}")

    # 1. 握手
    result_q.put("READY")

    # 2. 映射方法名 → 本地可调用对象
    method_table = {
        "execute_model": dummy_execute_model,
    }

    # 3. 主循环
    while True:
        item = work_q.get()
        if item is None:                      # Poison Pill
            result_q.put(None)
            break
        method_name, args, kwargs = item
        func = method_table[method_name]
        print(f"[Worker {rank}] 执行 {method_name}{args}")
        output = func(args, **(kwargs or {}))
        result_q.put(output)


def master(num_workers: int = 2):
    #master只干三件事 1.往wark_q里塞活  2.从result_queues里边收集结果 3.统计谁干完谁退出
    print("=== Master 启动 ===")
    work_q = mp.Queue() #派工队列：所有worker共享的命令队列：Master往这里边put Worker从里边get
    result_queues = [mp.Queue() for _ in range(num_workers)] #结果队列 一人一个
    procs = [mp.Process(target=worker, args=(rank, work_q, result_queues[rank]))
             for rank in range(num_workers)] #创建进程 但还没启动
    for p in procs:
        p.start() #这一行后 才真的spawn出紫禁城，worker开始执行，Master不等它们，继续往下跑

    # 等待 READY  #在worker里 初始化完成后，向自己的result_queue发送一条"READY"
    while sum(rq.get() == "READY" for rq in result_queues) < num_workers:
        pass
    print(">>> 所有 Worker 就绪 <<<")  #这里还有不就绪的时候？

    # 下发 RPC 调用
    calls = [
        ("execute_model", ("scheduler_output_0",), {}),
        ("execute_model", ("scheduler_output_1",), {}),
        ("execute_model", ("scheduler_output_2",), {}),
    ]
    #calls * num_workers 是什么鬼？第一次见这种写法啊
    for call in calls * num_workers:        # 让每个 Worker 都执行一遍
        work_q.put(call)

    # 发结束信号
    for _ in range(num_workers):
        work_q.put(None)

    # 收集结果
    results = {rank: [] for rank in range(num_workers)}
    finished = 0
    while finished < num_workers:
        for rank, rq in enumerate(result_queues):
            if not rq.empty():
                item = rq.get()
                if item is None:
                    finished += 1
                else:
                    results[rank].append(item)
    for p in procs: #等待所有worker进程结束
        p.join()
    print("=== 最终结果汇总 ===")
    for rank, outs in results.items():
        print(f"Worker {rank}: {outs}")


if __name__ == "__main__":
    #规定以后创建子进程，用spawn这种启动方式；force=True就算之前已经设hi过 也要强行改成spawn告诉 Python：以后起子进程时，不要“复制我自己”，而是“新开一个干净的 Python 进程”。
    mp.set_start_method("spawn", force=True)
    master(num_workers=1)