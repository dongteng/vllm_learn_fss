"""
@Author    : zhjm
@Time      : 2026/2/1 
@File      : learn_scatter.py
@Desc      : 
"""
import torch
import torch.distributed as dist

def init_process():
    #初始化 PyTorch 的分布式进程组，使用 NCCL 作为 GPU 间通信后端（这是目前多 GPU 训练/推理最快、最常用的后端）。
    dist.init_process_group(backend='nccl')

    #torch.cuda.set_device()：设置当前进程默认使用的CUDA设备相当于告诉yTorch：“我这个进程以后所有.cuda()、model.to('cuda')都默认用rank对应的那张卡”
    torch.cuda.set_device(dist.get_rank()) #让当前进程绑定到对应的 GPU 上

def example_scatter():
    #Scatter：纯数据分发，没有做任何计算 把一个节点的数据切片后分给所有人（每个人只拿一部分）
    #只有rank0负责准备数据，其他进程不需要准备
    if dist.get_rank() == 0:
        scatter_list = [
            torch.tensor([i] * 5, dtype=torch.float32).cuda()
            for i in range(dist.get_world_size())
            ]
        print(f"Rank 0: Tensor to scatter: {scatter_list}")
    else:
        scatter_list = None
    tensor = torch.empty(5, dtype=torch.float32).cuda() #每个进程准备接受缓冲区 创建一个空的长度为5的tensor

    print(f"Before scatter on rank {dist.get_rank()}: {tensor}")
    #下边这句的核心调用时 把src=0（rank=0）把scatter_list中的每个tensor分发出去
    #规则是rank 0 收到 scatter_list[0]  rank 1 收到 scatter_list[1]  rank 2 收到 scatter_list[2]  rank 3 收到 scatter_list[3]
    #scatter_list 是一个长度等于 world_size 的列表，列表的第 i 个元素（scatter_list[i]）会被发送给 rank i
    dist.scatter(tensor, scatter_list, src=0)
    print(f"After scatter on rank {dist.get_rank()}: {tensor}")
    print("="*20)

if __name__ == "__main__":
    init_process()
    example_scatter()