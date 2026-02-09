"""
@Author    : zhjm
@Time      : 2026/2/1 
@File      : learn_reduce_scatter.py
@Desc      : 
"""
import torch
import torch.distributed as dist

def init():
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(dist.get_rank())

def example_reduce_scatter():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    #在 reduce_scatter 中，每个进程提供的 input_tensor 列表长度必须等于 world_size，且每个 tensor 的 shape 必须完全相同。
    #假设world_size = 4，我们来看每个rank会生成什么样的

    input_tensor = [
        # torch.tensor([(rank + 1) * i for i in range(1, 5)], dtype=torch.float32).cuda()**(j+1)
        torch.tensor([(rank + 1) * i for i in range(1, 5)], dtype=torch.float32).cuda()

        for j in range(world_size)
        ]
    output_tensor = torch.zeros(4, dtype=torch.float32).cuda()
    print(f"Before ReduceScatter on rank {rank}: {input_tensor}")
    dist.reduce_scatter(output_tensor, input_tensor, op=dist.ReduceOp.SUM)
    print(f"\n After ReduceScatter on rank {rank}: {output_tensor}")
if __name__ == '__main__':
    init()
    example_reduce_scatter()