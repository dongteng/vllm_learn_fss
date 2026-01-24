"""
@Author    : zhjm
@Time      : 2026/1/18 
@File      : learn_cuda_graph.py
@Desc      : 
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import torch
import time

assert torch.cuda.is_available(), "CUDA not available"

device = 'cuda'
N  = 1024
num_warmup  = 5
num_iters = 100

#创建固定大小的输入（必须在捕获前分配好）
x = torch.randn(N, N,device=device)
y = torch.randn(N, N,device=device)
z = torch.empty(N, N,device=device)

graph = torch.cuda.CUDAGraph()
stream = torch.cuda.Stream()

with torch.cuda.stream(stream):
    for _ in range(num_warmup):
        z = torch.mm(x, y) + torch.sin(x)
torch.cuda.synchronize()

with  torch.cuda.graph(graph):
    z = torch.mm(x,y) + torch.sin(x)
torch.cuda.synchronize()

start = time.time()
for _ in range(num_iters):
    graph.replay()
torch.cuda.synchronize()
graph_time = time.time() - start
print(f"CUDA Graph time: {graph_time:.4f} s")

start = time.time()
for _ in range(num_iters):
    z = torch.mm(x, y) + torch.sin(x)      # 每次都重新发射 kernel
torch.cuda.synchronize()
normal_time = time.time() - start
print(f"Normal time:     {normal_time:.4f} s")