"""
@Author    : zhjm
@Time      : 2026/1/29 
@File      : learn_simple_linear.py
@Desc      : 
"""
import torch
import torch.nn as nn

class SimpleColumnParallelLinear(nn.Module):
    def __init__(self, input_size, output_size, tp_size=2, gather_output=False):
        super().__init__()
        assert input_size % tp_size == 0
        self.tp_size  = tp_size
        self.output_size = output_size
        self.out_per_part = output_size // tp_size
        self.gather_output = gather_output

        #每个partition 管理一个子线性层
        self.linears = nn.ModuleList([
            nn.Linear(input_size, self.out_per_part)
            for _ in range(tp_size)
            ])
    def forward(self, x):
        parts = [ lin(x) for lin in self.linears]
        return torch.cat(parts,dim=-1)


if __name__ =="__main__":
    batch,in_dim ,out_dim = 4,8,12
    tp_size =4
    model = SimpleColumnParallelLinear(in_dim,out_dim,tp_size=4,gather_output=True)
    x = torch.randn(batch,in_dim)
    y_full = model(x)

    model2= SimpleColumnParallelLinear(in_dim,out_dim,tp_size=4,gather_output=False)
    parts = model2(x)
    for i,p in enumerate(parts):
        print(f"第 {i} 个部分输出形状: {p.shape}")