import torch
import torch.nn as nn

class SimpleColumnParallelLinear(nn.Module):
    def __init__(self, input_size, output_size, tp_size=2, gather_output=False):
        super().__init__()
        assert output_size % tp_size == 0
        self.tp_size = tp_size
        self.output_size = output_size
        self.out_per_part = output_size // tp_size
        self.gather_output = gather_output
        
        # 每个 partition 管理一个子线性层
        self.linears = nn.ModuleList([
            nn.Linear(input_size, self.out_per_part)
            for _ in range(tp_size)
        ])
    
    def forward(self, x):
        # x: [batch, input_size]
        parts = [lin(x) for lin in self.linears]  # 每个部分if self.gather_output:
        return torch.cat(parts, dim=-1)        # 拼接得到完整 yelse:
            # return parts                           # 返回子结果列表# 测试
batch, in_dim, out_dim = 4, 8, 12
tp_size = 4

model = SimpleColumnParallelLinear(in_dim, out_dim, tp_size=tp_size, gather_output=True)
x = torch.randn(batch, in_dim)
y_full = model(x)  # shape: [4, 6]print("完整输出 Y:", y_full, y_full.shape)

model2 = SimpleColumnParallelLinear(in_dim, out_dim, tp_size=tp_size, gather_output=False)
parts = model2(x)
for i, p in enumerate(parts):
    print(f"GPU {i} 的部分输出 Y_{i}:", p.shape)