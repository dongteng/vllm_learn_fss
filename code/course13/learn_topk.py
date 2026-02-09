"""
@Author    : zhjm
@Time      : 2026/2/1 
@File      : learn_topk.py
@Desc      : 
"""
import torch
import torch.nn.functional as F

#输入router_logits张量（来自gate层输出）#形状：[num_tokens,num_experts]
router_logits = torch.tensor([
    [0.1, 0.2, 0.5, 0.1, 0.05, 0.03, 0.01, 0.01],  # token 0
    [0.3, 0.4, 0.1, 0.1, 0.05, 0.03, 0.01, 0.01],  # token 1
    [0.05, 0.1, 0.1, 0.1, 0.3, 0.25, 0.05, 0.05],  # token 2
    [0.2, 0.1, 0.4, 0.2, 0.05, 0.03, 0.01, 0.01],  # token 3
])

# 1. Top-k 选择
top_k = 2
topk_logits, topk_ids = torch.topk(router_logits, k=top_k, dim=-1)

#2. 权重重新归一化
topk_weights = F.softmax(topk_logits, dim=-1, dtype=torch.float32)

# --- 输出结果 (数学上一致) ---
# topk_weights 是对 topk_logits 应用 softmax 的结果
# tensor([[0.5744, 0.4256],  <-- softmax([0.5, 0.2])
#         [0.5250, 0.4750],  <-- softmax([0.4, 0.3])
#         [0.5125, 0.4875],  <-- softmax([0.3, 0.25])
#         [0.5498, 0.4502]]) <-- softmax([0.4, 0.2])
print("topk_weights:\n", topk_weights)

# topk_ids 是 router_logits 中最高分的专家索引
# tensor([[2, 1],  <-- 专家 2 (0.5), 专家 1 (0.2)
#         [1, 0],  <-- 专家 1 (0.4), 专家 0 (0.3)
#         [4, 5],  <-- 专家 4 (0.3), 专家 5 (0.25)
#         [2, 0]]) <-- 专家 2 (0.4), 专家 0 (0.2)
# 注意：在 token 3 中，有两个分数为 0.2 的专家 (索引 0 和 3)。
# torch.topk 在值相同时的行为是稳定的，但可能不保证返回哪个索引。
# 在此例中，它返回了索引 0。
print("\ntopk_ids:\n", topk_ids)