"""
@Author    : zhjm
@Time      : 2026/2/2 
@File      : learn_simulate_moe_forward.py
@Desc      : 
"""
import torch
import torch.nn.functional as F
from typing import Tuple,List

class MockMoELayer:
    """简化的MOR层仿真实现"""
    def __init__(self,num_experts:int=8, top_k:int=2, hidden_size:int=1024, intermediate_size:int=2048,ep_size:int=2,ep_rank:int=0):
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.ep_size = ep_size
        self.ep_rank = ep_rank

        #计算专家映射：EP_SIZE=2 每个GPU负责4个专家
        self.local_num_experts = num_experts // ep_size
        self.expert_map = self._build_expert_map()

        #初始化权重（简化：使用随即权重）
