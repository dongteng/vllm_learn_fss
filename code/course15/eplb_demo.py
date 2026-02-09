"""
@Author    : zhjm
@Time      : 2026/2/3 
@File      : eplb_demo.py
@Desc      : 
"""
import torch
# import eplb
from eplb_np import rebalance_experts
from visual_tool import reshape_map,visualize_ep_inputs,visualize_4d_array
#weight形状[2,12]表示模型有2层moe,每层12个专家
#数值越大，说明这个专家在历史推理中被选中的次数（或 token 量）越多 → 越“热”
weight = torch.tensor([[ 90, 132,  40,  61, 104, 165,  39,   4,  73,  56, 183,  86],
                       [ 20, 107, 104,  64,  19, 197, 187, 157, 172,  86,  16,  27]])

num_replicas = 16 #总共创建16个专家副本 原12个+额外复制4个
num_groups = 4    #专家分组数，通常等于节点数
num_nodes = 2     #集群有2个物理节点
num_gpus = 8      #总gpu数 = 2节点 * 每节点4张卡

#调用核心函数 rebalance_experts，它会做两件主要事：1决定哪些专家被复制
#把所有副本（16个）均匀放到8张gpu上，同事尽量满足分组约束
#phy2log ：形状[num_layers,num_gpus]
phy2log, log2phy, logcnt = rebalance_experts(weight, num_replicas, num_groups, num_nodes, num_gpus)
# print(f"===phy2log:{phy2log}")
# print(f"===log2phy:{log2phy}")
# print(f"===logcnt:{logcnt}")
np_phy2log = reshape_map(phy2log,num_nodes,num_gpus)
visualize_ep_inputs(weight)
visualize_4d_array(np_phy2log)