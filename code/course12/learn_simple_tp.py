"""
@Author    : zhjm
@Time      : 2026/1/26 
@File      : learn_simple_tp.py
@Desc      : 
"""
import numpy as np
def gelu(x):
    return 0.5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+0.044715*np.power(x,3))))

#1. 初始化参数
np.random.seed(42)
B = 4   # Batch size
H = 8   # Hidden dimension (输入维度)
D = 4   # Output dimension (输出维度)

# 创建输入矩阵 X (B, H) 和 权重矩阵 A (H, D)
X = np.random.randn(B, H)
A = np.random.randn(H, D)

#方法1：
#直接计算 Y = GeLU(X @ A)

target_output = gelu(np.dot(X, A))
print(f"基准输出形状: {target_output.shape}")


