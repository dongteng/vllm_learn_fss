"""
@Author    : zhjm
@Time      : 2026/3/3 
@File      : demo.py
@Desc      : 
"""
import zmq, torch

tensor = torch.randn(1000,1000)
msg = zmq.Message(tensor.numpy().tobytes())
del tensor
print(msg)