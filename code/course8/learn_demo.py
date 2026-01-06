"""
@Author    : zhjm
@Time      : 2026/1/6 
@File      : learn_demo.py
@Desc      : 
"""
import torch
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(
        X_ptr, #输入张量X的指针
        Y_ptr, #输入张量Y的指针
        Z_ptr, #输出张量Z的指针
        N,     #向量长度
        BLOCK_SIZE: tl.constexpr, #块大小，编译时常量 #这里为啥要是一个编译时常量？
):

    #1.计算当前block的起始位置和线程索引
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets= block_start + tl.arange(0,BLOCK_SIZE)
    mask = offsets<N


    #2.加载数据
    x = tl.load(X_ptr+offsets,mask=mask)
    y = tl.load(Y_ptr+offsets,mask=mask)

    z = x + y

    #4.存储结果
    tl.store(Z_ptr+offsets,z,mask=mask)




def vector_add_torch(x:torch.Tensor,y:torch.Tensor):
    """
    使用Triton 实现向量加法 （Z = X + Y）
    支持任意大小的1D张量
    """
    assert x.is_cuda and y.is_cuda, "输入张量必须在 GPU 上"
    assert x.shape == y.shape, "输入张量形状必须相同"
    N = x.numel()
    assert N>0, "张量不能为空"
    assert N > 0,'输入张量不能为空'

    #输出张量
    z= torch.empty_like(x)

    #定义块大小（通常为2的幂，例如1024）
    BLOCK_SIZE = 1024

    #计算需要多少个block
    num_blocks = triton.cdiv(N,BLOCK_SIZE) #triton.cdiv 向上取整函数

    #启动kernel
    vector_add_kernel[(num_blocks,)](
        X_ptr=x,
        Y_ptr=y,
        Z_ptr=z,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return z





if __name__ =="__main__":
    a = torch.randn(131072,device='cuda')
    b = torch.randn(131072,device='cuda')
    c_torch = a+b
    c_triton = vector_add_torch(a,b)
    print(torch.allclose(c_torch,c_triton))

    #验证是否一致
    print("max diff:", torch.max(torch.abs(c_torch-c_triton)))
    print("correct:",torch.allclose(c_torch,c_triton))