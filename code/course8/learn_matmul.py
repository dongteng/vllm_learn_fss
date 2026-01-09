"""
@Author    : zhjm
@Time      : 2026/1/7 
@File      : learn_matmul.py
@Desc      : 
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _fused_linear_kernel_fwd(
        x_ptr,#输入数据矩阵首元素指针
        w_ptr,#权重矩阵首元素指针
        z_ptr,#输出结果地址
        M,N,K,
        BLOCK_SIZE_M: tl.constexpr =128,
        BLOCK_SIZE_N: tl.constexpr =128,
        BLOCK_SIZE_K: tl.constexpr =64,
):
    #首先 拿到一个块在网格中的索引
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    #一个triton block的处理范围（M,N轴上）
    offs_m = pid_m* BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:,None]
    offs_n = pid_n* BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None,:]

    z = tl.zeros((BLOCK_SIZE_M,BLOCK_SIZE_N),dtype=tl.float32)
    for k in range(0,K , BLOCK_SIZE_K):
        x_k = tl.arange(0,BLOCK_SIZE_K)[None,:] + k
        x =  tl.load(x_ptr+ offs_m*K  + x_k,mask = offs_m< M & x_k < K, other = 0.0)
        x= x.to(tl.float16)

        w_k = tl.arange(0,BLOCK_SIZE_K)[:,None] + k

        w = tl.load(w_ptr+ w_k*N +offs_n,mask= w_k < K & offs_n < N,other = 0.0)
        w= w.to(tl.float16)

        z= tl.dot(x,w,acc=z)


@torch.no_grad()
def fused_ffn(x ,weight):
    out_shape_0 = x.shape[:-1]
    x = x.view((-1,x.shape[-1]))
    M,K = x.shape
    N = weight.shape[-1]

    #allocate output
    z= torch.empty((M,N),device =x.device,dtype =x.dtype)

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32

    #2d launch kernel where each block geits its own program
    grid = (triton.cdiv(M,BLOCK_SIZE_M),triton.cdiv(N,BLOCK_SIZE_N),1)

    _fused_linear_kernel_fwd[grid](
        x,
        weight,
        z,
        M,N,K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return z.view((*out_shape_0, N))





if  __name__ == "__main__":
    batch_size = 64
    sequence_length = 128
    hidden_dim = 1280

    #假设权重矩阵weight的形状为[hidden_dim, output_dim]
    output_dim = 2560

    x = torch.randn((batch_size ,sequence_length, hidden_dim),device='cuda',dtype = torch.float16)
    weight = torch.randn(hidden_dim, output_dim, device='cuda',dtype = torch.float16)

    #warmup
    for i in range(5):
        golden = x @ weight
        output = fused_ffn(x, weight)
        x = torch.randn((batch_size ,sequence_length, hidden_dim),device='cuda',dtype = torch.float16)
        weight = torch.randn(hidden_dim, output_dim, device='cuda',dtype = torch.float16)

    repeat_time = 5
    import time
    times_torch = []
    times_triton = []
    for i in range(repeat_time):
        x = torch.randn((batch_size ,sequence_length, hidden_dim),device='cuda',dtype = torch.float16)
        weight = torch.randn(hidden_dim, output_dim, device='cuda',dtype = torch.float16)
        torch.cuda.synchronize() #平时别用 测试和排错用，让 CPU 等待，直到当前 GPU 上已经提交的所有 CUDA 操作全部执行完成。

        t1 = time.time()
        output = fused_ffn(x, weight)
        torch.cuda.synchronize()
        t2 = time.time()
        print('triton time:{}'.format(t2 - t1))
        times_triton.append(t2 - t1)

        t1 = time.time()
        golden = x @ weight
        torch.cuda.synchronize()
        t2 = time.time()
        times_torch.append(t2 - t1)
        print('pytorch time:{}'.format(t2 - t1))

    import matplotlib.pyplot as plt

    # 将时间从秒转换为毫秒
    times_torch_ms = [t * 1000 for t in times_torch]
    times_triton_ms = [t * 1000 for t in times_triton]

    sizes = [i for i in range(repeat_time)]
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times_torch_ms, label='torch (matrix_multiply)', marker='o')
    plt.plot(sizes, times_triton_ms, label='triton (matrix_multiply)', marker='o')

    plt.xlabel('Run Index')
    plt.ylabel('Time (milliseconds)')
    plt.title('Matrix Multiplication Performance Comparison (Torch vs Triton)')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('cc.png')
    #
