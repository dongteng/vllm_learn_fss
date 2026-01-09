"""
@Author    : zhjm
@Time      : 2026/1/6 
@File      : learn_softmax.py
@Desc      : 
"""
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel_v1(
        input_ptr,
        output_ptr,
        input_row_stride,
        output_row_stride,
        n_cols,
        BLOCK_SIZE:tl.constexpr
):
    row_idx = tl.program_id(0)  #获取当前块在网格（grid）中的索引  也就是获取是当前第几个线程块
    row_start_ptr = input_ptr + row_idx * BLOCK_SIZE #这个就是每个线程块负责的区域的起始地址，输入同样是一维的
    col_offsets = tl.arange(0,BLOCK_SIZE)
    mask = col_offsets < n_cols


    input_ptr = row_start_ptr + col_offsets #负责的每行起点位置 +每个线程处理的位置 就是每个线程负责的位置
    tl.static_print(input_ptr)
    tl.device_print("input_ptr values:", input_ptr)
    row = tl.load(input_ptr, mask = mask, other= -float('inf')) #这里是每个线程拿数据 还是每行拿？这个row到底是行还是一个数据
    tl.static_print(row)
    tl.device_print("row values:", row)
    row_minus_max = row - tl.max(row,axis = 0)
    numerator  = tl.exp(row_minus_max)
    denominator = tl.sum(numerator,axis = 0)
    softmax_output = numerator / denominator

    out_row_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = out_row_ptr + col_offsets
    tl.store(output_ptrs, softmax_output,mask = mask)

def triton_softmax_v1(x):
    n_rows , n_cols = x.shape
    y = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n_cols) #返回大于或等于给定数字的、最接近的 2 的整数次幂（power of 2）。

    num_warps = 4
    if BLOCK_SIZE >=2048:
        num_warps = 8
    if BLOCK_SIZE >=4096:
        num_warps = 16

    #有多少行就启用多少个线程块，每个线程块负责一行
    softmax_kernel_v1[(n_rows,)](
        x, y,
        x.stride(0), y.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return y

def naive_softmax(x):
    """
    计算输入张量的softmax值
    """
    x_max = x.max(dim=1, keepdim=True)[0]
    z = x - x_max
    numerator = torch.exp(z)
    deniminator = numerator.sum(dim=1, keepdim=True)
    return numerator / deniminator



if __name__ == '__main__':
    torch.manual_seed(42)

    shapes = [
        (8,512),
        (3,123),
        (17,54),
        (1000,256),
        (2048,2048),
    ]
    for shape in shapes:
        print(f"{'='*60}")
        print(f"Testing shape: {shape}")
        print(f"{'='*60}")
        x = torch.randn(*shape, device='cuda')

        y_torch = naive_softmax(x)
        y_triton_v1 = triton_softmax_v1(x)

        y_builtin = torch.softmax(x, dim=1)
        atol = 1e-6;
        print(f"correctness check")
        print(f" torch vs builtin:{torch.allclose(y_torch, y_builtin, atol=atol)}")
        print(f" Triton v1 vs builtin:{torch.allclose(y_triton_v1, y_builtin, atol=atol)}")

        sample_row = 0
        print(f"\n>>> Sample Output Row {sample_row} (first 5 elements):")
        print(f"  Torch:   {y_torch[sample_row, :5].cpu().numpy()}")
        print(f"  TritonV1:{y_triton_v1[sample_row, :5].cpu().numpy()}")
    try:
        from triton.testing import do_bench

        print(f"\n{'='*60}")
        print("⏱️  Performance Benchmark (shape=1024x1024)")
        print('=' * 60)

        x_large = torch.randn(8192,8192,device='cuda')

        ms_torch = do_bench(lambda : naive_softmax(x_large))
        ms_v1 = do_bench(lambda:triton_softmax_v1(x_large))

        print(f"PyTorch:   {ms_torch:.3f} ms")
        print(f"Triton V1: {ms_v1:.3f} ms")
        print(f"Speedup V1 over Torch: {ms_torch/ms_v1:.2f}x")

    except Exception as e:
        print(f"Benchmark skipped: {e}")


