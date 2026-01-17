# https://github.com/ModelTC/lightllm/blob/main/lightllm/models/llama/triton_kernel/context_flashattention_nopad.py
# https://github.com/ELS-RD/kernl/blob/main/src/kernl/implementations/attention.py#L438

import torch, math
import triton
import triton.language as tl
from torch.cuda.amp import custom_fwd
from typing import List, Optional, Union
import torch.nn.functional as F


# TODO: integrating rope with flash-attn
@triton.jit
def flash_attention_v1_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    q_batch_stride,
    q_heads_stride,
    q_seq_stride,
    q_dim_stride,
    k_batch_stride,
    k_heads_stride,
    k_seq_stride,
    k_dim_stride,  # matrix Q stride for columns, [seq_len, head_dim]
    v_batch_stride,
    v_heads_stride,
    v_seq_stride,
    v_dim_stride,
    out_batch_stride,
    out_heads_stride,
    out_seq_stride,
    out_dim_stride,
    num_kv_groups,  # group of kv heads 多少个q head 共享同一个 k v head
    n_heads,  # number of heads
    m_size,
    n_size,  # sequence length of k, also be rows of K matrix
    BLOCK_DHEAD_SIZE: tl.constexpr,  # head_dim dimension
    BLOCK_M_SIZE: tl.constexpr,  # BLOCK size of m_size dimension，即 Q 矩阵行数分成了(m_size // BLOCK_M_SIZE) 块，块大小是 BLOCK_M_SIZE
    BLOCK_N_SIZE: tl.constexpr,  # n_size dimension
    sm_scale,
    causal_mask,
):
    """
    flashattention 内核实现
    """
    block_m_idx = tl.program_id(0) #获取当前正在执行的block的索引 axis=-表示拿网格第0维的索引；表示当前这个kernel block正在处理Q的第几个M-tile(第几块查询token)
    head_idx = tl.program_id(1) #当前block正在处理的全局head索引（从0开始到batch_size* heads的总数）

    cur_batch_idx = head_idx // n_heads
    cur_head_idx = head_idx % n_heads

    cur_kv_head_idx = cur_head_idx // num_kv_groups #意思是当前query head对应到哪个kv head

    m_range_offs = tl.arange(0, BLOCK_M_SIZE)  #当前block内，要处理的 Q的行偏移
    n_range_offs = tl.arange(0, BLOCK_N_SIZE)  #当前block内，要处理的 K V的行偏移
    dhead_range_offs = tl.arange(0, BLOCK_DHEAD_SIZE) #这才是真正的head维度的偏移

    m_offs = block_m_idx * BLOCK_M_SIZE + m_range_offs #计算当前block正在处理的Q矩阵的全局行号

    # Compute offsets for the first block on matrix Q K V Output
    #计算 当前block 要从q内存中读取的全部元素的起始地址偏移量  就是BLOCK_M_SIZE * BLOCK_DHEAD_SIZE 这么大小的块
    #这里其实是定位的问题，先找到是哪个批次，再找到哪个head，最后就是用行号+列号
    #我咋感觉这个 dhead_range_offs定位不到呢？ 比如说dim0 dim1  dim2 dim3这里也区分不出来吧？ 难道说这里是triton并行做的？ 是我想错了第二维度就是head啊

    #前边已经有行号m_offs了，为什么还要q_offs？
    # 这一整行在做的事是：
    # 把上面得到的行号（m_offs）和维度号（dhead_range_offs），
    # 结合batch / head / stride
    # 信息，转换成内存里每个元素的真实偏移量（字节数）。
    # 最终q_offs的形状是：
    # [BLOCK_M_SIZE, BLOCK_DHEAD_SIZE]比如[128, 128]
    # 里面的每一个元素都是一个数字，表示：
    # 从q_ptr这个基地址开始，要偏移多少个元素（或字节）才能到达对应的Q值
    q_offs = (
        cur_batch_idx * q_batch_stride
        + cur_head_idx * q_heads_stride
        + (m_offs[:, None] * q_seq_stride + dhead_range_offs[None, :] * q_dim_stride) #这里最后其实没必要*q_dim_stride
    )
    #为什么q_offs用的m_offs? 为什么k_offs用的n_range_offs?
    #取的是哪些 key token（行）,这些 key token 的位置信息由 n_range_offs + block_n_start_idx 决定。
    k_offs = (
        cur_batch_idx * k_batch_stride
        + cur_kv_head_idx * k_heads_stride
        + (
            n_range_offs[:, None] * k_seq_stride        #这决定行
            + dhead_range_offs[None, :] * k_dim_stride  #这决定列
        )
    )

    v_offs = (
        cur_batch_idx * v_batch_stride
        + cur_kv_head_idx * v_heads_stride
        + (
            n_range_offs[:, None] * v_seq_stride
            + dhead_range_offs[None, :] * v_dim_stride
        )
    )

    o_offs = (
        cur_batch_idx * out_batch_stride
        + cur_head_idx * out_heads_stride
        + (
            m_offs[:, None] * out_seq_stride
            + dhead_range_offs[None, :] * out_dim_stride
        )
    )
    #这里不理解，上边不是定位到了吗？为什么还加上q_ptr?因为q_offs只是“相对偏移量”（一个数字矩阵），它告诉你“要从起点跳多少步”
    #q_ptr 才是真正的“起点地址”
    q_ptrs = q_ptr + q_offs
    k_ptrs = k_ptr + k_offs #[BLOCK_N_SIZE, BLOCK_DHEAD_SIZE]，但里面的每个值现在都是真正的内存指针（绝对地址），可以直接用来从全局内存读取数据。
    v_ptrs = v_ptr + v_offs
    out_ptrs = o_ptr + o_offs

    # 初始化用于计算 softmax 归一化项的 m 和 d, 意义见 online-softmax, 这里
    l_i = tl.zeros((BLOCK_M_SIZE,), dtype=tl.float32) - float("inf") #记录qk 每一行的最大值 不是 [BLOCK_M_SIZE, 1]，而是单纯的 [BLOCK_M_SIZE]（一维向量）
    d_i = tl.zeros((BLOCK_M_SIZE,), dtype=tl.float32)                #记录qk 每一行的累积（exp(qk-max)）
    acc = tl.zeros((BLOCK_M_SIZE, BLOCK_DHEAD_SIZE), dtype=tl.float32) #输出O的累加器 最终结果 这个数据类型一般是多少？

    #加载Q分块 ，这里其实有隐藏的Triton并行在这
    q_mask = m_offs[:, None] < m_size
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # 逐块加载 K V，这里其实可以看PPT的abc分块 这个遍历主要是遍历蓝色那个竖条，我突然好奇 为什么是第二列？为啥不是第3列？
    for block_n_start_idx in range(0, n_size, BLOCK_N_SIZE):
        block_n_offs = block_n_start_idx + n_range_offs #当前block正在处理的K V矩阵的行偏移
        k_mask = block_n_offs[:, None] < n_size

        #不理解这块加载了多大啊？
        #首先要知道k_ptrs = k_ptr + k_offs ，k_offs 是形状为 [BLOCK_N_SIZE, BLOCK_DHEAD_SIZE]
        #这个block_n_offs能定义加载哪几行，k分块的哪几列是哪里决定的？比如说我这个[2,2,6,4]的K矩阵,  已知q分块为1*1，K分块是2*1,但是K的列定位是哪里知道的？
        #K 本身没有沿着 d_head 分块，每次加载的 K 小块在维度上是完整的。令震惊了哦！
        #那我就不理解了 k_ptrs是一整块[BLOCK_N_SIZE, BLOCK_DHEAD_SIZE]，加上block_n_start_idx * k_seq_stride是什么意思？那岂不是还是[BLOCK_N_SIZE, BLOCK_DHEAD_SIZE]
        #不理解k_ptrs 与 blcok_n_offs的区别。
        #这里为什么不是tl.load(k_ptrs) 后边的 block_n_start_idx * k_seq_stride 是在干什么？我理解k_ptrs已经定位了，我擦！k_ptrs是第一块的定位
        k = tl.load(k_ptrs + block_n_start_idx * k_seq_stride, mask=k_mask, other=0.0)

        qk = tl.zeros((BLOCK_M_SIZE, BLOCK_N_SIZE), dtype=tl.float32) #为啥要设定一个形状先
        qk += tl.dot(q, tl.trans(k)) #为啥不能直接有这一句？在 Triton 中，tl.dot 本身不会自动创建输出缓冲区，它只是执行矩阵乘法运算，并把结果写到你已经准备好的目标张量里。

        # 应用因果遮罩
        if causal_mask:
            offs_k = block_n_offs
            offs_m = m_offs
            # casual 模型的 causal mask 下三角矩阵
            # 这样整成二维矩阵了
            mask = offs_m[:, None] >= offs_k[None, :]  #这个比较也会发生广播的，行号大于 K的行号的 都会被mask掉
            # mask = offs_m[:, None] < offs_k[None, :]
            qk = tl.where(mask, qk * sm_scale, -1.0e8) #where的作用是 保留mask为True的，其余的全体换为-1.0e8
        else:
            qk = qk * sm_scale

        #更新前要明确三个变量
        # l_i :到目前为止所有处理过的块中，全局最大logit（之前所有qk的最大值）
        # d_i ：到目前为止，所有已处理的归一化分母累积和（相当于 Σ exp(qk - l_i) 的总和，经过缩放调整）
        # acc: 到目前为止累积的输出向量,acc = Σ [exp(qk - l_i) / d_i] · V   ← 这是我们最想得到的最终形式


        l_j = tl.max(qk, 1)        #找到本块注意力矩阵 每行的最大值，后边居然是个1？ cao  是纬度值
        # numerators = tl.exp(qk - l_j) #这里为什么不能写成这样  难道不是有自动广播计算吗？ 此处看笔记
        numerators = tl.exp(qk - l_j[:, None])  #当前块 注意力矩阵的 exp值,  numerators_{i,j} = exp( (q_i · k_j) - l_j )
        d_j = tl.sum(numerators, 1)  # 1d vector  当前块每行的exp之和（局部分母和）,d_j = Σ_j exp( (q_i · k_j) - l_j )

        #现在进入最关键部分，在线合并softmax
        l_new = tl.maximum(l_i, l_j)     #到目前为止 每行的全局最大和。l_i之前所有block的全局最大值，l_j 当前block 注意力的最大值
        alpha = tl.exp(l_i - l_new)      #历史最大值 相对于 新最大值的缩放因子
        beta = tl.exp(l_j - l_new)       #当前块相对于 新最大值的缩放因子，如果新最大值就是本块产生的，那么beta就是1
        d_new = alpha * d_i + beta * d_j ## 新总分母 = 历史分母×缩放 + 当前分母×缩放

        # compute softmax(qk)
        p_scale = beta / d_new           #这里为啥用beta   直接下一行不行？ 哦 它是想统一分子，到目前为止numerators还没放缩呢
        p = numerators * p_scale[:, None] #把当前小块的局部exp值，缩放成真正意义上的softmax值
        # acc scaling
        sigma = d_i / d_new * alpha      #对之前整体acc进行缩放， 这里acc就是O
        acc = acc * sigma[:, None]       #acc = Σ [ exp(qk - l_i) / d_i ] · V

        # compute O = PV
        v = tl.load(v_ptrs + block_n_start_idx * v_seq_stride, mask=k_mask, other=0.0)
        p = p.to(q_ptr.dtype.element_ty) #把注意力概率p转换成和Q相同得到数据类型

        acc += tl.dot(p, v)

        # update the normalizer (l and d) for next iteration
        l_i = l_new
        d_i = d_new

    out_mask = m_offs[:, None] < m_size
    tl.store(out_ptrs, acc, mask=out_mask)#当前结果写回全局内存中的位置


@torch.no_grad()
@custom_fwd(cast_inputs=torch.float16)
def flash_attention_v1(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
):
    """Compute Flash-attention, can't support fp32 input
    参数:
        q: Query tensor, shape: [bs, n_heads, m_size, head_dim], decode 阶段, q 的 seq_len 和 k v 不一致, 其值为 1
        k: Key tensor,  shape: [bs, n_heads, n_size, head_dim].
        v: Value tensor, shape is consistent with k.
        output: Attention ouput tensor, shape is consistent with q.
        attention_mask: Attention mask matrix broadcastable to (batch, head_size, m_size, n_size).
    """
    num_kv_groups = q.shape[1] // k.shape[1]  # num_q_heads // num_k_heads
    output = torch.empty_like(q)
    assert q.device.type == "cuda", "Input tensor q must be on CUDA device"
    assert k.device.type == "cuda", "Input tensor keys must be on CUDA device"

    assert q.shape[-1] == k.shape[-1] == v.shape[-1]
    assert (
        q.dtype == k.dtype == v.dtype == output.dtype
    ), f"All tensors must have the same dtype: {q.dtype}, {k.dtype}, {v.dtype}, {output.dtype}"

    # sequence length of q, also be rows of Q matrix
    bs, n_heads, m_size, HEAD_DIM = q.size()
    causal_mask = False
    if m_size > 1:
        causal_mask: bool = True

    n_size = k.shape[2]
    sm_scale = 1 / math.sqrt(HEAD_DIM)
    # BLOCK_M_SIZE = 128
    grid = lambda meta: (
        triton.cdiv(m_size, meta["BLOCK_M_SIZE"]),
        bs * n_heads,
        1,
    )  # 二维 grid

    flash_attention_v1_kernel[grid](
        q,
        k,
        v,
        output,
        *q.stride(),  # (batch, heads, m_size, head_dim)
        *k.stride(),  # (batch, heads, n_size, head_dim)
        *v.stride(),  # (batch, heads, n_size, head_dim)
        *output.stride(),  # (batch, heads, m_size, n_size)
        num_kv_groups,
        n_heads,
        m_size,
        n_size,
        HEAD_DIM,
        32,  # BLOCK_M_SIZE
        32,  # BLOCK_N_SIZE
        sm_scale,
        causal_mask,
    )
    return output


def standard_attention(Q, K, V, sm_scale, mask=None):
    """
    标准的 PyTorch 实现的自注意力机制。

    Args:
        Q (torch.Tensor): 查询张量，形状 (batch_size, num_heads, seq_length, head_dim)
        K (torch.Tensor): 键张量，形状 (batch_size, num_heads, seq_length, head_dim)
        V (torch.Tensor): 值张量，形状 (batch_size, num_heads, seq_length, head_dim)
        sm_scale (float): Softmax 缩放因子
        mask (torch.Tensor, optional): 遮罩张量，形状 (batch_size, num_heads, seq_length, seq_length)

    Returns:
        torch.Tensor: 注意力输出，形状与 Q 相同
    """
    # 计算 QK^T
    attn_scores = (
        torch.matmul(Q, K.transpose(-2, -1)) * sm_scale
    )  # (batch_size, num_heads, seq_length, seq_length)

    if mask is not None:
        attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

    # print("attn_scores", attn_scores)
    attn_weights = F.softmax(attn_scores, dim=-1)

    # 计算注意力输出
    out = torch.matmul(attn_weights, V)  # (batch_size, num_heads, seq_length, head_dim)

    return out


def test_prefill_stage():
    # 设置测试参数
    batch_size = 2
    num_heads = 4
    seq_length = 32
    head_dim = 32
    BLOCK_M = 32
    BLOCK_N = 32

    # 生成固定的输入张量（使用固定随机种子以确保可重复性）
    torch.manual_seed(0)
    q = torch.randn(
        batch_size, num_heads, seq_length, head_dim, device="cuda", dtype=torch.float32
    )
    k = torch.randn(
        batch_size, num_heads, seq_length, head_dim, device="cuda", dtype=torch.float32
    )
    v = torch.randn(
        batch_size, num_heads, seq_length, head_dim, device="cuda", dtype=torch.float32
    )

    # 计算 Softmax 缩放因子
    sm_scale = 1.0 / math.sqrt(head_dim)  # 1 / sqrt(d_k) * 1/log(2)

    # 调用 Triton 内核
    out = flash_attention_v1(q, k, v)

    # 使用标准 PyTorch 实现计算注意力输出
    # 创建下三角矩阵
    mask = (
        torch.tril(torch.ones((seq_length, seq_length)))
        .unsqueeze(0)
        .unsqueeze(0)
        .type_as(q)
    )  # (1, 1, seq, seq)
    standard_o = standard_attention(q, k, v, sm_scale, mask)

    # 比较 Triton 内核输出与标准实现的输出
    if torch.allclose(out, standard_o, atol=1e-2):
        print(
            "Prefill Stage Test Passed: Triton output matches PyTorch standard implementation."
        )
    else:
        max_diff = (out - standard_o).abs().max()
        print(f"Prefill Stage Test Failed: Maximum difference {max_diff}")
        # 可选择打印更多信息进行调试


def test_decode_stage():
    # 设置测试参数
    batch_size = 1
    num_heads = 4
    initial_seq_length = 16
    generated_seq_length = 16
    head_dim = 64
    BLOCK_M = 16
    BLOCK_N = 16

    # 生成固定的初始输入张量
    torch.manual_seed(0)
    q_initial = torch.randn(
        batch_size,
        num_heads,
        initial_seq_length,
        head_dim,
        device="cuda",
        dtype=torch.float32,
    )
    k_initial = torch.randn(
        batch_size,
        num_heads,
        initial_seq_length,
        head_dim,
        device="cuda",
        dtype=torch.float32,
    )
    v_initial = torch.randn(
        batch_size,
        num_heads,
        initial_seq_length,
        head_dim,
        device="cuda",
        dtype=torch.float32,
    )
    o_initial = torch.zeros_like(q_initial, device="cuda", dtype=torch.float32)
    new_token_q = torch.randn(
        batch_size, num_heads, 1, head_dim, device="cuda", dtype=torch.float32
    )

    triton_k_extended = k_initial
    triton_v_extended = v_initial
    torch_k_extended = k_initial
    torch_v_extended = v_initial
    torch_new_token_q = new_token_q
    triton_new_token_q = new_token_q
    # 模拟生成过程中逐步增加序列长度
    for step in range(1, generated_seq_length + 1):
        # 生成新的 token
        triton_k_extended = torch.cat([triton_k_extended, triton_new_token_q], dim=2)
        triton_v_extended = torch.cat([triton_v_extended, triton_new_token_q], dim=2)

        torch_k_extended = torch.cat([torch_k_extended, torch_new_token_q], dim=2)
        torch_v_extended = torch.cat([torch_v_extended, torch_new_token_q], dim=2)

        # 扩展 Q, K, V 和 Out
        # q_extended = torch.cat([q_initial, new_token_q], dim=2)

        # 计算 Softmax 缩放因子, sm_scale * 1.4426950408889634 精度可控制在 1e-2 内
        sm_scale_extended = 1.0 / math.sqrt(head_dim)

        # 计算 Triton 内核输出
        triton_new_token_q = flash_attention_v1(
            new_token_q, triton_k_extended, triton_v_extended
        )

        # 使用标准 PyTorch 实现计算扩展后的注意力输出
        torch_new_token_q = standard_attention(
            new_token_q, torch_k_extended, torch_v_extended, sm_scale_extended
        )

        # 比较 Triton 内核输出与标准实现的输出
        if torch.allclose(triton_new_token_q, torch_new_token_q, atol=1e-1):
            print(
                f"Decode Stage Step {step} Test Passed: Triton output matches PyTorch standard implementation."
            )
        else:
            max_diff = (triton_new_token_q - torch_new_token_q).abs().max()
            print(
                f"Decode Stage Step {step} Test Failed: Maximum difference {max_diff}"
            )
            # 可选择打印更多信息进行调试
            break  # 根据需要是否停止测试


if __name__ == "__main__":
    print("Running Prefill Stage Test...")
    test_prefill_stage()
    print("\nRunning Decode Stage Test...")
    test_decode_stage()

"""
Running Prefill Stage Test...
Prefill Stage Test Passed: Triton output matches PyTorch standard implementation.

Running Decode Stage Test...
Decode Stage Step 1 Test Passed: Triton output matches PyTorch standard implementation.
Decode Stage Step 2 Test Passed: Triton output matches PyTorch standard implementation.
Decode Stage Step 3 Test Passed: Triton output matches PyTorch standard implementation.
Decode Stage Step 4 Test Passed: Triton output matches PyTorch standard implementation.
Decode Stage Step 5 Test Passed: Triton output matches PyTorch standard implementation.
Decode Stage Step 6 Test Passed: Triton output matches PyTorch standard implementation.
Decode Stage Step 7 Test Passed: Triton output matches PyTorch standard implementation.
Decode Stage Step 8 Test Passed: Triton output matches PyTorch standard implementation.
Decode Stage Step 9 Test Passed: Triton output matches PyTorch standard implementation.
Decode Stage Step 10 Test Passed: Triton output matches PyTorch standard implementation.
Decode Stage Step 11 Test Passed: Triton output matches PyTorch standard implementation.
Decode Stage Step 12 Test Passed: Triton output matches PyTorch standard implementation.
Decode Stage Step 13 Test Passed: Triton output matches PyTorch standard implementation.
Decode Stage Step 14 Test Passed: Triton output matches PyTorch standard implementation.
Decode Stage Step 15 Test Passed: Triton output matches PyTorch standard implementation.
Decode Stage Step 16 Test Passed: Triton output matches PyTorch standard implementation.
"""