"""
@Author    : zhjm
@Time      : 2026/1/9 
@File      : learn_attention.py
@Desc      : 
"""
import math
import torch
import triton
import triton.language as tl
import torch.nn.functional as F
from torch.cuda.amp import custom_fwd


def standard_attention(Q, K,V, sm_scale, mask=None):
    """
    :param Q: [batch_size, seq_len, d_k]
    :param K: [batch_size, seq_len, d_k]
    :param V: [batch_size, seq_len, d_v]
    :param mask: [batch_size, seq_len]
    :return:
    """
    #计算QK^T
    attn_scores = torch.matmul(Q, K.transpose(-2,-1))*sm_scale
    if mask is not None:
        attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
    attn_weights = F.softmax(attn_scores, dim=-1)

    out = torch.matmul(attn_weights,V)

    return out


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
        num_kv_groups,  # group of kv heads
        n_heads,  # number of heads
        m_size,  # sequence length of q
        n_size,  # sequence length of k
        BLOCK_DHEAD_SIZE:tl.constexpt,
        BLOCK_M_SIZE:tl.constexpt, # BLOCK size of m_size dimension，即 Q 矩阵行数分成了(m_size // BLOCK_M_SIZE) 块，块大小是 BLOCK_M_SIZE
        BLOCK_N_SIZE:tl.constexpt, #K/V 的 seq_len 方向被 BLOCK_N_SIZE 分块，
        sm_scale,
        causal_mask
):
    """
    flashattention内核实现
    """
    block_m_idx = tl.program_id(0)
    head_idx = tl.program_id(1) #这个是 bs * n_heads 的索引。

    cur_batch_idx = head_idx // n_heads #看ppt图示
    cur_head_idx = head_idx % n_heads   #Q head在batch内的index

    #首先要知道num_kv_groups 是指一个 k v head被多少个Q head共享
    #就按照GQA来想   假设q.shape[1] = 8  k.shape[1] = 2，则num_kv_groups=4 ,
    cur_kv_head_idx = cur_head_idx // num_kv_groups  #这个 Q head 对应的 K/V head index

    m_range_offs = tl.arange(0,BLOCK_M_SIZE)
    n_range_offs = tl.arange(0,BLOCK_N_SIZE)
    dhead_range_offs = tl.arange(0,BLOCK_DHEAD_SIZE) #这个是 head_dim 维度

    m_offs = block_m_idx * BLOCK_M_SIZE + m_range_offs #这倒是容易理解啊 该线程块在Grid下负责的行索引

    #算当前block内元素的全局索引
    q_offs = (cur_batch_idx* q_batch_stride + cur_head_idx*q_heads_stride + m_offs*q_seq_stride
        +( m_offs[:,None]*q_seq_stride + dhead_range_offs*q_dim_stride)
              )
    k_offs = (
        cur_batch_idx * k_batch_stride + cur_kv_head_idx * k_heads_stride + (
            n_range_offs[:,None] * k_seq_stride + dhead_range_offs * k_dim_stride
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

    q_ptrs = q_ptr + q_offs
    k_ptrs = k_ptr + k_offs
    v_ptrs = v_ptr + v_offs
    o_ptrs = o_ptr + o_offs

    l_i  = tl.zeros((BLOCK_M_SIZE), dtype=tl.float32) - float('inf')
    d_i = tl.zeros((BLOCK_M_SIZE), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M_SIZE, BLOCK_DHEAD_SIZE), dtype=tl.float32)

    q_mask = m_offs[:,None] < m_size
    q = tl.load(q_ptrs, mask = q_mask, other=0.0)

    for block_n_start_idx in range(0,n_size,BLOCK_N_SIZE):
        block_n_offs = block_n_start_idx + n_range_offs
        k_mask = block_n_offs[:,None] < n_size
        #现在的k分块维度是多少？这里怎么还在变呢？上边不是已经挪动位置了？
        #k_ptr是
        k = tl.load(k_ptrs + block_n_start_idx*k_seq_stride, mask=k_mask, other=0.0)

        qk = tl.zeros((BLOCK_M_SIZE, BLOCK_N_SIZE), dtype=tl.float32)
        qk += tl.dot(q,tl.trans(k))

        if causal_mask:
            offs_k = block_n_offs
            offs_m = m_offs
            mask = offs_m[:,None] <= offs_k[None,:]
            qk = tl.where(mask, qk*sm_scale, -1.0e8)
        else:
            qk = qk * sm_scale

        l_j = tl.max(qk, 1)
        numerators = tl.exp(qk- l_j[:,None]) #这里写成tl.exp(qk- l_j)也行吧？会广播吧？
        d_j = tl.sum(numerators, 1)

        l_new = tl.maximum(l_i, l_j)
        alpha = tl.exp(l_i - l_new)
        beta = tl.exp(l_j - l_new)
        d_new = alpha*d_i + beta*d_j

        #compute softmax(qk)
        p_scale = beta / d_new
        p = numerators * p_scale[:,None]

        sigma = d_i / d_new *alpha
        acc = acc * sigma[:,None]

        v = tl.load(v_ptrs+ block_n_start_idx *v_seq_stride,mask=k_mask, other=0.0)
        p = p.tp(q_ptr.dtype.element_ty)

        acc += tl.dot(p, v)

        l_i = l_new
        d_i = d_new
    out_mask = m_offs[:,None] < m_size
    tl.store(o_ptrs, acc, mask=out_mask)


@torch.no_grad
@custom_fwd(cast_inputs=torch.float16) #custom_fwd 不是 PyTorch 原生 API，给 AMP（自动混合精度）用的 forward 装饰器 无论调用这个函数时输入是什么 dtype，在进入函数前，自动把输入 cast 成 float16
def flash_attention_v1(q:torch.tensor, k:torch.tensor, v:torch.tensor):
    """
    compute flash-attention ，can't support fp32 input
    参数；
        q: Query tensor , shape : [bs. n_heads, m_size, head_dim],decode 阶段，q的seq_len和k v 不一样，其值为1
        k: Key tensor , shape : [bs.  n_heads, n_size, head_dim]
        v: Value tensor , shape : [bs.  n_heads, m_size, head_dim]
        output: Attention output tensor, shape is consistent with q
        attention_mask:Attention mask matrix broadCastable to (bs, head_size, m_size, n_size)
    """
    num_kv_groups = q.shape[1]//k.shape[1] #num_q_heads // num_k_heads  不理解要这句干啥
    output = torch.empty_like(q)  #里边是未定义的垃圾值
    assert q.device.type == "cuda", "Input tensor q must be on CUDA device"
    assert k.device.type == "cuda", "Input tensor keys must be on CUDA device"

    assert q.shape[-1] == k.shape[-1] == v.shape[-1]
    assert (
        q.dtype == k.dtype == v.dtype == output.dtype
    ), f"All tensors must have the same dtype: {q.dtype}, {k.dtype}, {v.dtype}, {output.dtype}"

    bs, n_heads , m_size,HEAD_DIM = q.size()   #m_size 是 query 的长度，n_size 是 key 的长度
    causal_mask = False
    if m_size > 1:
        causal_mask  : bool = True

    n_size = k.shape[2]
    sm_scale = 1 / math.sqrt(HEAD_DIM)

    #定义网格
    grid = lambda meta:(triton.cdiv(m_size ,meta["BLOCK_M_SIZE"]), bs*n_heads,1)
    #该句的等价语法
    # def grid(meta):
    #     return (
    #         triton.cdiv(m_size, meta["BLOCK_M_SIZE"]),
    #         bs * n_heads,
    #         1,
    #     )  # 二维 grid
    flash_attention_v1_kernel[grid](
        q,
        k,
        v,
        output,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *output.stride(),
        num_kv_groups,
        n_heads,
        m_size, # sequence length of q
        n_size, # sequence length of k v
        HEAD_DIM,
        32, # BLOCK_M_SIZE
        32, # BLOCK_N_SIZE
        sm_scale,
        causal_mask)
    return output



def test_prefill_stage():
    # 设置测试参数
    batch_size = 2
    num_heads = 4
    seq_length = 32
    head_dim = 32
    BLOCK_M = 32
    BLOCK_N = 32

    #生成固定的输入张量 （使用固定随机种子以确保可重复性）
    torch.manual_seed(0)
    q = torch.randn(batch_size, seq_length, head_dim,device ='cuda',dtype=torch.float32)
    k = torch.randn(batch_size, seq_length, head_dim,device ='cuda',dtype=torch.float32)
    v = torch.randn(batch_size, seq_length, head_dim,device ='cuda',dtype=torch.float32)

    #计算softmax缩放因子
    sm_scale = 1.0 / math.sqrt(head_dim)

    #调用Triton内核
    out = flash_attention_v1(q,k,v)


    #使用标准PyTorch 实现计算注意力输出
    #创建下三角矩阵
    #(1,1,seq ,seq)
    mask = torch.tril(torch.ones((seq_length, seq_length), device='cuda')).unsqueeze(0).unsqueeze(0)

    standard_o = standard_attention(q,k,v,sm_scale,mask)

    #比较Triton内核输出 与 标准实现的输出
    if torch.allocate(out,standard_o,atol=1e-2):
        print("Test Passed! Triton output matches PyTorch standard implementation.")
    else:
        max_diff = (out-standard_o).abs().max()
        print(f"Test Failed! Triton output does not match PyTorch standard implementation.Maximum difference {max_diff}")





if __name__ == "__main__":
    print("Running Prefill Stage Test...")
    test_prefill_stage()
