import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
@triton.jit
def store_kvcache_kernel(
    key_ptr,            # 输入 K 张量的指针
    key_stride,         # K 张量在 token 维度的步长
    value_ptr,          # 输入 V 张量的指针
    value_stride,       # V 张量在 token 维度的步长
    k_cache_ptr,        # K Cache 张量的指针
    v_cache_ptr,        # V Cache 张量的指针
    slot_mapping_ptr,   # slot_mapping 的指针
    D: tl.constexpr,    # 每个 token 的 KV 维度 (num_heads * head_dim)
):
    """
    Triton Kernel：将 K、V 向量写入 KV Cache 的指定槽位。

    为什么用 Triton 而非 PyTorch：
    1. slot_mapping 指定的位置不连续，PyTorch 索引操作效率低
    2. Triton 可以并行处理所有 token，每个 token 一个线程块
    3. 合并读写，减少显存带宽压力
    """
    # 当前处理的 token 索引（每个线程块处理一个 token）
    idx = tl.program_id(0)

    # 获取目标槽位 从slot_mapping索引表里面拿出token的物理位置索引
    slot = tl.load(slot_mapping_ptr + idx)

    # slot = -1 是 CUDA Graph 填充的无效位置，跳过
    if slot == -1:
        return

    # 从输入张量加载 K 和 V    取出当前 token 的地址
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    #将key_offsets和value_offsets地址的值加载到key和value中
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)

    # slot*D 是当前 token 在 Cache 中的起始地址
    cache_offsets = slot * D + tl.arange(0, D)
    #将k和v存入k_cache和v_cache中
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor,
                  k_cache: torch.Tensor, v_cache: torch.Tensor,
                  slot_mapping: torch.Tensor):
    """
    Python 封装：调用 Triton Kernel 写入 KV Cache。

    Args:
        key: 当前计算的 K，形状 [N, num_heads, head_dim]
        value: 当前计算的 V，形状 [N, num_heads, head_dim]
        k_cache: K Cache，形状 [num_blocks, block_size, num_heads, head_dim]
        v_cache: V Cache，形状同上
        slot_mapping: 槽位映射，形状 [N]
    """
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    # 验证张量布局
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    # 启动 Kernel，每个 token 一个线程块
    store_kvcache_kernel[(N,)](
        key, key.stride(0),
        value, value.stride(0),
        k_cache, v_cache,
        slot_mapping, D
    )


class Attention(nn.Module):
    """
    Attention 层，集成 KV Cache 的读写和 Attention 计算。

    支持两种模式：
    1. Prefill: 使用 flash_attn_varlen_func 处理变长序列
    2. Decode: 使用 flash_attn_with_kvcache 处理单 token
    """

    def __init__(self, num_heads, head_dim, scale, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        # k_cache 和 v_cache 在 ModelRunner.allocate_kv_cache 中绑定
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        执行 Attention 计算。

        Args:
            q: Query，形状 [N, num_heads, head_dim]
            k: Key，形状 [N, num_kv_heads, head_dim]
            v: Value，形状同上

        Returns:
            输出，形状同 q
        """
        # 获取当前上下文
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        # 1. 将当前计算的 K、V 写入 Cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        # 2. 根据阶段选择 Attention 计算方式
        if context.is_prefill:
            # Prefill 阶段
            if context.block_tables is not None:
                # Prefix Cache 命中：从 Cache 读取历史 KV
                # 传入的 k, v 只包含新计算的 token
                # flash_attn 内部会根据 block_table 拼接历史 KV
                k, v = k_cache, v_cache
            #在不浪费任何显存（不补零）的前提下，
            #以最快的速度，在可能已经碎片化的显存池里，完成多路变长序列的注意力计算。
            o = flash_attn_varlen_func(
                q, k, v,
                max_seqlen_q=context.max_seqlen_q,#最长序列长度
                cu_seqlens_q=context.cu_seqlens_q,#拼接起来长度的起始点
                max_seqlen_k=context.max_seqlen_k,
                cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.scale,
                causal=True,
                block_table=context.block_tables
            )
        else:
            # Decode 阶段
            # q: [batch_size, num_heads, head_dim] -> [batch_size, 1, num_heads, head_dim]
            o = flash_attn_with_kvcache(
                q.unsqueeze(1),
                k_cache, v_cache,
                cache_seqlens=context.context_lens,  # 每个序列的历史长度
                block_table=context.block_tables,     # 定位 Cache 中的 KV
                softmax_scale=self.scale,
                causal=True
            )
        return o
