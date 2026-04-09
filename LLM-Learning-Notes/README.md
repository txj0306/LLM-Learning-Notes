一、文件定位

nanovllm/
├── engine/
│   ├── block_manager.py   # Block 分配与回收、Prefix Caching
│   ├── sequence.py        # Sequence 数据结构（含 block_table）
│   ├── scheduler.py       # 调度器（调用 BlockManager）
│   └── model_runner.py    # KV Cache 物理显存分配、上下文准备
├── layers/
│   └── attention.py       # Attention 计算 + KV Cache 存取
├── utils/
│   └── context.py         # 运行时上下文（slot_mapping、block_tables等）
└── config.py              # kvcache_block_size 等配置

 1.1模块协作关系图
<img width="960" height="569" alt="image" src="https://github.com/user-attachments/assets/cef26016-b1b8-4f88-9978-28a7abe4cb39" />
**核心协作流程**：

1. **Scheduler** 决定哪些 Sequence 参与本次迭代
2. **BlockManager** 为 Sequence 分配/回收 Block，更新 Sequence 的 `block_table`
3. **ModelRunner** 读取 Sequence 的 `block_table`，构造 `slot_mapping` 等上下文信息
4. **Context** 保存运行时上下文，供 Attention 层使用
5. **Attention** 根据上下文读写 KV Cache，完成注意力计算

二、 PagedAttention 核心思想
 2.1PagedAttention 借鉴了操作系统虚拟内存管理的 **分页机制**：

| 操作系统概念 | PagedAttention 类比 |
| --- | --- |
| 页（Page） | Block（固定大小的 KV Cache 块） |
| 页表（Page Table） | Block Table（逻辑位置到物理 Block 的映射） |
| slot | token 级别的物理位置映射 |
| 进程（Process） | Sequence（一个推理请求） |
<img width="973" height="569" alt="image" src="https://github.com/user-attachments/assets/e03c68ba-4861-48fb-824a-770b8444d8e9" />

核心优势：

- **按需分配**：只为实际生成的 token 分配 Block
- **动态管理**：请求结束后立即回收 Block 供其他请求使用
- **支持共享**：相同前缀的请求可共享 Block（Prefix Caching）
- ### 1.1 KV Cache 的内存瓶颈

在 Transformer 架构的大语言模型推理过程中，每生成一个新 token 都需要用到之前所有 token 的 Key 和 Value 向量。为避免重复计算，我们会将这些 KV 向量缓存起来，这就是 **KV Cache**。

KV Cache 的显存占用可以用以下公式估算：

```
KV Cache Size = 2 × num_layers × seq_len × num_kv_heads × head_dim × dtype_size
```

以 Qwen3-0.6B 为例（28层，8个KV头，64维，bf16）：   
-<img width="732" height="427" alt="image" src="https://github.com/user-attachments/assets/f0f3bb8a-1a7a-46f1-b5bc-ac8c6bf452be" />
-<img width="732" height="427" alt="image" src="https://github.com/user-attachments/assets/96ff2b80-457e-477c-9114-2f61103b3b6b" />
- 进入28层每层8个kv头每个头负责64维度
- 单个请求、序列长度 4096：`2 × 28 × 4096 × 8 × 64 × 2 = 234 MB`
- 如果同时服务 32 个请求：约 **7.34 GB** 显存仅用于 KV Cache
- 如果序列长度为10那就会2 × 28 × 10 × 8 × 64 × 2

传统方案的问题在于**预分配最大长度:**
。假设`max_model_len = 4096`，即使一个请求实际只生成 100 个 token，也会预分配 4096 长度的 KV Cache 空间，
导致永远都是2 × 28 × 4096 × 8 × 64 × 2= 234 MB
**60%-80% 的显存浪费**。此外，不同请求的实际长度参差不齐，容易产生**显存碎片**
<img width="1388" height="768" alt="image" src="https://github.com/user-attachments/assets/d705b43c-4daa-4b28-9b6e-bad4539548b6" />
Parameters:是权重/参数
kv catch：是缓存
在相同40g显存下 有kv catch的一个 Batch 里塞进更多的请求（从 8 个变成近 40 个），显卡跑一次计算能同时处理更多人的回复。它的每秒生成 Token 数（0.9k）远远超过了橙色箭头（0.3k）。
2.2 Attention 层的 KV Cache 操作 nanovllm/layers/attention.py

模式 A：笨蛋模式（没有 KV Cache）
生成第 1 个字：计算 100 个词的 KV。
生成第 2 个字：计算 101 个词的 KV。
生成第 3 个字：计算 102 个词的 KV。
结果：计算量是 O(N^2)。当你聊到 2000 字时，显卡就得为了吐出一个字而计算几百万次，这种重复劳动会直接让显卡冒烟。

模式 B：天才模式（带 KV Cache）
Prefill：一次性算出前 100 个词的 KV，存进仓库。
Decode：吐第 1 个字时，只算这 1 个字的 KV，加上仓库里的 100 个。
Decode：吐第 2 个字时，只算这 1 个字的 KV，加上仓库里的 101 个。结果：每一轮的计算量几乎是恒定的（只算当前最新的Q, K, V）。

2.3 Block 类 nanovllm/engine/block_manager.py