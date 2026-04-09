from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:
    """
    物理 Block，KV Cache 的最小存储单元。

    每个 Block 可存储 block_size 个 token 的 KV 向量。
    通过 ref_count 支持多个 Sequence 共享同一 Block（Prefix Caching）。
    通过 hash 和 token_ids 支持缓存查找和碰撞校验。
    """

    def __init__(self, block_id):
        # block_id: 物理 Block 的唯一标识，对应 KV Cache Tensor 的第 2 维索引
        # 创建后不变，范围是 [0, num_kvcache_blocks)
        self.block_id = block_id

        # ref_count: 引用计数
        # = 0: Block 空闲，在 free_block_ids 中
        # = 1: 被一个 Sequence 独占使用
        # > 1: 被多个 Sequence 共享（Prefix Caching 场景）
        self.ref_count = 0

        # hash: Block 内容的 xxhash 值，用于 Prefix Caching 快速查找
        # = -1: Block 未填满，或不参与缓存
        # != -1: Block 已填满，可被后续请求复用
        self.hash = -1

        # token_ids: Block 中存储的 token 序列
        # 用于 hash 碰撞时的精确校验，确保内容真正相同
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        """
        更新 Block 的缓存标识。
        只在 Block 填满（包含完整 block_size 个 token）时调用。
        更新后，该 Block 可被后续具有相同前缀的请求复用。
        """
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        """
        重置 Block 状态，供新分配使用。
        在从 free_block_ids 取出并分配给新 Sequence 时调用。
        """
        self.ref_count = 1      # 新分配，初始引用计数为 1
        self.hash = -1          # 清除旧的 hash（新内容待写入）
        self.token_ids = []     # 清除旧的 token_ids

class BlockManager:
    """
    Block 管理器，PagedAttention 的核心组件。
 
    职责：
    1. 管理物理 Block 的分配和回收
    2. 维护空闲 Block 池
    3. 实现 Prefix Caching（通过 hash 索引和引用计数）
    """

    def __init__(self, num_blocks: int, block_size: int):
        # block_size: 每个 Block 容纳的 token 数，默认 256
        self.block_size = block_size

        # blocks: 所有物理 Block 实例的列表
        # 索引即 block_id，长度为 num_blocks
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]

        # hash_to_block_id: hash 值到 block_id 的映射
        # Prefix Caching 的核心索引，用于 O(1) 查找是否存在相同内容的 Block
        self.hash_to_block_id: dict[int, int] = dict()

        # free_block_ids: 空闲 Block ID 队列
        # 使用 deque 实现 FIFO 分配策略
        self.free_block_ids: deque[int] = deque(range(num_blocks))

        # used_block_ids: 已使用的 Block ID 集合
        # 用于 O(1) 判断某个 Block 是否正在被使用
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """
        计算 Block 内容的 hash 值。

        使用链式 hash：当前 Block 的 hash 依赖于前缀 Block 的 hash。
        这确保了只有「前缀完全相同」的 Block 序列才能匹配。

        Args:
            token_ids: 当前 Block 的 token 列表
            prefix: 前一个 Block 的 hash 值，-1 表示这是第一个 Block

        Returns:
            64 位整数 hash 值
        """
        h = xxhash.xxh64()
        if prefix != -1:
            # 将前缀 hash 纳入计算，实现链式依赖
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        """
        内部方法：将指定 Block 从空闲池移到已使用集合。
        """
        block = self.blocks[block_id]
        assert block.ref_count == 0  # 确保 Block 确实是空闲的
        block.reset()                # 重置状态
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        """
        内部方法：将指定 Block 从已使用集合移回空闲池。
        注意：不清除 hash 和 token_ids，以便后续可能的缓存命中。
        """
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)  # 放到队尾，FIFO

    def can_allocate(self, seq: Sequence) -> bool:
        """
        检查是否有足够的空闲 Block 为 Sequence 分配。

        这是保守估计，未考虑 Prefix Caching 命中可能减少的需求。
        实际分配时可能因缓存命中而需要更少的 Block。
        """
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        """
        为 Sequence 分配 Block，核心方法。

        包含完整的 Prefix Caching 逻辑：
        1. 遍历 Sequence 的每个逻辑 Block
        2. 计算链式 hash，查找缓存
        3. 缓存命中则复用，未命中则新分配
        4. 更新 Sequence 的 block_table

        调用时机：Prefill 阶段，新请求开始执行时
        """
        assert not seq.block_table  # 确保是新请求，block_table 应为空
        h = -1                       # 前缀 hash，用于链式计算
        cache_miss = False           # 一旦发生 miss，后续都是 miss

        for i in range(seq.num_blocks):
            token_ids = seq.block(i)  # 获取第 i 个逻辑 Block 的 token

            # 只有完整 Block（包含 block_size 个 token）才计算 hash
            # 最后一个未填满的 Block 不参与缓存
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1

            # 在缓存索引中查找
            block_id = self.hash_to_block_id.get(h, -1)

            # 双重校验：hash 匹配 + 内容匹配
            # 防止 hash 碰撞或 Block 被覆写导致的错误命中
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True

            if cache_miss:
                # Cache Miss：从空闲池分配新 Block
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                # Cache Hit：复用已有 Block
                seq.num_cached_tokens += self.block_size  # 累加缓存命中的 token 数
                if block_id in self.used_block_ids:
                    # Block 正被其他 Sequence 使用，增加引用计数
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    # Block 在空闲池中（之前被回收但 hash 保留），重新激活
                    block = self._allocate_block(block_id)

            # 更新 Block 的 hash 和 token_ids（仅完整 Block）
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id

            # 将 block_id 加入 Sequence 的 block_table
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        """
        释放 Sequence 占用的所有 Block。

        通过引用计数实现：
        - ref_count 减 1
        - 只有当 ref_count 降为 0 时才真正释放

        调用时机：
        1. 请求完成（postprocess 中检测到 EOS 或达到 max_tokens）
        2. 请求被抢占（preempt）
        """
        for block_id in reversed(seq.block_table):  # 逆序遍历（栈语义）
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """
        检查是否能为 Sequence 追加新 token（可能需要新 Block）。

        只有当 len(seq) % block_size == 1 时（即上一个 Block 刚满），
        才需要分配新 Block。其他情况直接写入现有 Block。

        调用时机：Decode 阶段，每次迭代前检查
        """
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """
        Decode 阶段追加 Block 的处理。

        三种情况：
        1. len % block_size == 1: 需要新 Block（上一个已满）
        2. len % block_size == 0: 当前 Block 刚填满，更新其 hash
        3. 其他: Block 正在填充中，无需操作

        调用时机：Decode 阶段，can_append 返回 True 后调用
        """
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]

        if len(seq) % self.block_size == 1:
            # 情况1：刚好需要新 Block（上一个已满）
            assert last_block.hash != -1  # 上一个 Block 应该已经完整并有 hash
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)

        elif len(seq) % self.block_size == 0:
            # 情况2：当前 Block 刚好填满，更新其 hash（供后续 Prefix Cache）
            assert last_block.hash == -1  # 之前应该是未完成状态
            token_ids = seq.block(seq.num_blocks - 1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id

        else:
            # 情况3：Block 正在填充中，无需操作
            assert last_block.hash == -1  # 确认是未完成状态
