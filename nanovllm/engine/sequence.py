from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams

class SequenceStatus(Enum):
    """Sequence 的状态枚举"""
    WAITING = auto()   # 等待调度
    RUNNING = auto()   # 正在执行
    FINISHED = auto()  # 已完成


class Sequence:
    """
    推理请求的抽象，包含输入 token、生成状态和 Block 映射。

    核心职责：
    1. 存储 prompt 和生成的 token
    2. 维护 block_table（逻辑 Block 到物理 Block 的映射）
    3. 记录 Prefix Caching 命中信息
    """

    # 类变量：所有 Sequence 共享的 Block 大小
    block_size = 256

    # 类变量：Sequence ID 生成器，确保每个请求有唯一 ID
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params=SamplingParams()):
        # seq_id: 唯一标识符，用于结果排序和追踪
        self.seq_id = next(Sequence.counter)

        # status: 当前状态（WAITING -> RUNNING -> FINISHED）
        self.status = SequenceStatus.WAITING

        # token_ids: 完整的 token 序列（prompt + 已生成的 token）
        # 使用 copy 避免外部修改影响
        self.token_ids = copy(token_ids)

        # last_token: 最后一个 token，Decode 阶段的输入
        self.last_token = token_ids[-1]

        # num_tokens: 当前总 token 数（prompt + 已生成）
        self.num_tokens = len(self.token_ids)

        # num_prompt_tokens: prompt 的 token 数，不变
        self.num_prompt_tokens = len(token_ids)

        # num_cached_tokens: Prefix Caching 命中的 token 数
        # 由 BlockManager.allocate 设置，用于跳过已缓存部分的计算
        self.num_cached_tokens = 0

        # block_table: 逻辑 Block 索引 -> 物理 Block ID 的映射
        # 例如 [7, 3, 12] 表示：
        #   逻辑 Block 0 -> 物理 Block 7
        #   逻辑 Block 1 -> 物理 Block 3
        #   逻辑 Block 2 -> 物理 Block 12
        self.block_table = []

        # 采样参数
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        """返回当前 token 总数"""
        return self.num_tokens

    def __getitem__(self, key):
        """支持切片访问 token_ids"""
        return self.token_ids[key]

    @property
    def is_finished(self):
        """是否已完成"""
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        """已生成的 token 数（不含 prompt）"""
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        """prompt 部分的 token"""
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        """生成部分的 token"""
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        """Prefix Caching 命中的 Block 数"""
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        """
        当前需要的 Block 总数。
        向上取整：(num_tokens + block_size - 1) // block_size
        """
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        """
        最后一个 Block 中的 token 数量。
        可能不满 block_size（正在填充中）。
        """
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        """
        获取第 i 个逻辑 Block 对应的 token 列表。
        用于计算 hash 和内容校验。
        """
        assert 0 <= i < self.num_blocks
        return self.token_ids[i * self.block_size: (i + 1) * self.block_size]

    def append_token(self, token_id: int):
        """
        追加新生成的 token。
        在 Scheduler.postprocess 中调用。
        """
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        """
        序列化支持（用于多进程通信）。
        只传输必要的字段，减少通信开销。
        """
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens,
                self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    def __setstate__(self, state):
        """反序列化"""
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]

