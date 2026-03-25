# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV cache metrics tracking."""

import random
import time
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import KVCacheBlock

from vllm.v1.metrics.stats import KVCacheEvictionEvent


class BlockMetricsState: #给一个kv cache block记生命周期日志的小记录器  会记录什么时候被访问 最近几次访问之间的间隔。 能帮助分析kv cache是否太小（lifetime 很短则说明block很快被eviction）；cache是否浪费（idle_time 很长说明block占用但没人用）；prefix reuse情况（若reuse_gaps很短，说明block反复被利用）
    """Tracks lifecycle metrics for a single KV cache block.跟踪一个 KV cache block 的生命周期指标"""

    def __init__(self):
        now_ns = time.monotonic_ns() #返回单调时间，不会被系统时间修改影响，只会递增
        self.birth_time_ns = now_ns  #表示block什么时候被创建
        self.last_access_ns = now_ns #最近访问时间 表示block刚被用到
        # Bounded to prevent unbounded growth if a block is accessed many times.
        self.access_history: deque[int] = deque(maxlen=4) #访问历史，双端队列，只留最近4次访问

    def record_access(self) -> None:
        now_ns = time.monotonic_ns()
        self.last_access_ns = now_ns
        self.access_history.append(now_ns)

    def get_lifetime_seconds(self) -> float:#计算block存活了多久
        now_ns = time.monotonic_ns()
        return (now_ns - self.birth_time_ns) / 1e9

    def get_idle_time_seconds(self) -> float: #计算block空闲了多久
        now_ns = time.monotonic_ns()
        return (now_ns - self.last_access_ns) / 1e9

    def get_reuse_gaps_seconds(self) -> list[float]: #计算使用间隔
        if len(self.access_history) < 2:
            return []
        history = list(self.access_history)
        return [(history[i] - history[i - 1]) / 1e9 for i in range(1, len(history))]


class KVCacheMetricsCollector:
    """Collects KV cache residency metrics with sampling. 通过采样方式收集驻留（residency）指标  ， 驻留指的是kv block在gpu cache里存活多久"""

    def __init__(self, sample_rate: float = 0.01):
        assert 0 < sample_rate <= 1.0, (
            f"sample_rate must be in (0, 1.0], got {sample_rate}"
        )
        self.sample_rate = sample_rate

        self.block_metrics: dict[int, BlockMetricsState] = {}  #例如{101：BlockMetricsState, 102:BlockMetricsState}

        self._eviction_events: list[KVCacheEvictionEvent] = [] #事件列表  用于保存block被驱逐时产生的统计结果  这是最终输出给监控系统的数据

    def should_sample_block(self) -> bool: #采样函数，随机数<采样率。所以大约1% block会被记录
        return random.random() < self.sample_rate

    def on_block_allocated(self, block: "KVCacheBlock") -> None:
        if self.should_sample_block():  #如果被采样 则创建 BlockMetricsState
            self.block_metrics[block.block_id] = BlockMetricsState()

    def on_block_accessed(self, block: "KVCacheBlock") -> None:
        metrics = self.block_metrics.get(block.block_id)  #取出统计状态
        if metrics:
            metrics.record_access()

    def on_block_evicted(self, block: "KVCacheBlock") -> None: #生命周期必须到 eviction 才完整，
        metrics = self.block_metrics.pop(block.block_id, None)
        if not metrics:
            return

        lifetime = metrics.get_lifetime_seconds()
        idle_time = metrics.get_idle_time_seconds()
        reuse_gaps = tuple(metrics.get_reuse_gaps_seconds())

        self._eviction_events.append(
            KVCacheEvictionEvent(
                lifetime_seconds=lifetime,
                idle_seconds=idle_time,
                reuse_gaps_seconds=reuse_gaps,
            )
        )

    def reset(self) -> None:
        """Clear all state on cache reset."""
        self.block_metrics.clear()
        self._eviction_events.clear()

    def drain_events(self) -> list[KVCacheEvictionEvent]:
        events = self._eviction_events
        self._eviction_events = []
        return events
