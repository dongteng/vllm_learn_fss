# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import vllm.envs as envs
from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.v1.metrics.perf import PerfStats
from vllm.v1.spec_decode.metrics import SpecDecodingStats

if TYPE_CHECKING:
    from vllm.v1.engine import EngineCoreEvent, EngineCoreOutput, FinishReason


@dataclass
class BaseCacheStats:
    """Stores cache hit statistics."""

    reset: bool = False
    """Whether the cache was reset."""

    requests: int = 0
    """The number of requests in this update."""

    queries: int = 0
    """The number of queries in these requests."""

    hits: int = 0
    """The number of hits in these requests."""


class CachingMetrics:
    """Metrics for caching with a hit rate of the most recent N requests.
    Args:
        interval: The number of the most recent requests to aggregate.
            Defaults to 1000.
    """

    def __init__(self, max_recent_requests: int = 1000) -> None:
        super().__init__()

        self.max_recent_requests = max_recent_requests
        # The current aggregated values.
        self.aggregated_requests = 0
        self.aggregated_query_total = 0
        self.aggregated_query_hit = 0

        # A deque of (requests, queries, hits) for the most recent requests.
        self.query_queue = deque[tuple[int, int, int]]()

    def observe(self, stats: BaseCacheStats):
        """Observe the prefix caching for a set of requests.

        This function is called with information gathered when new requests
        are being scheduled and are looking for computed blocks.

        When there are more than `max_recent_requests` requests, the oldest set
        of requests are removed from the metrics.

        Args:
            stats: The prefix cache stats.
        """
        # reset_prefix_cache was invoked before the current update.
        # Reset the metrics before aggregating the current stats.
        if stats.reset:
            self.reset()

        # DO NOT appending empty stats to avoid helpful info get kicked out
        # due to sliding window.
        if stats.requests == 0:
            return

        # Update the metrics.
        self.query_queue.append((stats.requests, stats.queries, stats.hits))
        self.aggregated_requests += stats.requests
        self.aggregated_query_total += stats.queries
        self.aggregated_query_hit += stats.hits

        # Remove the oldest stats until number of requests does not exceed
        # the limit.
        # NOTE: We preserve the latest added stats regardless.
        while (
            len(self.query_queue) > 1
            and self.aggregated_requests > self.max_recent_requests
        ):
            old_requests, old_queries, old_hits = self.query_queue.popleft()
            self.aggregated_requests -= old_requests
            self.aggregated_query_total -= old_queries
            self.aggregated_query_hit -= old_hits

    def reset(self):
        """Reset the metrics."""
        self.aggregated_requests = 0
        self.aggregated_query_total = 0
        self.aggregated_query_hit = 0
        self.query_queue.clear()

    @property
    def empty(self) -> bool:
        """Return true if no requests have been observed."""
        return self.aggregated_requests == 0

    @property
    def hit_rate(self) -> float:
        """Calculate the hit rate for the past N requests."""
        if self.aggregated_query_total == 0:
            return 0.0
        return self.aggregated_query_hit / self.aggregated_query_total


@dataclass
class PrefixCacheStats(BaseCacheStats):
    """
    Stores prefix cache hit statistics.
    - `reset`: Whether `reset_prefix_cache` was invoked.
    - `queries`: Refers to the number of tokens that were queried.
    """

    preempted_requests: int = 0 #本次更新中，有多少个请求是"之前被抢占的请求"
    """The number of previously preempted requests in this update."""

    preempted_queries: int = 0 #：所有被抢占过的请求，在本次更新中总共查询了多少个 token（尝试匹配前缀缓存）
    """The `queries` number for preempted requests."""

    preempted_hits: int = 0 #所有被抢占过的请求，在本次更新中总共命中了多少个 token（成功从前缀缓存复用）
    """The `hits` number for preempted requests."""

    def record(self, num_tokens: int, num_hits: int, preempted: bool) -> None:
        """Aggregate request information into the stats.
        每次有一个请求处理完前缀缓存匹配后，就调用这个方法来累加统计
        num_tokens：这个请求查询了多少个 token（尝试匹配前缀）
        num_hits：这个请求命中了多少个 token（成功复用缓存）
        preempted：这个请求是不是之前被抢占过的（是 True / 否 False）
        为什么分开统计？
        因为在 vLLM 的调度器里：
        新请求：通常从头 prefill，前缀缓存命中率较低
        被抢占后恢复的请求：之前已经 prefill 过一部分，恢复时有大量 token 可以从 prefix cache 命中
        如果混在一起统计，命中率会被“被抢占请求”拉高，看起来“缓存很强”，但实际上新请求的体验没改善
        分开统计后，vLLM 可以更准确地评估：
        新请求的前缀缓存效果（更能反映真实用户体验）
        抢占机制对缓存命中的影响（评估调度策略好坏）
        """
        if preempted:
            # Previously preempted request
            self.preempted_requests += 1
            self.preempted_queries += num_tokens
            self.preempted_hits += num_hits
        else:
            # New request
            self.requests += 1
            self.queries += num_tokens
            self.hits += num_hits


@dataclass
class MultiModalCacheStats(BaseCacheStats):
    """
    Stores multi-modal cache hit statistics.
    - `reset`: Whether `reset_mm_cache` was invoked.
    - `queries`: Refers to the number of multi-modal data items
      that were queried.
    """


@dataclass
class KVCacheEvictionEvent:
    """Single KV cache block eviction sample."""

    lifetime_seconds: float
    idle_seconds: float
    reuse_gaps_seconds: tuple[float, ...]


@dataclass
class SchedulerStats:
    """Stats associated with the scheduler.
    记录和统计vllm 调度器在当前时刻的各种运行状态和性能指标，主要用于内部监控、日志记录、负载均衡、调试、Prometheus 暴露指标等。
    简单说就是，一个调度器快照或调度器状态报告，每一步调度循环结束时都会更新或生成这样一个对象，里面包含了当前有多少请求在跑、等候、KV cache 使用率、LoRA 适配器负载、推测解码统计等所有关键信息。
    """

    num_running_reqs: int = 0 #正在运行的请求数
    num_waiting_reqs: int = 0 #在队列里的

    # These are used for internal DP load-balancing.
    #用来判断调度器负载、是否需要扩容、是否卡住
    step_counter: int = 0 #调度器执行的总步数
    current_wave: int = 0 #当前“波次”（wave），用于数据并行（DP）负载均衡，vLLM 的 DP 模式下，多个 GPU 会分担请求，wave 用来同步“这一波请求都处理完了吗”

    kv_cache_usage: float = 0.0 #前 KV cache 的使用率（0.0 ~ 1.0 或更高） 超过阈值时会触发 eviction（驱逐旧 cache）

    prefix_cache_stats: PrefixCacheStats = field(default_factory=PrefixCacheStats) #前缀缓存统计
    connector_prefix_cache_stats: PrefixCacheStats | None = None #连接器（多节点/多 GPU）的前缀缓存统计（分布式场景用）

    kv_cache_eviction_events: list[KVCacheEvictionEvent] = field(default_factory=list) #本次调度中发生的 KV cache 驱逐事件列表（哪些请求的哪些 block 被踢出）

    spec_decoding_stats: SpecDecodingStats | None = None #推测解码（speculative decoding）的统计（draft 命中率、验证失败率、加速倍数等）推测解码（speculative decoding）的统计（draft 命中率、验证失败率、加速倍数等）
    kv_connector_stats: dict[str, Any] | None = None #KV cache 连接器统计（多节点 KV transfer 的状态、延迟等）

    waiting_lora_adapters: dict[str, int] = field(default_factory=dict) #等待加载的 LoRA 适配器（adapter name → 等待请求数）
    running_lora_adapters: dict[str, int] = field(default_factory=dict) #正在运行的Lora适配器

    cudagraph_stats: CUDAGraphStat | None = None #CUDA Graph 的统计（捕获的 graph 数量、捕获时间、重放命中率等）

    perf_stats: PerfStats | None = None #整体性能统计（TTFT、TPOT、调度延迟、GPU 利用率等汇总）


@dataclass
class RequestStateStats:
    """Stats that need to be tracked across delta updates."""

    num_generation_tokens: int = 0

    # This is an engine frontend timestamp (wall-clock)
    arrival_time: float = 0.0

    # These are engine core timestamps (monotonic)
    queued_ts: float = 0.0
    scheduled_ts: float = 0.0
    first_token_ts: float = 0.0
    last_token_ts: float = 0.0

    # first token latency
    first_token_latency: float = 0.0

    # Track if this request is corrupted (NaNs in logits)
    is_corrupted: bool = False


@dataclass
class FinishedRequestStats:
    """Stats associated with a finished request."""

    finish_reason: "FinishReason"
    e2e_latency: float = 0.0
    num_prompt_tokens: int = 0
    num_generation_tokens: int = 0
    max_tokens_param: int | None = None
    queued_time: float = 0.0
    prefill_time: float = 0.0
    inference_time: float = 0.0
    decode_time: float = 0.0
    mean_time_per_output_token: float = 0.0
    is_corrupted: bool = False
    num_cached_tokens: int = 0


class IterationStats:
    """Stats associated with a single set of EngineCoreOutputs."""

    def __init__(self):
        self.iteration_timestamp = time.time()
        self.num_generation_tokens = 0
        self.num_prompt_tokens = 0
        self.num_preempted_reqs = 0
        self.finished_requests: list[FinishedRequestStats] = []
        self.max_num_generation_tokens_iter: list[int] = []
        self.n_params_iter: list[int] = []
        self.time_to_first_tokens_iter: list[float] = []
        self.inter_token_latencies_iter: list[float] = []
        self.num_corrupted_reqs: int = 0

    def __repr__(self) -> str:
        field_to_value_str = ", ".join(f"{k}={v}" for k, v in vars(self).items())
        return f"{self.__class__.__name__}({field_to_value_str})"

    def _time_since(self, start: float) -> float:
        """Calculate an interval relative to this iteration's timestamp."""
        return self.iteration_timestamp - start

    def update_from_output(
        self,
        output: "EngineCoreOutput",
        engine_core_timestamp: float,
        is_prefilling: bool,
        prompt_len: int,
        req_stats: RequestStateStats,
        lora_states: "LoRARequestStates",
        lora_name: str | None,
    ):
        num_new_generation_tokens = len(output.new_token_ids)

        self.num_generation_tokens += num_new_generation_tokens
        if is_prefilling:
            self.num_prompt_tokens += prompt_len

            first_token_latency = self._time_since(req_stats.arrival_time)
            self.time_to_first_tokens_iter.append(first_token_latency)
            req_stats.first_token_latency = first_token_latency

        req_stats.num_generation_tokens += num_new_generation_tokens

        # Track if this request is corrupted (only check once per request)
        # Early exit if already marked as corrupted to avoid redundant checks
        if (
            envs.VLLM_COMPUTE_NANS_IN_LOGITS
            and not req_stats.is_corrupted
            and output.num_nans_in_logits > 0
        ):
            req_stats.is_corrupted = True

        # Process request-level engine core events
        if output.events is not None:
            self.update_from_events(
                output.request_id,
                output.events,
                is_prefilling,
                req_stats,
                lora_states,
                lora_name,
            )

        # Process the batch-level "new tokens" engine core event
        if is_prefilling:
            req_stats.first_token_ts = engine_core_timestamp
        else:
            itl = engine_core_timestamp - req_stats.last_token_ts
            self.inter_token_latencies_iter.append(itl)

        req_stats.last_token_ts = engine_core_timestamp

    def update_from_events(
        self,
        req_id: str,
        events: list["EngineCoreEvent"],
        is_prefilling: bool,
        req_stats: RequestStateStats,
        lora_states: "LoRARequestStates",
        lora_name: str | None,
    ):
        # Avoid circular dependency
        from vllm.v1.engine import EngineCoreEventType

        for event in events:
            if event.type == EngineCoreEventType.QUEUED:
                req_stats.queued_ts = event.timestamp
                lora_states.request_waiting(req_id, lora_name)
            elif event.type == EngineCoreEventType.SCHEDULED:
                if req_stats.scheduled_ts == 0.0:  # ignore preemptions
                    req_stats.scheduled_ts = event.timestamp
                lora_states.request_running(req_id, lora_name)
            elif event.type == EngineCoreEventType.PREEMPTED:
                self.num_preempted_reqs += 1
                lora_states.request_waiting(req_id, lora_name)

    def update_from_finished_request(
        self,
        finish_reason: "FinishReason",
        num_prompt_tokens: int,
        max_tokens_param: int | None,
        req_stats: RequestStateStats,
        num_cached_tokens: int = 0,
    ):
        e2e_latency = self._time_since(req_stats.arrival_time)

        # Queued interval is from first QUEUED event to first SCHEDULED
        queued_time = req_stats.scheduled_ts - req_stats.queued_ts

        # Prefill interval is from first SCHEDULED to first NEW_TOKEN
        # Any preemptions during prefill is included in the interval
        prefill_time = req_stats.first_token_ts - req_stats.scheduled_ts

        # Decode interval is from first NEW_TOKEN to last NEW_TOKEN
        # Any preemptions during decode are included
        decode_time = req_stats.last_token_ts - req_stats.first_token_ts

        # Inference interval is from first SCHEDULED to last NEW_TOKEN
        # Any preemptions during prefill or decode are included
        inference_time = req_stats.last_token_ts - req_stats.scheduled_ts

        # Do not count the token generated by the prefill phase
        mean_time_per_output_token = (
            decode_time / (req_stats.num_generation_tokens - 1)
            if req_stats.num_generation_tokens - 1 > 0
            else 0
        )

        finished_req = FinishedRequestStats(
            finish_reason=finish_reason,
            e2e_latency=e2e_latency,
            num_prompt_tokens=num_prompt_tokens,
            num_generation_tokens=req_stats.num_generation_tokens,
            max_tokens_param=max_tokens_param,
            queued_time=queued_time,
            prefill_time=prefill_time,
            inference_time=inference_time,
            decode_time=decode_time,
            mean_time_per_output_token=mean_time_per_output_token,
            is_corrupted=req_stats.is_corrupted,
            num_cached_tokens=num_cached_tokens,
        )
        self.finished_requests.append(finished_req)

        # Count corrupted requests when they finish (only once per request)
        if req_stats.is_corrupted:
            self.num_corrupted_reqs += 1


class LoRAStats:
    """Tracks waiting and running request IDs for a single LoRA."""

    def __init__(self):
        self.waiting: set[str] = set()
        self.running: set[str] = set()

    def update(self, req_id: str, waiting: bool, running: bool):
        assert not (waiting and running)
        if waiting:
            self.waiting.add(req_id)
        else:
            self.waiting.discard(req_id)

        if running:
            self.running.add(req_id)
        else:
            self.running.discard(req_id)

    @property
    def empty(self) -> bool:
        return not (self.waiting or self.running)


class LoRARequestStates:
    """A per-LoRA count of running and waiting requests."""

    def __init__(self, log_stats: bool = False):
        self.log_stats = log_stats
        self.requests: defaultdict[str, LoRAStats] = defaultdict(LoRAStats)

    def _request_update(
        self, req_id: str, lora_name: str | None, waiting: bool, running: bool
    ):
        if not self.log_stats or lora_name is None:
            return

        lora_stats = self.requests[lora_name]
        lora_stats.update(req_id, waiting, running)
        if lora_stats.empty:
            del self.requests[lora_name]

    def request_waiting(self, req_id: str, lora_name: str | None):
        self._request_update(req_id, lora_name, waiting=True, running=False)

    def request_running(self, req_id: str, lora_name: str | None):
        self._request_update(req_id, lora_name, waiting=False, running=True)

    def request_finished(self, req_id: str, lora_name: str | None):
        self._request_update(req_id, lora_name, waiting=False, running=False)

    def update_scheduler_stats(self, scheduler_stats: SchedulerStats | None):
        if not self.log_stats or scheduler_stats is None:
            return
        for lora_name, stats in self.requests.items():
            scheduler_stats.waiting_lora_adapters[lora_name] = len(stats.waiting)
            scheduler_stats.running_lora_adapters[lora_name] = len(stats.running)
