# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING, Optional

from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorBase_V1
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.engine import EngineCoreOutputs
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.metrics.stats import SchedulerStats
    from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
    from vllm.v1.request import Request, RequestStatus
    from vllm.v1.structured_output import StructuredOutputManager


class SchedulerInterface(ABC):
    @abstractmethod
    def __init__(
        self,
        vllm_config: "VllmConfig",
        kv_cache_config: "KVCacheConfig",
        structured_output_manager: "StructuredOutputManager",
        block_size: int,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def schedule(self) -> "SchedulerOutput":
        """Schedule the requests to process in this scheduling step.

        The scheduling decision is made at the iteration level. Each scheduling
        step corresponds to a single forward pass of the model. Therefore, this
        method is called repeatedly by a busy loop in the engine.

        Essentially, the scheduler produces a dictionary of {req_id: num_tokens}
        that specifies how many tokens to process for each request in this
        scheduling step. For example, num_tokens can be as large as the number
        of prompt tokens for new requests, or it can be 1 for the requests that
        are auto-regressively generating new tokens one by one. Otherwise, it
        can be somewhere in between in case of chunked prefills, prefix caching,
        speculative decoding, etc.

        Additionally, the scheduler also returns useful data about each request
        or the batch as a whole. The model runner will use this information in
        preparing inputs to the model.

        Returns:
            A SchedulerOutput object containing information about the scheduled
            requests.
        在本调度步骤中安排需要处理的请求。
        调度决策是在每次迭代层面做出的。每个调度步骤对应模型的一次完整前向传播。
        因此，该方法会被 engine 中的一个忙等待循环（busy loop）反复调用。

        本质上，调度器会生成一个字典 {req_id: num_tokens}，用于说明在这个调度步骤中，
        每个请求需要处理多少个 token。例如：
        - 对于新到达的请求，num_tokens 可能大到等于其完整的 prompt token 数量；
        - 对于正在自回归逐个生成新 token 的请求，num_tokens 通常为 1；
        - 在使用分块预填充（chunked prefills）、前缀缓存（prefix caching）、推测解码（speculative decoding）等场景下，
          则可能取介于两者之间的某个值。

        此外，调度器还会返回关于各个请求或整个批次的一些有用信息。
        模型运行时（model runner）会利用这些信息来准备模型的输入。

        返回：
            一个 SchedulerOutput 对象，包含已调度请求的相关信息。
        """
        raise NotImplementedError

    @abstractmethod
    def get_grammar_bitmask(
        self, scheduler_output: "SchedulerOutput"
    ) -> "GrammarOutput | None":
        """根据当前调度输出返回 grammar 约束的 bitmask。

            返回 None 表示不施加任何语法约束。"""
        raise NotImplementedError

    @abstractmethod
    def update_from_output(
        self,
        scheduler_output: "SchedulerOutput",
        model_runner_output: "ModelRunnerOutput",
    ) -> dict[int, "EngineCoreOutputs"]:
        """Update the scheduler state based on the model runner output.

        This method is called after the model runner has processed the scheduled
        requests. The model runner output includes generated token ids, draft
        token ids for next step, etc. The scheduler uses this information to
        update its states, checks the finished requests, and returns the output
        for each request.

        Returns:
            A dict of client index to EngineCoreOutputs object containing the
            outputs for each request originating from that client.

        根据模型运行时的输出，更新调度器的内部状态。

        该方法在模型运行时（model runner）完成当前批次的推理/生成后被调用。
        模型运行时会返回本次生成的 token id、用于下一轮的 draft token（若使用推测解码）、
        logits（部分实现可能包含）、以及其他相关信息。

        调度器需要利用这些信息完成以下工作：
        - 更新每个请求的已生成序列、KV cache 位置、生成状态等
        - 判断哪些请求已经完成（遇到 EOS、达到最大长度、满足停止条件等）
        - 处理推测解码的接受/拒绝逻辑（若启用）
        - 清理已结束请求占用的资源
        - 收集并组织每个请求的输出结果

        参数:
            scheduler_output:   本次调度步骤中调度器刚刚做出的调度决定
                                （即哪些请求、每个请求分配了多少 token 位置等）
            model_runner_output:模型运行时对本次批次实际执行后的输出结果，
                                包含生成的 token ids、各序列的 logits（可选）、
                                推测 token 的接受掩码等信息

        返回:
            dict[int, EngineCoreOutputs]:
                键为客户端索引（client index，通常对应不同的请求来源或会话分组），
                值为 EngineCoreOutputs 对象，包含该客户端下所有相关请求在本轮的输出结果。
                （通常包括：新生成的 token、是否结束、生成的文本或 token 列表等）

        典型调用时机：
            engine 主循环中：调度 → 模型前向 → update_from_output → 返回给上层/客户端
        """
        raise NotImplementedError

    @abstractmethod
    def update_draft_token_ids(
        self,
        draft_token_ids: "DraftTokenIds",
    ) -> None:
        """Update the draft token ids for the scheduled requests."""
        raise NotImplementedError

    @abstractmethod
    def add_request(self, request: "Request") -> None:
        """Add a new request to the scheduler's internal queue.

        Args:
            request: The new request being added.
        """
        raise NotImplementedError

    @abstractmethod
    def finish_requests(
        self,
        request_ids: str | Iterable[str],
        finished_status: "RequestStatus",
    ) -> None:
        """Finish the requests in the scheduler's internal queue. If the request
        is not in the queue, this method will do nothing.

        This method is called in two cases:
        1. When the request is aborted by the client.
        2. When the frontend process detects a stop string of the request after
           de-tokenizing its generated tokens.

        Args:
            request_ids: A single or a list of request IDs.
            finished_status: The finished status of the given requests.
        """
        raise NotImplementedError

    @abstractmethod
    def get_num_unfinished_requests(self) -> int:
        """Number of unfinished requests in the scheduler's internal queue."""
        raise NotImplementedError

    def has_unfinished_requests(self) -> bool:
        """Returns True if there are unfinished requests in the scheduler's
        internal queue."""
        return self.get_num_unfinished_requests() > 0

    @abstractmethod
    def has_finished_requests(self) -> bool:
        """Returns True if there are finished requests that need to be cleared.
        NOTE: This is different from `not self.has_unfinished_requests()`.

        The scheduler maintains an internal list of the requests finished in the
        previous step. This list is returned from the next call to schedule(),
        to be sent to the model runner in the next step to clear cached states
        for these finished requests.

        This method checks if this internal list of finished requests is
        non-empty. This information is useful for DP attention.
        """
        raise NotImplementedError

    def has_requests(self) -> bool:
        """Returns True if there are unfinished requests, or finished requests
        not yet returned in SchedulerOutputs."""
        return self.has_unfinished_requests() or self.has_finished_requests()

    @abstractmethod
    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        """Reset the prefix cache for KV cache.

        This is particularly required when the model weights are live-updated.

        Args:
            reset_running_requests: If True, all the running requests will be
                preempted and moved to the waiting queue. Otherwise, this method
                will only reset the KV prefix cache when there is no running request
                taking KV cache.
        """
        raise NotImplementedError

    @abstractmethod
    def get_request_counts(self) -> tuple[int, int]:
        """Returns (num_running_reqs, num_waiting_reqs)."""
        raise NotImplementedError

    @abstractmethod
    def make_stats(self) -> Optional["SchedulerStats"]:
        """Make a SchedulerStats object for logging.

        The SchedulerStats object is created for every scheduling step.
        """
        raise NotImplementedError

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the scheduler."""
        raise NotImplementedError

    def get_kv_connector(self) -> Optional["KVConnectorBase_V1"]:
        return None
