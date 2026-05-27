# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Optional

import torch

from vllm import SamplingParams

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class MoveDirectionality(Enum):
    # One-way i1->i2 req move within batch
    UNIDIRECTIONAL = auto()
    # Two-way i1<->i2 req swap within batch
    SWAP = auto()


# Batch indices of any removed requests.
RemovedRequest = int

# (index, params, prompt_tok_ids, output_tok_ids) tuples for new
# requests added to the batch.
AddedRequest = tuple[int, SamplingParams, list[int] | None, list[int]]

# (index 1, index 2, directionality) tuples representing
# one-way moves or two-way swaps of requests in batch
MovedRequest = tuple[int, int, MoveDirectionality]


@dataclass(frozen=True)                                                                     #不可变数据类
class BatchUpdate:
    """Persistent batch state change info for logitsprocs"""                                #这一step中persistent batch的变化信息(供logits processors使用)

    batch_size: int  # Current num reqs in batch                                            #当前batch的大小

    # Metadata for requests added to, removed from, and moved
    # within the persistent batch.
    #
    # Key assumption: the `output_tok_ids` list (which is an element of each                #关键假设:added中每个元素的output_tok_ids实际上是只想request内部输出token列表的引用
    # tuple in `added`) is a reference to the request's running output tokens               这意味着logits processsors看到的始终是最新生成的token列表(因为是引用 而不是拷贝)
    # list; via this reference, the logits processors always see the latest                 举例:request.output_token_ids = [1,2]  processors 拿到的是这个 list 的引用   后面变成 [1,2,3] processors 自动看到更新
    # list of generated output tokens.
    #
    # NOTE:                                                                                 注意事项:
    # * Added or moved requests may replace existing requests with the same                     1.add或moved的request 可能会覆盖同一个index上的旧request
    #   index.
    # * Operations should be processed in the following order:                                  2.所有操作必须按一下顺序执行,这个顺序是为了避免index冲突和错位问题
    #   - removed, added, moved
    removed: Sequence[RemovedRequest]
    added: Sequence[AddedRequest]
    moved: Sequence[MovedRequest]


class LogitsProcessor(ABC):
    @classmethod
    def validate_params(cls, sampling_params: SamplingParams):
        """Validate sampling params for this logits processor.                              校验该Processor所需的sampling参数是否合法    

        Raise ValueError for invalid ones.
        """
        return None

    @abstractmethod
    def __init__(
        self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool
    ) -> None:                                                                              #必须实现初始化逻辑
        raise NotImplementedError

    @abstractmethod
    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply LogitsProcessor to batch logits tensor.                                    对batch的logits进行处理,输入:logits:shape通常是[batch_size, vocab_size]

        The updated tensor must be returned but may be                                      
        modified in-place.
        """
        raise NotImplementedError

    @abstractmethod
    def is_argmax_invariant(self) -> bool:
        """True if logits processor has no impact on the                                    是否影响greedy(argmax)结果
        argmax computation in greedy sampling.
        NOTE: may or may not have the same value for all
        instances of a given LogitsProcessor subclass,
        depending on subclass implementation.
        """
        raise NotImplementedError

    @abstractmethod
    def update_state(
        self,
        batch_update: Optional["BatchUpdate"],
    ) -> None:
        """Called when there are new output tokens, prior                                   每次forward前调用,用于更新内部状态
        to each forward pass.                                                               调用时机:每一步生成前,在apply之前

        Args:
            batch_update: Non-None iff there have been changes                              参数:batch_update
                to the batch makeup.                                                        用于同步batch内request的变化
        """
        raise NotImplementedError
