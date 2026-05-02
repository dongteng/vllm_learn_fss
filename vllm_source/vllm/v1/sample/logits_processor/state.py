# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterator
from itertools import chain
from typing import TYPE_CHECKING

from vllm.v1.sample.logits_processor.interface import (
    AddedRequest,
    BatchUpdate,
    MovedRequest,
    RemovedRequest,
)

if TYPE_CHECKING:
    from vllm.v1.sample.logits_processor.interface import LogitsProcessor


class BatchUpdateBuilder:
    """Helps track persistent batch state changes and build                             用于跟踪持久性batch的变化,并构建一个用于logits processors的batch更新数据结构
    a batch update data structure for logitsprocs
    Assumptions:                                                                        假设条件:
    * All information about requests removed from persistent batch                      - 在一个step中,所有从persistent batch中被移除的request信息,都会在step开始时通过调用
      during a step is aggregated in self._removed through calls to                       self.removed_append()被统一收集到self._removed中,并且这一过程必须发生在该step中第一次调用
      self.removed_append() at the beginning of a step. This must happen                  self.removed self.pop_removed()或self.peed_removed()之前
      before the first time that self.removed, self.pop_removed()
      or self.peek_removed() are invoked in a given step
    * After the first time that self.removed, self.pop_removed()                        - 一旦在某个step中第一次读取了self.removed 、self.pop_removed()或self.peek_removed(),就不允许
      or self.peek_removed() are read in a step, no new removals                          再通过self.removed_append()注册新的移除操作       
      are registered using self.removed_append()
    * Elements of self._removed are never directly modified, added or                   - self._removed中的元素不允许被直接修改 添加或删除(也就是说所有修改只能通过self.removed_append()和self.pop_removed)
      removed (i.e. modification is only via self.removed_append() and
      self.pop_removed())
    Guarantees under above assumptions:                                                 在上述假设成立的前提下,可以保证:
    * self.removed is always sorted in descending order                                 - self.removed始终按降序排列
    * self.pop_removed() and self.peek_removed() both return                            - self.pop_removed()和self.peek_removed()返回的,都是当前step中被移除的request中index最小那个
      the lowest removed request index in the current step
    example:
    Current Batch State:
    Index:  0    1    2    3    4
    Req:    A    B    C    D    E
    1. Start of step: Requests B (index 1) and D (index 3) are finished.
        >>> builder.removed_append(1)
        >>> builder.removed_append(3)
        Internal State: _removed = [3, 1]  (Stored in descending order)
    2. Processing Removals:
        >>> builder.peek_removed() -> 1  (Returns the lowest index)
        >>> builder.pop_removed()  -> 1  (Removes and returns 1)
    ⚠️ CRITICAL: After the first peek/pop, calling removed_append() 
        is FORBIDDEN. Doing so would violate sorting guarantees.
    3. Next removal:    >>> builder.pop_removed()  -> 3

    """

    _removed: list[RemovedRequest]                                                      #被移除的request列表(内部使用)注意这里记录的是RemoveRequest(包含原始index等信息)
    _is_removed_sorted: bool                                                            #标记_removed是否已经排好序(按降序) 用于避免重复排序(性能优化)
    added: list[AddedRequest]                                                           #新加入batch的request列表,每个元素描述一个新增request放到哪个位置
    moved: list[MovedRequest]                                                           #在batch内发生位置移动的request列表,例如B从index2到index0

    def __init__(
        self,
        removed: list[RemovedRequest] | None = None,
        added: list[AddedRequest] | None = None,
        moved: list[MovedRequest] | None = None,
    ) -> None:
        self._removed = removed or []
        self.added = added or []
        self.moved = moved or []
        self._is_removed_sorted = False

        # Used to track changes in the pooling case                                      #用于pooling模型,在这种模式下,可能不会填充added列表,所以需要一个往外标志来表示batch是否发生变化.
        # where we don't populate the added list.
        self.batch_changed = False

    def _ensure_removed_sorted(self) -> None:
        """Sort removed request indices in                                               将被移除的request按index降序排序,在同一个step内，多次调用是安全的(只有第一次真正排序)
        descending order.                                                                一旦reset后,才会重新参与下一轮排序
        Idempotent after first call in a
        given step, until reset.
        """
        if not self._is_removed_sorted:
            self._removed.sort(reverse=True)
            self._is_removed_sorted = True

    @property
    def removed(self) -> list[RemovedRequest]:
        """Removed request indices sorted in                                               返回被移除的reqquest列表,按index降序排列
        descending order"""
        self._ensure_removed_sorted()
        return self._removed

    def removed_append(self, index: int) -> None:
        """Register the removal of a request from the persistent batch.                    登记一个request将被从persistent batch中移除。

        Must not be called after the first time self.removed,                              注意:必须在第一次访问removed/pop_removed/peek_removed之前调用
        self.pop_removed() or self.peek_removed() are invoked.                                  一旦开始读取removed,就不允许再新增

        Args:
          index: request index
        """
        if self._is_removed_sorted:                                                        #如果已经排序过m说明处理阶段已经开始,此时不允许再新增removed
            raise RuntimeError(
                "Cannot register new removed request after self.removed has been read."
            )
        self._removed.append(index)                                                        #将该index加入待移除列表(只是登记 还没真正开始处理)
        self.batch_changed = True                                                          #标记batch已发生变化(用于后续调度/更新逻辑)

    def has_removed(self) -> bool:
        return bool(self._removed)                                                          #检查当前是否还有待处理的removed请求

    def peek_removed(self) -> int | None:
        """Return lowest removed request index"""                                          #返回当前最小index的被移除request(不删除)
        if self.has_removed():                                                             #如果还有未处理的removed
            self._ensure_removed_sorted()                                               
            return self._removed[-1]
        return None                                                                                 

    def pop_removed(self) -> int | None:
        """Pop lowest removed request index"""                                              #删除当前最小index的被移除request
        if self.has_removed():
            self._ensure_removed_sorted()
            return self._removed.pop()
        return None

    def reset(self) -> bool:
        """Returns True if there were any changes to the batch."""
        self._is_removed_sorted = False
        self._removed.clear()
        self.added.clear()
        self.moved.clear()
        batch_changed = self.batch_changed
        self.batch_changed = False
        return batch_changed

    def get_and_reset(self, batch_size: int) -> BatchUpdate | None:
        """Generate a logitsprocs batch update data structure and reset                       #生成一个logitsprocs用的batch更新结构,并重置内部状态
        internal batch update builder state.

        Args:
          batch_size: current persistent batch size                                           当前persistent batch的大小

        Returns:                
          Frozen logitsprocs batch update instance; `None` if no updates                      一个冻结的BatchUpdate(本step的变化记录),如果这一轮没有任何变化,则返回None
        """
        # Reset removal-sorting logic
        self._is_removed_sorted = False
        self.batch_changed = False
        if not any((self._removed, self.moved, self.added)):                                  #如果这一轮没有变化(没有removed/moved/added)
            # No update; short-circuit
            return None                                                                       #直接返回None,避免构造空update(性能优化)
        # Build batch state update
        batch_update = BatchUpdate(                                                           #构造这一轮的变化快照,注意这只是打包引用,还没清空内部数据
            batch_size=batch_size,
            removed=self._removed,
            moved=self.moved,
            added=self.added,
        )
        self._removed = []
        self.moved = []
        self.added = []
        return batch_update


class LogitsProcessors:
    """Encapsulates initialized logitsproc objects.封装初始化好的logits processor(对logits做后处理的组件)"""

    def __init__(self, logitsprocs: Iterator["LogitsProcessor"] | None = None) -> None:
        self.argmax_invariant: list[LogitsProcessor] = []                                       #存放不会改变argmax结果的processor,例如对所有logits加同一个常数(不影响最大值是谁)
        self.non_argmax_invariant: list[LogitsProcessor] = []                                   #存放可能改变argmax结果的processor,例如topk top-p penality bad words等
        if logitsprocs:
            for logitproc in logitsprocs:                                                       #根据该processsor是否保持argmax不变进行分类
                (
                    self.argmax_invariant
                    if logitproc.is_argmax_invariant()
                    else self.non_argmax_invariant
                ).append(logitproc)

    @property
    def all(self) -> Iterator["LogitsProcessor"]:
        """Iterator over all logits processors."""
        return chain(self.argmax_invariant, self.non_argmax_invariant)                           #使用chain拼接2个列表,chain的作用是把多个可迭代对象按顺序拼接成一个迭代器,而且不做拷贝
