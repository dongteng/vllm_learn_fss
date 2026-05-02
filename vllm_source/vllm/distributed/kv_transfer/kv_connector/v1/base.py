# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
KVConnectorBase_V1 Class for Distributed KV Cache & Hidden State            用于vllm v1中分布式kv cache和hidden state通信的基础类
communication in vLLM v1

The class provides the following primitives:                                提供如下基础操作 primotives
    Scheduler-side: runs in the scheduler, binds metadata, which            scheduler侧： 绑定元数据 这些metadata会被worker侧用来加载 保存 kv cache
    is used by the worker-side to load/save KV cache.
        get_num_new_matched_tokens() - get number of new tokens                 get_num_new_matched_tokens:获取远端kv cache中已经存在的新token数量
            that exist in the remote KV cache. Might be called multiple                                    对同一个请求可能会被调用多次，并且必须是无副作用的（side-effect free）
            times for a given request and should be side-effect free.
        update_state_after_alloc() - update KVConnector state after             update_state_after_alloc：在CacheManager分配临时buffer之后，更新KVConnector的内部状态
            temporary buffer alloc by the CacheManager.
        update_connector_output() - update KVConnector state after              update_connector_output：在收到 worker 侧 connector 的输出之后，更新 KVConnector 的状态。
            output is received from worker-side connectors.
        request_finished() - called once when a request is finished,            request_finished：当一个请求完成时调用（只调用一次），并传入该请求对应的KV cache blocks
            with the computed kv cache blocks for the request.                                    返回值表示：是否应该立即释放kv cache;  或者由connector接管，在后台异步释放这些block
            Returns whether KV cache should be freed now or if the                                           同时还可以可选 返回KV传输相关参数
            connector now assumes responsibility for freeing the
            the blocks asynchronously. Also optionally returns KV
            transfer params.
        take_events() - returns new KV events that were collected                                 返回自上次调用依赖connector收集到的新kv 相关事件
            by the connector since the last call.

    Worker-side: runs in each worker, loads/saves KV cache to/from          worker侧：运行在每个worker上，负责根据metadata,从connector中加载、保存kv cache
    the Connector based on the metadata.                                           start_load_kv:开始记载所有kv （可能是异步的）
        start_load_kv() - starts loading all KVs (maybe async)
        wait_for_layer_load() - blocks until layer i load is done                  wait_for_layer_load:阻塞直到第i层的kv 加载完成

        save_kv_layer() - starts saving KV for layer i (maybe async)               save_kv_layer：开始保存第i层的kv（可能是异步的） 
        wait_for_save() - blocks until all saves are done                          wait_for_save:阻塞知道所有kv保存完成

        get_finished() - called with ids of finished requests, returns              get_finished:输入已完成的请求ID  输出那些已经完成异步发送、接收 KV的请求ID
            ids of requests that have completed async sending/recving.
"""

import enum
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Optional

import torch

from vllm.attention.backends.abstract import AttentionBackend, AttentionMetadata
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import KVConnectorOutput

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_events import KVCacheEvent, KVConnectorKVEvents
    from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
        KVConnectorPromMetrics,
        KVConnectorStats,
        PromMetric,
        PromMetricT,
    )
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

# s_tensor_list, d_tensor_list, s_indices, d_indices, direction
CopyBlocksOp = Callable[
    [
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        list[int],
        list[int],
        Literal["h2d", "d2h"],
    ],
    None,
]

logger = init_logger(__name__)


class SupportsHMA(ABC):
    """
    The class that indicates the corresponding connector supports hybrid memory         一个标识类，用来表明对应额度connector支持混合内存分配器（HMA）
    allocator (HMA).                        
    This is required to use the connector together with hybrid memory allocator.        如果想要将该connector与HMA一起使用，这是一个必要条件
    """

    @abstractmethod
    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        """                                                         
        Called exactly once when a request has finished for all kv cache groups,        当一个请求在所有KV cache分组（groups）上都完成时调用
        before its blocks are freed for each group.                                     且发生在每个分组的block被释放之前，该函数只会被调用一次

        NOTE(Kuntai): This function is only supported by connectors that support HMA.   该函数只适用于支持HMA的connector (HMA指的是 一种把KV cache分层放在多种内存(CPU GPU 远端)里的内存管理策略)

        The connector may assumes responsibility for freeing the blocks                 可以通过返回True，表示由其接管这些block的释放
        asynchronously by returning True.                                               并在后台异步完成释放

        Returns:
            True if the request is being saved/sent asynchronously and blocks           如果返回True,表示该请求正在进行异步保存/发送
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs             dict[str, Any] | None：可选的 KVTransferParams，会被包含在 engine 返回的请求输出中。
            returned by the engine.
        """
        raise NotImplementedError


def supports_hma(connector: Any) -> bool:
    if isinstance(connector, type):
        return issubclass(connector, SupportsHMA)
    else:
        return isinstance(connector, SupportsHMA)


class KVConnectorRole(enum.Enum):
    # Connector running in the scheduler process
    SCHEDULER = 0

    # Connector running in the worker process
    WORKER = 1


class KVConnectorHandshakeMetadata(ABC):  # noqa: B024
    """
    Metadata used for out of band connector handshake between
    P/D workers. This needs to serializeable.
    """

    pass


class KVConnectorMetadata(ABC):  # noqa: B024
    """
    Abstract Metadata used to communicate between the
    Scheduler KVConnector and Worker KVConnector.
    """

    pass


class KVConnectorBase_V1(ABC):
    """
    Base class for KV connectors.                                               KV connector的抽象基类,作用是定义KV Cache在不同设备/节点之间如何传输 管理的统一接口
                                                                                可以把它理解为:KV Cache的通信+生命周期管理的抽象层
    Attributes:
        prefer_cross_layer_blocks (bool): Indicates whether this connector      prefer_cross_layer_blocks 是否更倾向使用跨层(cross) kv block
            prefers KV blocks that hold KV data for all layers (for speeding     cross-layer block指的是一个block里包含所有layer的KV 而不是每一层单独一个Block
            up KV data transfers).
            Defaults to False.
    """

    prefer_cross_layer_blocks: ClassVar[bool] = False                           #类变量,所有实例共享

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ):
        """
        初始化KVConnector 
        参数说明:vllm_config:全局配置
                role:当前connector的角色(非常关键)-scheduler侧
                                                 - worker侧  表明这个connector是做决策还是干活
        kv_cache_config: kv cache的布局/block/分配策略配置
        """
        
        logger.warning(
            "Initializing KVConnectorBase_V1. This API is experimental and "    #实验阶段,未来可能会变
            "subject to change in the future as we iterate the design."
        )
        self._connector_metadata: KVConnectorMetadata | None = None             #用于 scheduler 和 worker 之间传递 KV 信息（比如 block mapping、状态等）
        self._vllm_config = vllm_config
        if vllm_config.kv_transfer_config is not None:
            self._kv_transfer_config = vllm_config.kv_transfer_config
        else:
            raise ValueError("kv_transfer_config must be set for KVConnectorBase_V1")
        self._kv_cache_config = kv_cache_config
        if self._kv_cache_config is None:
            logger.warning(
                "KVConnectorBase_V1 initialized without kv_cache_config. "
                "This is deprecated - please update your connector to accept "
                "kv_cache_config as the third constructor argument and pass it "
                "to super().__init__()."
            )
        self._role = role

    @property
    def role(self) -> KVConnectorRole:
        return self._role

    # ==============================
    # Worker-side methods
    # ==============================

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        """Set the connector metadata from the scheduler.                           从scheduler设置(绑定)connector的元数据

        This function should be called by the model runner every time               该函数应由model runner在每次模型执行前调用一次
        before the model execution. The metadata will be used for runtime           这些元数据会在运行时用于KV CACHE的加载与保存
        KV cache loading and saving.

        Args:
            connector_metadata (dict): the connector metadata.                      
        """
        self._connector_metadata = connector_metadata

    def clear_connector_metadata(self) -> None:
        """Clear the connector metadata.

        This function should be called by the model runner every time
        after the model execution.
        """
        self._connector_metadata = None

    def _get_connector_metadata(self) -> KVConnectorMetadata:
        """Get the connector metadata.

        This function should only be called inside the connector.

        Returns:
            ConnectorMetadata: the connector metadata.
        """
        # Should only be called while set to valid metadata.
        assert self._connector_metadata is not None
        return self._connector_metadata

    def has_connector_metadata(self) -> bool:
        """Check whether the connector metadata is currently set.

        Returns:
            bool: True if connector metadata exists, False otherwise.
        """
        return self._connector_metadata is not None

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """
        Initialize with the KV caches. Useful for pre-registering the
        KV Caches in the KVConnector (e.g. for NIXL).

        Args:
            kv_caches: dictionary of layer names, kv cache
        """
        return

    def register_cross_layers_kv_cache(
        self, kv_cache: torch.Tensor, attn_backend: type["AttentionBackend"]
    ):
        """
        Initialize with a single KV cache tensor used by all layers.                    使用一个跨所有层共享的kv cache张量进行初始化
        The first dimension should be num_layers.                                       该张量的第一个维度应为num_layers(层数)
        This function will only be called for models with uniform layers,               该函数只会在以下情况被调用:
        and only if the prefers_cross_layer_blocks is set to True.                          模型的各层是统一结构（uniform layers）
        Only one of the functions                                                           且 prefers_cross_layer_blocks 被设置为 True
        {register_kv_caches, register_cross_layers_kv_cache} will be called.            在 {register_kv_caches, register_cross_layers_kv_cache} 这两个函数中，只会调用其中一个。
     

        Args:
            kv_cache: a cross-layers kv cache tensor                                    一个跨层的kv cache张量
            attn_backend: The attention backend that corresponds to all layers          对应所有曾使用的attention backend
        """
        return

    def set_host_xfer_buffer_ops(self, copy_operation: CopyBlocksOp):
        """
        Set the xPU-specific ops for copying KV between host and device.                设置用于在 host（CPU）和 device（GPU/xPU）之间拷贝 KV cache 的专用操作。
        Needed when host buffer is used for kv transfer (e.g., in NixlConnector)
        """
        return

    @abstractmethod
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """
        Start loading the KV cache from the connector to vLLM's paged                  开始从connector中加载kv cache, 并写入vllm的分页kv buffer 
        KV buffer. This is called from the forward context before the                  这个过程发生在forward pass之前,由forward context调用
        forward pass to enable async loading during model execution.                   目的是在模型执行期间支持 异步kv加载

        Args:
            forward_context (ForwardContext): the forward context.                     当前forward执行的上下文信息
            **kwargs: additional arguments for the load operation                      包含本次推理所需的状态 request信息等

        Note:
            The number of elements in kv_caches and layer_names should be
            the same.

        """
        pass

    @abstractmethod
    def wait_for_layer_load(self, layer_name: str) -> None:
        """
        Block until the KV for a specific layer is loaded into vLLM's
        paged buffer. This is called from within attention layer to ensure
        async copying from start_load_kv is complete.

        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        pass

    @abstractmethod
    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        """
        Start saving a layer of KV cache from vLLM's paged buffer
        to the connector. This is called from within attention layer to
        enable async copying during execution.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """
        pass

    @abstractmethod
    def wait_for_save(self):
        """
        Block until all the save operations is done. This is called
        as the forward context exits to ensure that the async saving
        from save_kv_layer is complete before finishing the forward.

        This prevents overwrites of paged KV buffer before saving done.
        """
        pass

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens on the worker.
        The scheduler process (via the Executors) will use this output
        to track which workers are done.

        Returns:
            ids of requests that have finished asynchronous transfer
            (requests that previously returned True from request_finished()),
            tuple of (sending/saving ids, recving/loading ids).
            The finished saves/sends req ids must belong to a set provided in a
            call to this method (this call or a prior one).
        """
        return None, None

    def get_block_ids_with_load_errors(self) -> set[int]:
        """
        Get the set of block IDs that failed to load.

        Returns:
            Set of block IDs that encountered load errors.
            Empty set if no load errors occurred.

        Notes:
            - Applies to both sync- and async-loading requests.
            - Async loading: failed blocks may be reported in any forward pass
              up to and including the pass where the request ID is returned by
              `get_finished()`. Even if failures occur, the request must still
              be reported via `get_finished()`, and the failed block IDs must
              appear here no later than that same pass.
            - Sync loading: failed blocks should be reported in the forward
              pass in which they are detected.
        """
        return set()

    def shutdown(self):
        """
        Shutdown the connector. This is called when the worker process
        is shutting down to ensure that all the async operations are
        completed and the connector is cleaned up properly.
        """
        return None

    def get_kv_connector_stats(self) -> Optional["KVConnectorStats"]:
        """
        Get the KV connector stats collected during the last interval.
        """
        return None

    def get_kv_connector_kv_cache_events(self) -> Optional["KVConnectorKVEvents"]:
        """
        Get the KV connector kv cache events collected during the last interval.
        This function should be called by the model runner every time after the
        model execution and before cleanup.
        """
        return None

    def get_handshake_metadata(self) -> KVConnectorHandshakeMetadata | None:
        """
        Get the KVConnector handshake metadata for this connector.
        This metadata is used for out-of-band connector handshake
        between P/D workers.

        Returns:
            KVConnectorHandshakeMetadata: the handshake metadata.
            None if no handshake metadata is available.
        """
        return None

    # ==============================
    # Scheduler-side methods
    # ==============================

    @abstractmethod
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """
        Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            A tuple with the following elements:
                - An optional number of tokens that can be loaded from the
                  external KV cache beyond what is already computed.
                  If None, it means that the connector needs more time to
                  determine the number of matched tokens, and the scheduler
                  should query for this request again later.
                - `True` if external KV cache tokens will be loaded
                  asynchronously (between scheduler steps). Must be
                  'False' if the first element is 0.

        Notes:
            The connector should only consider the largest prefix of prompt-
            tokens for which KV cache is actually available at the time of the
            call. If the cache cannot be loaded for some tokens (e.g., due to
            connectivity issues or eviction), those tokens must not be taken
            into account.
        """
        pass

    @abstractmethod
    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        """
        Update KVConnector state after block allocation.

        If get_num_new_matched_tokens previously returned True for a
        request, this function may be called twice for that same request -
        first when blocks are allocated for the connector tokens to be
        asynchronously loaded into, and second when any additional blocks
        are allocated, after the load/transfer is complete.

        Args:
            request (Request): the request object.
            blocks (KVCacheBlocks): the blocks allocated for the request.
            num_external_tokens (int): the number of tokens that will be
                loaded from the external KV cache.
        """
        pass

    @abstractmethod
    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """
        Build the connector metadata for this step.

        This function should NOT modify fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        pass

    def update_connector_output(self, connector_output: KVConnectorOutput):
        """
        Update KVConnector state from worker-side connectors output.

        Args:
            connector_output (KVConnectorOutput): the worker-side
                connectors output.
        """
        return

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Called exactly once when a request has finished, before its blocks are
        freed.

        The connector may assumes responsibility for freeing the blocks
        asynchronously by returning True.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """
        return False, None

    def take_events(self) -> Iterable["KVCacheEvent"]:
        """
        Take the KV cache events from the connector.

        Yields:
            New KV cache events since the last call.
        """
        return ()

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: "VllmConfig") -> str | None:
        """
        Get the required KV cache layout for this connector.
        Args:
            vllm_config (VllmConfig): the vllm config.

        Returns:
            str: the required KV cache layout. e.g. HND, or NHD.
            None if the connector does not require a specific layout.
        """

        if cls is KVConnectorBase_V1:
            raise TypeError(
                "get_required_kvcache_layout should not be called "
                "on the abstract base class"
            )
        return None

    def get_finished_count(self) -> int | None:
        """
        Get the count of requests expected to complete send/receive operations   获取预计会通过connector完成发送/接收操作的请求数量.该方法用于初始化KVOutputAggregator
        via this connector. This method is used to initialize the                并会覆盖默认的world-size 
        KVOutputAggregator, overwriting the default world_size.

        Returns:
            int: expected sending or receiving completion count.
        """

        return None

    @classmethod
    def build_kv_connector_stats(
        cls, data: dict[str, Any] | None = None
    ) -> Optional["KVConnectorStats"]:
        """
        KVConnectorStats resolution method. This method allows dynamically
        registered connectors to return their own KVConnectorStats object,
        which can implement custom aggregation logic on the data dict.
        """
        return None

    def set_xfer_handshake_metadata(
        self, metadata: dict[int, KVConnectorHandshakeMetadata]
    ) -> None:
        """
        Set the KV connector handshake metadata for this connector.

        Args:
            metadata (KVConnectorHandshakeMetadata): the handshake metadata to set.
        """
        return None

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config: "VllmConfig",
        metric_types: dict[type["PromMetric"], type["PromMetricT"]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> Optional["KVConnectorPromMetrics"]:
        """
        Create a KVConnectorPromMetrics subclass which should register
        per-connector Prometheus metrics and implement observe() to
        expose connector transfer stats via Prometheus.
        """
        return None

    def reset_cache(self) -> bool | None:
        """
        Reset the connector's internal cache.

        Returns:
            bool: True if the cache was successfully reset, False otherwise.
        """
        logger.debug(
            "Connector cache reset requested, but %s does not implement reset_cache().",
            type(self).__name__,
        )

        return None
