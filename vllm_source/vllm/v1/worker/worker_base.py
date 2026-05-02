# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

import torch
import torch.nn as nn

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import worker_receiver_cache_from_config
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.utils.system_utils import update_environment_variables
from vllm.v1.kv_cache_interface import KVCacheSpec
from vllm.v1.serial_utils import run_method

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput
else:
    SchedulerOutput = object
    GrammarOutput = object
    AsyncModelRunnerOutput = object
    ModelRunnerOutput = object

logger = init_logger(__name__)

_R = TypeVar("_R")


class WorkerBase:
    """Worker interface that allows vLLM to cleanly separate implementations for
    different hardware. Also abstracts control plane communication, e.g., to
    communicate request metadata to other workers.
    Worker的基类（接口类），用于实现vLLM对不同硬件的后端抽象。
    主要作用：
    1.统一不同硬件（CPU GPU TPU）的Worker实现接口
    2.抽象控制平面通信（control plane communication），比如在不同worker之间传递请求元数据（request metadata）等。
    3.作为所有Worker实现的基类，定义了Worker需要实现的核心方法（如init_device、execute_model、sample_tokens等），以及一些通用方法（如get_kv_cache_spec、add_lora等）。
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ) -> None:
        """
        Initialize common worker components.

        Args:
            vllm_config: Complete vLLM configuration                    vLLM 的完整配置对象，包含模型、并行、调度、缓存等所有配置
            local_rank: Local device index                              本地设备编号（当前进程在本地机器上的 rank，常用于多 GPU）
            rank: Global rank in distributed setup                      
            distributed_init_method: Distributed initialization method  分布式初始化方法（如 "env://", "tcp://..." 等）
            is_driver_worker: Whether this worker handles driver        是否为 Driver Worker（主 Worker），负责协调和最终结果收集
                responsibilities
        """
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config
        self.kv_transfer_config = vllm_config.kv_transfer_config
        self.compilation_config = vllm_config.compilation_config

        from vllm.platforms import current_platform

        self.current_platform = current_platform

        #分布式相关信息
        self.parallel_config.rank = rank
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker

        # Device and model state
        self.device: torch.device | None = None
        self.model_runner: nn.Module | None = None

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """Get specifications for KV cache implementation.返回当前 Worker 所使用的 KV Cache 规格信息（块大小、数据类型等）"""
        raise NotImplementedError

    def compile_or_warm_up_model(self) -> None:
        """Prepare model for execution through compilation/warmup.编译或预热模型（用于 torch.compile、CUDA Graph 等优化）"""
        raise NotImplementedError

    def check_health(self) -> None:
        """Basic health check (override for device-specific checks).基础健康检查，子类可重写实现设备特定的健康检查逻辑"""
        return

    def init_device(self) -> None:
        """Initialize device state, such as loading the model or other on-device  始化设备相关状态，包括加载模型、分配显存等关键操作
        memory allocations.
        """
        raise NotImplementedError

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        """Initialize the KV cache with the given size in blocks.根据给定的 GPU/CPU block 数量初始化 KV Cache"""
        raise NotImplementedError

    def reset_mm_cache(self) -> None:
        """重置多模态（Multi-Modal）缓存，例如图像、视频等嵌入缓存"""
        reset_fn = getattr(self.model_runner, "reset_mm_cache", None)
        if callable(reset_fn):
            reset_fn()

    def get_model(self) -> nn.Module: 
        """获取当前 Worker 加载的模型实例"""
        raise NotImplementedError

    def apply_model(self, fn: Callable[[nn.Module], _R]) -> _R:
        """Apply a function on the model inside this worker.在当前 Worker 的模型上应用传入的函数，常用于模型参数修改等操作"""
        return fn(self.get_model())

    def load_model(self) -> None:
        """Load model onto target device. 将模型加载到目标设备（GPU/CPU 等）"""
        raise NotImplementedError

    def execute_model(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput | None:
        """If this method returns None, sample_tokens should be called immediately after
        to obtain the ModelRunnerOutput.

        Note that this design may be changed in future if/when structured outputs
        parallelism is re-architected.
        执行模型前向计算（forward）。
        注意：
        - 如果该方法返回 None，则必须立即调用 sample_tokens() 来获取最终输出。
        - 这种设计主要为了支持某些特殊并行策略（未来结构化输出并行可能改变此设计）。
        """
        raise NotImplementedError

    def sample_tokens(
        self, grammar_output: GrammarOutput
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        """Should be called immediately after execute_model iff it returned None.执行采样（sampling）操作。
        仅当 execute_model() 返回 None 时才需要调用此方法。"""
        raise NotImplementedError

    def get_cache_block_size_bytes(self) -> int:
        """Return the size of a single cache block, in bytes. Used in
        speculative decoding.回单个 KV Cache block 的大小（字节），主要用于 speculative decoding 计算
        """
        raise NotImplementedError

    def add_lora(self, lora_request: LoRARequest) -> bool:
        raise NotImplementedError

    def remove_lora(self, lora_id: int) -> bool:
        raise NotImplementedError

    def pin_lora(self, lora_id: int) -> bool:
        raise NotImplementedError

    def list_loras(self) -> set[int]:
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size from model configuration.获取模型的词表大小（vocabulary size）。"""
        return self.model_config.get_vocab_size()

    def shutdown(self) -> None:
        """Clean up resources held by the worker.清理 Worker 占用的资源（模型、缓存、进程等）。"""
        return


class WorkerWrapperBase:
    """
    This class represents one process in an executor/engine. It is responsible
    for lazily initializing the worker and handling the worker's lifecycle.
    We first instantiate the WorkerWrapper, which remembers the worker module
    and class name. Then, when we call `update_environment_variables`, and the
    real initialization happens in `init_worker`.
    这个类表示执行器/引擎中的 一个进程。它负责延迟初始化 worker 并管理 worker 的生命周期。
    我们首先创建 WorkerWrapper 实例，它会记住 worker 模块和类名。
    然后，当调用 update_environment_variables 时，会设置环境变量，真正的初始化工作在 init_worker 方法中完成。
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        rpc_rank: int = 0,
        global_rank: int | None = None,
    ) -> None:
        """
        Initialize the worker wrapper with the given vllm_config and rpc_rank.
        Note: rpc_rank is the rank of the worker in the executor. In most cases,
        it is also the rank of the worker in the distributed group. However,
        when multiple executors work together, they can be different.
        e.g. in the case of SPMD-style offline inference with TP=2,
        users can launch 2 engines/executors, each with only 1 worker.
        All workers have rpc_rank=0, but they have different ranks in the TP
        group.
        使用给定的 vllm_config 和 rpc_rank 初始化 worker wrapper。
        注意：rpc_rank 表示 worker 在 executor 中的编号（rank）。在大多数情况下，它也等于 worker 在分布式组中的编号。但是，当多个 executor 协同工作时，这两个编号可能不同。

        例如，在 SPMD 风格的离线推理（TP=2） 场景下，用户可以启动 2 个引擎/执行器，每个执行器只有 1 个 worker。
        所有 worker 的 rpc_rank=0，但它们在 TP（张量并行）组中的 rank 是不同的。
        """
        self.rpc_rank = rpc_rank
        self.global_rank = self.rpc_rank if global_rank is None else global_rank #worker 在整个分布式组中的编号，如果没有传就用 rpc_rank
        self.worker: WorkerBase | None = None

        # do not store this `vllm_config`, `init_worker` will set the final
        # one.
        # TODO: investigate if we can remove this field in `WorkerWrapperBase`,
        # `init_cached_hf_modules` should be unnecessary now.
        self.vllm_config: VllmConfig | None = None

        # `model_config` can be None in tests
        model_config = vllm_config.model_config
        if model_config and model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils.import_utils import init_cached_hf_modules

            init_cached_hf_modules()

    def shutdown(self) -> None:
        if self.worker is not None:
            self.worker.shutdown()

    def adjust_rank(self, rank_mapping: dict[int, int]) -> None:
        """
        Adjust the rpc_rank based on the given mapping.
        It is only used during the initialization of the executor,
        to adjust the rpc_rank of workers after we create all workers.
        """
        if self.rpc_rank in rank_mapping:
            self.rpc_rank = rank_mapping[self.rpc_rank]

    def update_environment_variables(
        self,
        envs_list: list[dict[str, str]],
    ) -> None:
        envs = envs_list[self.rpc_rank]
        key = "CUDA_VISIBLE_DEVICES"
        if key in envs and key in os.environ:
            # overwriting CUDA_VISIBLE_DEVICES is desired behavior
            # suppress the warning in `update_environment_variables`
            del os.environ[key]
        update_environment_variables(envs)

    def init_worker(self, all_kwargs: list[dict[str, Any]]) -> None:
        """
        Here we inject some common logic before initializing the worker.
        Arguments are passed to the worker class constructor.
        """
        kwargs = all_kwargs[self.rpc_rank]
        self.vllm_config = kwargs.get("vllm_config")
        assert self.vllm_config is not None, (
            "vllm_config is required to initialize the worker"
        )
        self.vllm_config.enable_trace_function_call_for_thread() #开启线程级函数调用 trace。可以理解为：给当前 worker 打开调试/跟踪能力，方便记录调用链。这通常用于 profiling 或调试。

        from vllm.plugins import load_general_plugins

        load_general_plugins() #加载通用插件 这些插件可能是：自定义采样、自定义调度、日志增强、量化支持... 等用户可能想插入的东西

        if isinstance(self.vllm_config.parallel_config.worker_cls, str):
            #检查配置里指定的 worker 类是不是字符串（比如 "vllm.worker.Worker" 这种写法）
            #如果是字符串->通过字符串找到真正的类（动态导入）
            worker_class = resolve_obj_by_qualname(
                self.vllm_config.parallel_config.worker_cls
            )
        else:
            raise ValueError(
                "passing worker_cls is no longer supported. Please pass keep the class in a separate module and pass the qualified name of the class as a string."  # noqa: E501
            )
        if self.vllm_config.parallel_config.worker_extension_cls:
            #如果配置里还额外指定了一个扩展类（worker_extension_cls）
            worker_extension_cls = resolve_obj_by_qualname(
                self.vllm_config.parallel_config.worker_extension_cls
            )
            extended_calls = []
            if worker_extension_cls not in worker_class.__bases__:
                # check any conflicts between worker and worker_extension_cls
                for attr in dir(worker_extension_cls):
                    if attr.startswith("__"):
                        continue
                    assert not hasattr(worker_class, attr), (
                        f"Worker class {worker_class} already has an attribute"
                        f" {attr}, which conflicts with the worker"
                        f" extension class {worker_extension_cls}."
                    )
                    if callable(getattr(worker_extension_cls, attr)):
                        extended_calls.append(attr)
                # dynamically inherit the worker extension class
                #动态给原始类添加一个父类（多继承）当于让原来的 Worker 类动态继承了扩展类
                worker_class.__bases__ = worker_class.__bases__ + (
                    worker_extension_cls,
                )
                logger.info(
                    "Injected %s into %s for extended collective_rpc calls %s",
                    worker_extension_cls,
                    worker_class,
                    extended_calls,
                )

        shared_worker_lock = kwargs.pop("shared_worker_lock", None)
        if shared_worker_lock is None:
            msg = (
                "Missing `shared_worker_lock` argument from executor. "
                "This argument is needed for mm_processor_cache_type='shm'."
            )

            mm_config = self.vllm_config.model_config.multimodal_config
            if mm_config and mm_config.mm_processor_cache_type == "shm":
                raise ValueError(msg)
            else:
                logger.warning_once(msg)

            self.mm_receiver_cache = None
        else:
            self.mm_receiver_cache = worker_receiver_cache_from_config(
                self.vllm_config,
                MULTIMODAL_REGISTRY,
                shared_worker_lock,
            )

        with set_current_vllm_config(self.vllm_config):
            # To make vLLM config available during worker initialization
            self.worker = worker_class(**kwargs)
            assert self.worker is not None

    def initialize_from_config(self, kv_cache_configs: list[Any]) -> None:
        kv_cache_config = kv_cache_configs[self.global_rank]
        assert self.vllm_config is not None
        with set_current_vllm_config(self.vllm_config):
            self.worker.initialize_from_config(kv_cache_config)  # type: ignore

    def init_device(self):
        assert self.vllm_config is not None
        with set_current_vllm_config(self.vllm_config):
            # To make vLLM config available during device initialization
            self.worker.init_device()  # type: ignore

    def execute_method(self, method: str | bytes, *args, **kwargs):
        try:
            # method resolution order:
            # if a method is defined in this class, it will be called directly.
            # otherwise, since we define `__getattr__` and redirect attribute
            # query to `self.worker`, the method will be called on the worker.
            ## 方法解析顺序：
            # 如果这个方法在当前类中定义了，就直接调用。否则，因为我们实现了 __getattr__，
            # 并且把属性查询重定向到了 self.worker，那么这个方法就会在 worker 上执行。
            #举例，假设method ="generate",args=("hello",) ,  实际执行就类似self.generate("hello")，如果self没有generate方法，那么就self.worker.generate("hello")
            #这是通过__getattr__自动转发实现的
            return run_method(self, method, args, kwargs)
        except Exception as e:
            # if the driver worker also execute methods,
            # exceptions in the rest worker may cause deadlock in rpc like ray
            # see https://github.com/vllm-project/vllm/issues/3455
            # print the error and inform the user to solve the error
            msg = (
                f"Error executing method {method!r}. "
                "This might cause deadlock in distributed execution."
            )
            logger.exception(msg)
            raise e

    def __getattr__(self, attr: str):
        return getattr(self.worker, attr)

    def _apply_mm_cache(self, scheduler_output: SchedulerOutput) -> None:
        mm_cache = self.mm_receiver_cache
        if mm_cache is None:
            return

        for req_data in scheduler_output.scheduled_new_reqs:
            req_data.mm_features = mm_cache.get_and_update_features(
                req_data.mm_features
            )

    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        *args,
        **kwargs,
    ) -> ModelRunnerOutput | None:
        self._apply_mm_cache(scheduler_output)

        assert self.worker is not None
        return self.worker.execute_model(scheduler_output, *args, **kwargs)

    def reset_mm_cache(self) -> None:
        mm_receiver_cache = self.mm_receiver_cache
        if mm_receiver_cache is not None:
            mm_receiver_cache.clear_cache()

        assert self.worker is not None
        self.worker.reset_mm_cache()
