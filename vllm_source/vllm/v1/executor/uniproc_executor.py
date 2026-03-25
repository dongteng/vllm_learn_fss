# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from functools import cached_property
from multiprocessing import Lock
from typing import Any

import torch
import torch.distributed as dist

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.utils.network_utils import get_distributed_init_method, get_ip, get_open_port
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.engine import ReconfigureDistributedRequest, ReconfigureRankType
from vllm.v1.executor.abstract import Executor
from vllm.v1.outputs import AsyncModelRunnerOutput, DraftTokenIds, ModelRunnerOutput
from vllm.v1.serial_utils import run_method
from vllm.v1.worker.worker_base import WorkerWrapperBase

logger = init_logger(__name__)


class UniProcExecutor(Executor):
    """
    单进程+ 单GPU的执行器
    可以理解为：把分布式Executor的接口，在单机模式下做一层封装
    特点：
        1.没有真正的PRC/多进程  2.但接口保持一致（方便切换到分布式） 3.所有调用都会落到driver_worker上执行
    """
    def _init_executor(self) -> None:
        """Initialize the worker and load the model. 初始化执行器（核心：创建 worker + 加载模型）
        整体流程：1.创建driver_worker(真正干活的对象) 2初始化分布式参数（即使是单机也要伪造）3初始化worker(环境/设备)  4加载模型（权重）
        """
        # === 1. 创建 driver worker ===driver_worker  是真正干活的对象（封装了模型、设备和推理逻辑）。rpc_rank=0：单进程模式下 rank 固定为 0。
        self.driver_worker = WorkerWrapperBase(vllm_config=self.vllm_config, rpc_rank=0)
        # === 2. 获取分布式参数（伪分布式）===
        distributed_init_method, rank, local_rank = self._distributed_args()
        # === 3. 构造 worker 初始化参数 ===
        kwargs = dict(
            vllm_config=self.vllm_config,
            local_rank=local_rank,                              #GPU id
            rank=rank,                                          #全局rank
            distributed_init_method=distributed_init_method,
            is_driver_worker=True,                              #标记为主worker
            shared_worker_lock=Lock(),                          #一个共享锁（单线程安全）
        )
        # === 4. 异步输出线程（用于非阻塞执行）===
        self.async_output_thread: ThreadPoolExecutor | None = None
        if self.max_concurrent_batches > 1: # 用线程池处理异步 output（避免阻塞主线程）
            self.async_output_thread = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="WorkerAsyncOutput"
            )

        self.driver_worker.init_worker(all_kwargs=[kwargs]) # === 5. 初始化 worker ===  → 初始化分布式环境（即使是假的）  # → 初始化内部状态
        self.driver_worker.init_device()                    # === 6. 初始化设备 ===#设置 cuda device、创建 stream、检查显存等
        self.driver_worker.load_model()                     # === 7. 加载模型 ===#真正加载模型权重、分片、量化、移动到GPU

    def _distributed_args(self) -> tuple[str, int, int]:
        """Return (distributed_init_method, rank, local_rank).构造“伪分布式”参数
        get_ip() → 获取当前机器 IP，比如 "192.168.1.10"
        get_open_port() → 随机找到一个空闲端口，比如 29500
        get_distributed_init_method(ip, port) → 返回一个字符串，告诉分布式框架初始化方法，类似 "tcp://192.168.1.10:29500"
        """
        distributed_init_method = get_distributed_init_method(get_ip(), get_open_port())
        # set local rank as the device index if specified
        device_info = self.vllm_config.device_config.device.__str__().split(":")
        local_rank = int(device_info[1]) if len(device_info) > 1 else 0
        return distributed_init_method, 0, local_rank

    @cached_property
    def max_concurrent_batches(self) -> int:
        """
        最大并发batch数，async_scheduling=True → 允许 overlap（返回2），否则 → 串行执行（1）
        """
        return 2 if self.scheduler_config.async_scheduling else 1

    def collective_rpc(  # type: ignore[override]
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: bool = False, #False为阻塞调用，True为非阻塞（异步）
        single_value: bool = False, #如果只有一个返回值，就直接返回，否则返回列表
    ) -> Any:
        """
        统一的伪rpc调用入口，本质是在单机模式下模拟分布式RPC，
        做的事情：1.调用driver_worker的run_method方法，2.可选同步/异步 3.统一返回格式（list或单值）
        """
        if kwargs is None:
            kwargs = {}
        # === 1. 同步调用 ===
        if not non_block:
            result = run_method(self.driver_worker, method, args, kwargs)
            return result if single_value else [result]
        # === 2. 异步调用 ===
        try:
            result = run_method(self.driver_worker, method, args, kwargs)
            #这里 result 有两种情况：
            # 普通值（比如张量、数字）
            # === 2.1 如果是异步输出对象 ===
            if isinstance(result, AsyncModelRunnerOutput):
                if (async_thread := self.async_output_thread) is not None:
                    if single_value: #单值返回
                        return async_thread.submit(result.get_output)

                    def get_output_list() -> list[Any]: # 多值返回
                        return [result.get_output()]

                    return async_thread.submit(get_output_list)
                # fallback：直接同步取结果
                result = result.get_output()
            # === 2.2 包装成 Future ===
            future = Future[Any]() ## 创建一个 Future 对象
            future.set_result(result if single_value else [result]) # # 直接把结果塞进去
        except Exception as e:
            future = Future[Any]()
            future.set_exception(e)
        return future

    def execute_model(  # type: ignore[override]
        self, scheduler_output: SchedulerOutput, non_block: bool = False
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        return self.collective_rpc(
            "execute_model",
            args=(scheduler_output,),
            non_block=non_block,
            single_value=True,
        )

    def sample_tokens(  # type: ignore[override]
        self, grammar_output: GrammarOutput | None, non_block: bool = False
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        return self.collective_rpc(
            "sample_tokens",
            args=(grammar_output,),
            non_block=non_block,
            single_value=True,
        )

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        return self.collective_rpc("take_draft_token_ids", single_value=True)

    def check_health(self) -> None:
        # UniProcExecutor will always be healthy as long as
        # it's running.
        return

    def reinitialize_distributed(
        self, reconfig_request: ReconfigureDistributedRequest
    ) -> None:
        self.driver_worker.reinitialize_distributed(reconfig_request)
        if (
            reconfig_request.new_data_parallel_rank
            == ReconfigureRankType.SHUTDOWN_CURRENT_RANK
        ):
            self.shutdown()

    def shutdown(self) -> None:
        if worker := self.driver_worker:
            worker.shutdown()


class ExecutorWithExternalLauncher(UniProcExecutor):
    """An executor that uses external launchers to launch engines,
    specially designed for torchrun-compatible launchers, for
    offline inference with tensor parallelism.

    see https://github.com/vllm-project/vllm/issues/11400 for
    the motivation, and examples/offline_inference/torchrun_example.py
    for the usage example.

    The key idea: although it is tensor-parallel inference, we only
    create one worker per executor, users will launch multiple
    engines with torchrun-compatible launchers, and all these engines
    work together to process the same prompts. When scheduling is
    deterministic, all the engines will generate the same outputs,
    and they don't need to synchronize the states with each other.
    """

    def _init_executor(self) -> None:
        """Initialize the worker and load the model."""
        assert not envs.VLLM_ENABLE_V1_MULTIPROCESSING, (
            "To get deterministic execution, "
            "please set VLLM_ENABLE_V1_MULTIPROCESSING=0"
        )
        super()._init_executor()

    def _distributed_args(self) -> tuple[str, int, int]:
        # engines are launched in torchrun-compatible launchers
        # so we can use the env:// method.
        # required env vars:
        # - RANK
        # - LOCAL_RANK
        # - MASTER_ADDR
        # - MASTER_PORT
        distributed_init_method = "env://"
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        return distributed_init_method, rank, local_rank

    def determine_available_memory(self) -> list[int]:  # in bytes
        # we need to get the min across all ranks.
        memory = super().determine_available_memory()
        from vllm.distributed.parallel_state import get_world_group

        cpu_group = get_world_group().cpu_group
        memory_tensor = torch.tensor([memory], device="cpu", dtype=torch.int64)
        dist.all_reduce(memory_tensor, group=cpu_group, op=dist.ReduceOp.MIN)
        return [memory_tensor.item()]
