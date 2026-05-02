# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import queue
import signal
import threading
import time
from collections import deque
from collections.abc import Callable, Generator
from concurrent.futures import Future
from contextlib import ExitStack, contextmanager
from inspect import isclass, signature
from logging import DEBUG
from typing import Any, TypeVar, cast

import msgspec
import zmq

from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed import stateless_destroy_torch_distributed_process_group
from vllm.envs import enable_envs_cache
from vllm.logger import init_logger
from vllm.logging_utils.dump_input import dump_engine_exception
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import engine_receiver_cache_from_config
from vllm.tasks import POOLING_TASKS, SupportedTask
from vllm.transformers_utils.config import maybe_register_config_serialize_by_value
from vllm.utils.gc_utils import (
    freeze_gc_heap,
    maybe_attach_gc_debug_callback,
)
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.utils.network_utils import make_zmq_socket
from vllm.utils.system_utils import decorate_logs, set_process_title
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    generate_scheduler_kv_cache_config,
    get_kv_cache_configs,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine import (
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequest,
    EngineCoreRequestType,
    FinishReason,
    ReconfigureDistributedRequest,
    ReconfigureRankType,
    UtilityOutput,
    UtilityResult,
)
from vllm.v1.engine.utils import (
    EngineHandshakeMetadata,
    EngineZmqAddresses,
    get_device_indices,
)
from vllm.v1.executor import Executor
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
from vllm.v1.structured_output import StructuredOutputManager
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger(__name__)

POLLING_TIMEOUT_S = 2.5
HANDSHAKE_TIMEOUT_MINS = 5

_R = TypeVar("_R")  # Return type for collective_rpc


class EngineCore:
    """Inner loop of vLLM's Engine.
    这是 vLLM v1 架构中最重要、最核心的类。
    它负责统筹整个推理过程：接收请求、调度请求、决定什么时候跑模型、管理 KV Cache 等。
    可以把它想象成一个“大管家”。
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        executor_fail_callback: Callable | None = None,
    ):
        # plugins need to be loaded at the engine/scheduler level too           加载各种插件
        from vllm.plugins import load_general_plugins

        load_general_plugins()

        self.vllm_config = vllm_config
        if vllm_config.parallel_config.data_parallel_rank == 0:                 #只在主进程打印初始化日志，避免重复输出
            logger.info(
                "Initializing a V1 LLM engine (v%s) with config: %s",
                VLLM_VERSION,
                vllm_config,
            )

        self.log_stats = log_stats

        # ====================== 1. 创建模型执行器 ======================
        # ModelExecutor 负责管理所有 Worker（GPU卡），真正跑模型的就是它
        # Setup Model.
        self.model_executor = executor_class(vllm_config)
        if executor_fail_callback is not None:                                  ## 如果传入了失败回调函数，则注册（当 Worker 崩溃时会调用）
            self.model_executor.register_failure_callback(executor_fail_callback)

        self.available_gpu_memory_for_kv_cache = -1

        # ====================== 2. 初始化 KV Cache（非常重要！） ======================
        # KV Cache 是大模型加速的核心技术，用于缓存 Attention 的中间结果
        # 这里会根据可用显存，计算出能用多少个 GPU block 和 CPU block
        # Setup KV Caches and update CacheConfig after profiling.
        num_gpu_blocks, num_cpu_blocks, kv_cache_config = self._initialize_kv_caches(
            vllm_config
        )

        vllm_config.cache_config.num_gpu_blocks = num_gpu_blocks
        vllm_config.cache_config.num_cpu_blocks = num_cpu_blocks
        self.collective_rpc("initialize_cache", args=(num_gpu_blocks, num_cpu_blocks))

        # ====================== 3. 结构化输出管理器 ======================
        # 用于支持 JSON 模式、工具调用等结构化输出功能
        self.structured_output_manager = StructuredOutputManager(vllm_config)
        
        # ====================== 4. 创建 Scheduler（调度器） ======================
        # Scheduler 是负责“决定哪些请求现在该跑模型”的关键组件
        # Setup scheduler.
        Scheduler = vllm_config.scheduler_config.get_scheduler_cls()

        # 如果模型不需要 KV Cache（比如纯 Encoder 模型），则关闭 chunked prefill
        if len(kv_cache_config.kv_cache_groups) == 0:  # noqa: SIM102
            # Encoder models without KV cache don't support
            # chunked prefill. But do SSM models?
            if vllm_config.scheduler_config.enable_chunked_prefill:
                logger.warning("Disabling chunked prefill for model without KVCache")
                vllm_config.scheduler_config.enable_chunked_prefill = False
        
        # 计算 scheduler 使用的 block 大小（考虑了上下文并行）
        scheduler_block_size = (
            vllm_config.cache_config.block_size
            * vllm_config.parallel_config.decode_context_parallel_size
            * vllm_config.parallel_config.prefill_context_parallel_size
        )
        # 创建调度器实例
        self.scheduler: SchedulerInterface = Scheduler(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=self.structured_output_manager,
            include_finished_set=vllm_config.parallel_config.data_parallel_size > 1,
            log_stats=self.log_stats,
            block_size=scheduler_block_size,
        )
        self.use_spec_decode = vllm_config.speculative_config is not None
        if self.scheduler.connector is not None:  # type: ignore                     # 如果使用了 KV Connector，则进行初始化
            self.model_executor.init_kv_output_aggregator(self.scheduler.connector)  # type: ignore

        # ====================== 5. 多模态相关（图片、视频等） ======================
        self.mm_registry = mm_registry = MULTIMODAL_REGISTRY
        self.mm_receiver_cache = engine_receiver_cache_from_config(
            vllm_config, mm_registry
        )

        
        # ====================== 6. KV Connector 握手元数据处理 ======================
        # 用于跨节点传输 KV Cache 时的握手信息（Disaggregated Prefill/Decode 架构使用）
        # If a KV connector is initialized for scheduler, we want to collect
        # handshake metadata from all workers so the connector in the scheduler
        # will have the full context
        kv_connector = self.scheduler.get_kv_connector()
        if kv_connector is not None:
            # Collect and store KV connector xfer metadata from workers
            # (after KV cache registration)
            xfer_handshake_metadata = (
                self.model_executor.get_kv_connector_handshake_metadata()
            )

            if xfer_handshake_metadata:
                # xfer_handshake_metadata is list of dicts from workers
                # Each dict already has structure {tp_rank: metadata}
                # Merge all worker dicts into a single dict
                content: dict[int, Any] = {}
                for worker_dict in xfer_handshake_metadata:
                    if worker_dict is not None:
                        content.update(worker_dict)
                kv_connector.set_xfer_handshake_metadata(content)

        # ====================== 7. Pipeline Parallelism 批量队列 ======================
        # 用于支持 Pipeline Parallelism，减少流水线气泡（pipeline bubble）
        # Setup batch queue for pipeline parallelism.
        # Batch queue for scheduled batches. This enables us to asynchronously
        # schedule and execute batches, and is required by pipeline parallelism
        # to eliminate pipeline bubbles.
        self.batch_queue_size = self.model_executor.max_concurrent_batches
        self.batch_queue: (
            deque[tuple[Future[ModelRunnerOutput], SchedulerOutput]] | None
        ) = None
        if self.batch_queue_size > 1:
            logger.info("Batch queue is enabled with size %d", self.batch_queue_size)
            self.batch_queue = deque(maxlen=self.batch_queue_size)

        # ====================== 8. 其他标志和配置 ======================
        self.is_ec_producer = (                                                             #EC = “KV cache 可以被外部系统接管、存储和迁移的一套扩展缓存传输体系”。
            vllm_config.ec_transfer_config is not None
            and vllm_config.ec_transfer_config.is_ec_producer
        )
        self.is_pooling_model = vllm_config.model_config.runner_type == "pooling"

        # 前缀缓存（Prefix Caching）用的哈希函数
        self.request_block_hasher: Callable[[Request], list[BlockHash]] | None = None
        if vllm_config.cache_config.enable_prefix_caching or kv_connector is not None:
            caching_hash_fn = get_hash_fn_by_name(                                           #根据配置选择hash算法例如 SHA1 / SHA256 / 自定义 hash）
                vllm_config.cache_config.prefix_caching_hash_algo
            )
            init_none_hash(caching_hash_fn)                                                  # 初始化 hash 函数（可能是注册默认状态 / 清理状态 / warmup）

            self.request_block_hasher = get_request_block_hasher(                            # 构造“请求 → block hash 列表”的生成器函数   最终效果:Request → [BlockHash1, BlockHash2, ...]
                scheduler_block_size, caching_hash_fn
            )

        # 根据是否启用 batch queue，选择不同的 step 函数
        self.step_fn = (
            self.step if self.batch_queue is None else self.step_with_batch_queue
        )
        self.async_scheduling = vllm_config.scheduler_config.async_scheduling

        # 用于存放被取消的请求
        self.aborts_queue = queue.Queue[list[str]]()

        
        # ====================== 9. GC 优化（垃圾回收） ======================
        # Mark the startup heap as static so that it's ignored by GC.           性能优化，减少python垃圾回收带来的卡顿
        # Reduces pause times of oldest generation collections.
        freeze_gc_heap()                                                        #把启动时的堆标记为静态，减少GC暂停
        # If enable, attach GC debugger after static variable freeze.
        maybe_attach_gc_debug_callback()                                        #如果开启了GC调试，则附加调试回调。就是如果你开起了某个环境变量，他会执行一个回调函数，打印GC出发时间 打印回收了多少对象，分析卡顿来源
        # Enable environment variable cache (e.g. assume no more
        # environment variable overrides after this point)
        enable_envs_cache()                                                     #缓存环境变量，假设之后不再修改，把环境变量读取结果缓存起来。

    def _initialize_kv_caches(
        self, vllm_config: VllmConfig
    ) -> tuple[int, int, KVCacheConfig]:
        """
        初始化kv cache(关键缓存)，主要工作：当前显存还能放下多少kv cache，并进行初始化
        """
        
        start = time.time()
        # ====================== 第一步：获取模型需要的 KV Cache 类型 ======================
        # 不同模型需要的 KV Cache 规格可能不一样（比如普通模型、MoE模型、多模态模型等）
        # Get all kv cache needed by the model
        kv_cache_specs = self.model_executor.get_kv_cache_specs()
        
        # 判断这个模型是否需要 KV Cache（有些模型如纯 Encoder 不需要），只要有任意一个layer需要kv cache，就认为这个模型需要kv cache
        has_kv_cache = any(kv_cache_spec for kv_cache_spec in kv_cache_specs)
        if has_kv_cache:
            # ------------------- 特殊启动模式（弹性扩容） -------------------
            if os.environ.get("VLLM_ELASTIC_EP_SCALE_UP_LAUNCH") == "1":
                # 在弹性扩容场景下，通过并行组同步获取可用显存
                dp_group = getattr(self, "dp_group", None)
                assert dp_group is not None
                self.available_gpu_memory_for_kv_cache = (
                    ParallelConfig.sync_kv_cache_memory_size(dp_group, -1)
                )
                # 把可用显存复制成和 kv_cache_specs 数量一致的列表
                available_gpu_memory = [self.available_gpu_memory_for_kv_cache] * len(
                    kv_cache_specs
                )
            else:
                # Profiles the peak memory usage of the model to determine how
                # much memory can be allocated for kv cache.
                # ------------------- 正常情况：性能分析（Profiling） -------------------
                # 通过实际跑一次模型，测量峰值显存占用，从而计算出还能分配多少显存给 KV Cache
                available_gpu_memory = self.model_executor.determine_available_memory()
                self.available_gpu_memory_for_kv_cache = available_gpu_memory[0]
        else:
            # Attention free models don't need memory for kv cache # 如果模型不需要 KV Cache（比如没有 Attention 的模型），则可用显存为 0
            available_gpu_memory = [0] * len(kv_cache_specs)

        # 确保规格数量和可用显存数量一致
        assert len(kv_cache_specs) == len(available_gpu_memory)

        # ====================== 第二步：生成 KV Cache 配置 ======================
        # 记录原始的 max_model_len（最大上下文长度），用于后面判断是否被自动调整
        # Track max_model_len before KV cache config to detect auto-fit changes
        max_model_len_before = vllm_config.model_config.max_model_len

        # 根据可用显存、模型配置、KV Cache 规格，计算最终的 KV Cache 配置
        # 这步非常重要：会决定能缓存多少个 block，以及是否要自动缩小 max_model_len
        kv_cache_configs = get_kv_cache_configs(
            vllm_config, kv_cache_specs, available_gpu_memory
        )

        # ====================== 第三步：处理 max_model_len 被自动调整的情况 ======================
        # 有时显存不够，vLLM 会自动把 max_model_len 调小（auto-fit）
        # If auto-fit reduced max_model_len, sync the new value to workers.
        # This is needed because workers were spawned before memory profiling
        # and have the original (larger) max_model_len cached.
        max_model_len_after = vllm_config.model_config.max_model_len
        
        # 如果 max_model_len 被改小了，需要通知所有 Worker 更新这个值
        # （因为 Worker 是提前启动的，可能还保存着旧的较大值）
        if max_model_len_after != max_model_len_before:
            self.collective_rpc("update_max_model_len", args=(max_model_len_after,))

        # ====================== 第四步：生成调度器使用的 KV Cache 配置 =====================
        scheduler_kv_cache_config = generate_scheduler_kv_cache_config(kv_cache_configs)
        num_gpu_blocks = scheduler_kv_cache_config.num_blocks                               ## GPU 上能用的 block 数量
        num_cpu_blocks = 0                                                                  ## CPU block 数量（目前 v1 暂不使用）                             

        # Initialize kv cache and warmup the execution
        # ====================== 第五步：真正初始化 KV Cache 并预热模型 ======================
        # 把上面计算好的配置应用到所有 Worker 上，真正分配显存并初始化 KV Cache
        # 同时会进行模型预热（warmup）
        self.model_executor.initialize_from_config(kv_cache_configs)

        elapsed = time.time() - start
        logger.info_once(
            "init engine (profile, create kv cache, warmup model) took %.2f seconds",
            elapsed,
            scope="local",
        )
        return num_gpu_blocks, num_cpu_blocks, scheduler_kv_cache_config                    # 返回三个关键值：GPU block 数、CPU block 数、调度器用的 KV Cache 配置

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.model_executor.supported_tasks

    def add_request(self, request: Request, request_wave: int = 0):
        """Add request to the scheduler.

        `request_wave`: indicate which wave of requests this is expected to
        belong to in DP case
        """
        # ====================== 1. 验证 request_id 类型 ======================
        # request_id 必须是字符串类型，这是 vLLM 的强制要求
        # Validate the request_id type.
        if not isinstance(request.request_id, str):
            raise TypeError(
                f"request_id must be a string, got {type(request.request_id)}"
            )

        # ====================== 2. 处理 Pooling 模型的特殊校验 ======================
        # Pooling 模型是指做 embedding、分类、rerank 等任务的模型（不是生成文本）
        if pooling_params := request.pooling_params:
            # 获取当前模型支持的所有 pooling 类任务（embedding, classify, score 等）
            supported_pooling_tasks = [
                task for task in self.get_supported_tasks() if task in POOLING_TASKS
            ]
            ## 如果请求的任务类型不在支持列表中，则报错
            if pooling_params.task not in supported_pooling_tasks:
                raise ValueError(
                    f"Unsupported task: {pooling_params.task!r} "
                    f"Supported tasks: {supported_pooling_tasks}"
                )
                
        # ====================== 3. KV Transfer 参数检查 ======================
        # KV Transfer 是用于 P/D 分离架构（Prefill 和 Decode 分开部署）的特性
        # 如果请求带了 kv_transfer_params，但当前引擎没有配置 KVConnector，则给出警告
        if request.kv_transfer_params is not None and (
            not self.scheduler.get_kv_connector()
        ):
            logger.warning(
                "Got kv_transfer_params, but no KVConnector found. "
                "Disabling KVTransfer for this request."
            )
        # ====================== 4. 真正把请求加入调度器 ======================
        # Scheduler 是负责请求排队、批处理决策的核心组件
        # 这里只是把请求交给 Scheduler，后续由 Scheduler 决定什么时候执行
        self.scheduler.add_request(request)

    def abort_requests(self, request_ids: list[str]):
        """Abort requests from the scheduler.
        把终止工作实际委托给了Scheduler
        """

        # TODO: The scheduler doesn't really need to know the       目前统一使用FINISHED_ABORTED状态
        # specific finish reason, TBD whether we propagate that     未来可能区分2中不同的终止原因：1客户端主动abort 2正常停止
        # (i.e. client-aborted vs stop criteria met).
        self.scheduler.finish_requests(request_ids, RequestStatus.FINISHED_ABORTED)

    @contextmanager
    def log_error_detail(self, scheduler_output: SchedulerOutput):
        """Execute the model and log detailed info on failure.   在执行模型前向计算时，提供详细的错误日志记录，就是说执行模型并在发生错误时记录详细调试信息"""
        try:
            yield
        except Exception as err:
            # We do not want to catch BaseException here since we're only
            # interested in dumping info when the exception is due to an
            # error from execute_model itself.

            # NOTE: This method is exception-free
            dump_engine_exception(
                self.vllm_config, scheduler_output, self.scheduler.make_stats()
            )
            raise err

    def _log_err_callback(self, scheduler_output: SchedulerOutput):
        """Log error details of a future that's not expected to return a result."""

        def callback(f, sched_output=scheduler_output):
            with self.log_error_detail(sched_output):
                result = f.result()
                assert result is None

        return callback

    def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
        """Schedule, execute, and make output.                                  单步执行引擎核心逻辑

        Returns tuple of outputs and a flag indicating whether the model        做了三件事：1.调度请求  2执行模型 3更新状态并产出结果
        was executed.
        """

        # Check for any requests remaining in the scheduler - unfinished,       返回：1.engine_core_outputs:每个request id对应的输出（token/finished等）
        # or finished and not yet removed from the batch.                             2.bool:本轮是否真的直行了模型（是否有token被调度）
        if not self.scheduler.has_requests():                                   #如果当前没有任何请求，直接返空
            return {}, False
        scheduler_output = self.scheduler.schedule()                            #scheduler阶段：决定哪些请求进入本轮执行；每个请求分配多少token(prefill/decode)；batch如何组织（动态batching）
        future = self.model_executor.execute_model(scheduler_output, non_block=True)#gpu_model_runner  #3.提交模型执行，non_block=True表示异步执行,不堵塞当前线程，返回一个future，提交后就开始执行了
        grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)   #grammer/约束信息
        with self.log_error_detail(scheduler_output):
            model_output = future.result()                                      #阻塞等待等待GPU推理完成
            if model_output is None:                                            #就是说如果返回一个完整的ModelRunnerOutput，里边已经包含了采样后的token
                model_output = self.model_executor.sample_tokens(grammar_output)#高性能优化场景下，excute_model故意只做前向计算，不做sampling，此时就会返回None

        # Before processing the model output, process any aborts that happened
        # during the model execution.
        self._process_aborts_queue()                                            #处理中途取消的请求，在模型执行期间，可能有请求被用户取消
        engine_core_outputs = self.scheduler.update_from_output(                #更新调度器状态（核心），给每个request追加新token，判断是否finished ，更新kv cache状态，生成返回给上层的数据结构
            scheduler_output, model_output
        )

        return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0 #返回2个值：1.engine_core_ouputs:处理后的结果（供上层使用），2是否真正了token,有些步只是调度但是没跑模型

    def post_step(self, model_executed: bool) -> None:
        # When using async scheduling we can't get draft token ids in advance,      执行step之后的后处理操作
        # so we update draft token ids in the worker process and don't              主要用于在模型推理完成后，处理推测解码相关的draft token ids
        # need to update draft token ids here.
        if not self.async_scheduling and self.use_spec_decode and model_executed:
            # Take the draft token ids.
            draft_token_ids = self.model_executor.take_draft_token_ids()
            if draft_token_ids is not None:
                self.scheduler.update_draft_token_ids(draft_token_ids)

    def step_with_batch_queue(
        self,
    ) -> tuple[dict[int, EngineCoreOutputs] | None, bool]:
        """Schedule and execute batches with the batch queue.                   使用batch queue进行调度和模型执行
        Note that if nothing to output in this step, None is returned.          这是pipline parallelism（流水线并行）场景下的核心执行函数，负责调度和执行模型，并且管理 batch queue。
                                                                                主要目标是让调度新请求和执行模型计算能够异步重叠进行，减少流水线气泡（pipeline bubble），提高吞吐量。
        The execution flow is as follows:                                         
        1. Try to schedule a new batch if the batch queue is not full.           1.如果队列还没满，就尝试调度一个新的batch,如果成功调度了新的batch，就直接返回一个空的engine输出
        If a new batch is scheduled, directly return an empty engine core
        output. In other words, fulfilling the batch queue has a higher priority
        than getting model outputs.
        2. If there is no new scheduled batch, meaning that the batch queue      2.如果没有新的batch被调度（也就是batch队列已经满了 或者没有更多请求可以加入）那么就会阻塞等待，知道任务队列中的第一个batch执行完成
        is full or no other requests can be scheduled, we block until the first
        batch in the job queue is finished.
        3. Update the scheduler from the output.                                 3.根据模型执行的输出结果，更新调度器的状态
        """
        batch_queue = self.batch_queue
        assert batch_queue is not None

        # Try to schedule a new batch if the batch queue is not full, but
        # the scheduler may return an empty batch if all requests are scheduled.
        # Note that this is not blocking.
        assert len(batch_queue) < self.batch_queue_size

        model_executed = False
        deferred_scheduler_output = None
        if self.scheduler.has_requests():
            scheduler_output = self.scheduler.schedule()
            exec_future = self.model_executor.execute_model(
                scheduler_output, non_block=True
            )
            if not self.is_ec_producer:
                model_executed = scheduler_output.total_num_scheduled_tokens > 0

            if self.is_pooling_model or not model_executed:
                # No sampling required (no requests scheduled).
                future = cast(Future[ModelRunnerOutput], exec_future)
            else:
                exec_future.add_done_callback(self._log_err_callback(scheduler_output))

                if not scheduler_output.pending_structured_output_tokens:
                    # We aren't waiting for any tokens, get any grammar output
                    # and sample immediately.
                    grammar_output = self.scheduler.get_grammar_bitmask(
                        scheduler_output
                    )
                    future = self.model_executor.sample_tokens(
                        grammar_output, non_block=True
                    )
                else:
                    # We need to defer sampling until we have processed the model output
                    # from the prior step.
                    deferred_scheduler_output = scheduler_output

            if not deferred_scheduler_output:
                # Add this step's future to the queue.
                batch_queue.appendleft((future, scheduler_output))
                if (
                    model_executed
                    and len(batch_queue) < self.batch_queue_size
                    and not batch_queue[-1][0].done()
                ):
                    # Don't block on next worker response unless the queue is full
                    # or there are no more requests to schedule.
                    return None, True

        elif not batch_queue:
            # Queue is empty. We should not reach here since this method should
            # only be called when the scheduler contains requests or the queue
            # is non-empty.
            return None, False

        # Block until the next result is available.
        future, scheduler_output = batch_queue.pop()
        with self.log_error_detail(scheduler_output):
            model_output = future.result()

        # Before processing the model output, process any aborts that happened
        # during the model execution.
        self._process_aborts_queue()
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )

        # NOTE(nick): We can either handle the deferred tasks here or save
        # in a field and do it immediately once step_with_batch_queue is
        # re-called. The latter slightly favors TTFT over TPOT/throughput.
        if deferred_scheduler_output:
            # We now have the tokens needed to compute the bitmask for the
            # deferred request. Get the bitmask and call sample tokens.
            grammar_output = self.scheduler.get_grammar_bitmask(
                deferred_scheduler_output
            )
            future = self.model_executor.sample_tokens(grammar_output, non_block=True)
            batch_queue.appendleft((future, deferred_scheduler_output))

        return engine_core_outputs, model_executed

    def _process_aborts_queue(self):
        """
        处理abort请求队列（取消请求队列） 当用户中途取消请求（例如点击停止生成、超时、客户端断开链接等）这些信号不会立即中断正在进行的GPU计算
        而是先放入一个队列，然后由Engine的主循环定期调用这个函数进行统一处理。
        
        这个函数的作用是，把队列中积累的所有abort请求一次性批量处理掉
        """
        
        if not self.aborts_queue.empty():
            request_ids = []
            while not self.aborts_queue.empty():
                ids = self.aborts_queue.get_nowait()            #非阻塞获取，如果队列为空立即抛出EMPTY异常
                if isinstance(ids, str):
                    # Should be a list here, but also handle string just in case.
                    ids = (ids,)
                request_ids.extend(ids)
            # More efficient to abort all as a single batch.  批量执行
            self.abort_requests(request_ids)

    def shutdown(self):
        self.structured_output_manager.clear_backend()
        if self.model_executor:
            self.model_executor.shutdown()
        if self.scheduler:
            self.scheduler.shutdown()

    def profile(self, is_start: bool = True):
        self.model_executor.profile(is_start)

    def reset_mm_cache(self):
        # NOTE: Since this is mainly for debugging, we don't attempt to
        # re-sync the internal caches (P0 sender, P1 receiver)
        if self.scheduler.has_unfinished_requests():
            logger.warning(
                "Resetting the multi-modal cache when requests are "
                "in progress may lead to desynced internal caches."
            )

        # The cache either exists in EngineCore or WorkerWrapperBase
        if self.mm_receiver_cache is not None:
            self.mm_receiver_cache.clear_cache()

        self.model_executor.reset_mm_cache()

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        return self.scheduler.reset_prefix_cache(
            reset_running_requests, reset_connector
        )

    def sleep(self, level: int = 1):
        self.model_executor.sleep(level)

    def wake_up(self, tags: list[str] | None = None):
        self.model_executor.wake_up(tags)

    def is_sleeping(self) -> bool:
        return self.model_executor.is_sleeping

    def execute_dummy_batch(self):
        self.model_executor.execute_dummy_batch()

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_executor.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_executor.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        return self.model_executor.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        return self.model_executor.pin_lora(lora_id)

    def save_sharded_state(
        self,
        path: str,
        pattern: str | None = None,
        max_size: int | None = None,
    ) -> None:
        self.model_executor.save_sharded_state(
            path=path, pattern=pattern, max_size=max_size
        )

    def collective_rpc(
        self,
        method: str | Callable[..., _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        return self.model_executor.collective_rpc(method, timeout, args, kwargs)

    def preprocess_add_request(self, request: EngineCoreRequest) -> tuple[Request, int]:
        """Preprocess the request.                                                  对即将添加的请求进行预处理
                                                                                        -设计目的：请求初始化（构造request、处理多模态、结构化输出等）通常比较重
        This function could be directly used in input processing thread to allow        -如果放在主调度/推理线程，会堵塞模型forward
        request initialization running in parallel with Model forward                   -因此这里允许在输入处理线程中提前做完这些工作
        """                                                                            #返回：（内部Request对象，当前request所属wave id） ， wave_id是哪一批被送进系统的请求
        # Note on thread safety: no race condition.                         
        # `mm_receiver_cache` is reset at the end of LLMEngine init,                    线程安全说明：mm_receiver_cache只在初始化后被input_thread使用，不会和其他线程并发修改，无竞态条件
        # and will only be accessed in the input processing thread afterwards.
        if self.mm_receiver_cache is not None and request.mm_features:                 #对多模态特征（图像embedding等）做缓存/复用/更新
            request.mm_features = self.mm_receiver_cache.get_and_update_features(      #典型作用：避免重复编码（比如同一张图）  统一特征格式
                request.mm_features
            )

        req = Request.from_engine_core_request(request, self.request_block_hasher)     #构造内部Request对象， 把外部EngingCoreRwquest转换成调度器使用的Request，同时会绑定：block hash(用于prefix cache/kv cache命中)，token信息，调度所需的各种元数据
        if req.use_structured_output:                                                  #结构化输出（Structured Output）初始化 
            # Note on thread safety: no race condition.                                #线程安全说明： 
            # `grammar_init` is only invoked in input processing thread. For                grammar_init只在input thread中调用
            # `structured_output_manager`, each request is independent and                  每个request的grammar是独立的
            # grammar compilation is async. Scheduler always checks grammar                 编译是异步的（不会阻塞主线程）
            # compilation status before scheduling request.                            #这里做的事情：初始化grammar(如json schema/正则/约束解码规则)， 触发异步编译（FSM/parser）
            self.structured_output_manager.grammar_init(req)                           #注意scheduler会在真正调度该request前，会检查grammar是否已准备好
        return req, request.current_wave                                               #返回结果，req：调度系统内部使用的请求对象。request.current_wave:用于batching/调度分组（wave概念）； 帮助scheduler做分批处理或公平调度


class EngineCoreProc(EngineCore):
    """ZMQ-wrapper for running EngineCore in background process.
    EngineCore的进程包装类
    作用：把EngineCore运行在一个独立的后台进程中，并通过ZMQ消息队列与前端API SERVER/Client进行通信
    为啥需要这个类？- EngineCore是纯逻辑核心，不能直接暴漏给外部进程通信
                    -EngineCoreProc 负责管理 ZMQ Socket、输入输出队列、后台线程等，实现了 EngineCore 与外部的异步、高性能通信。
                    -支持数据并行，多引擎实例等高级特性
                    -让模型计算（GPU密集）和网络IO可以更好地重叠执行。
    
    """

    ENGINE_CORE_DEAD = b"ENGINE_CORE_DEAD"

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        client_handshake_address: str | None = None,
        engine_index: int = 0,
    ):
        self.input_queue = queue.Queue[tuple[EngineCoreRequestType, Any]]()             ## 输入队列：用于接收来自 ZMQ 的请求（ADD、ABORT、UTILITY 等）
        self.output_queue = queue.Queue[tuple[int, EngineCoreOutputs] | bytes]()        # 输出队列：用于把 EngineCore 的处理结果发送回客户端
        executor_fail_callback = lambda: self.input_queue.put_nowait(                   ## 当 Executor（Worker）发生崩溃时，往输入队列里放入失败信号
            (EngineCoreRequestType.EXECUTOR_FAILED, b"")
        )

        self.engine_index = engine_index                                                ## 当前引擎实例的编号（多引擎时使用）
        identity = self.engine_index.to_bytes(length=2, byteorder="little")
        self.engines_running = False

        
        # ====================== 执行握手协议 ======================
        # 与前端、DP Coordinator 进行握手，交换通信地址等信息
        with self._perform_handshakes(
            handshake_address,
            identity,
            local_client,
            vllm_config,
            client_handshake_address,
        ) as addresses:
            self.client_count = len(addresses.outputs)

            # Set up data parallel environment.# 设置数据并行（DP）相关环境
            self.has_coordinator = addresses.coordinator_output is not None
            self.frontend_stats_publish_address = (
                addresses.frontend_stats_publish_address
            )
            logger.debug(
                "Has DP Coordinator: %s, stats publish address: %s",
                self.has_coordinator,
                self.frontend_stats_publish_address,
            )
            # Only publish request queue stats to coordinator for "internal"  # 决定是否要向 Coordinator 发布请求队列统计信息
            # and "hybrid" LB modes .
            self.publish_dp_lb_stats = (
                self.has_coordinator
                and not vllm_config.parallel_config.data_parallel_external_lb
            )

            self._init_data_parallel(vllm_config)

            super().__init__(
                vllm_config, executor_class, log_stats, executor_fail_callback
            )

            # Background Threads and Queues for IO. These enable us to
            # overlap ZMQ socket IO with GPU since they release the GIL,
            # and to overlap some serialization/deserialization with the
            # model forward pass.
            # Threads handle Socket <-> Queues and core_busy_loop uses Queue.
            # ====================== 启动后台 IO 线程 ======================
            # 这些线程非常重要：它们负责 ZMQ Socket 与 Queue 之间的数据搬运，
            # 释放 GIL，让网络 IO 和 GPU 计算可以真正并行（重叠）执行。
            ready_event = threading.Event()                                   # 用于等待握手完成的信号
            input_thread = threading.Thread(                                  # 输入线程：负责从 ZMQ Socket 接收请求，放入 input_queue
                target=self.process_input_sockets,
                args=(
                    addresses.inputs,
                    addresses.coordinator_input,
                    identity,
                    ready_event,
                ),
                daemon=True,                                                  # 设置为守护线程，进程退出时自动结束
            )
            input_thread.start()

            self.output_thread = threading.Thread(                            # 输出线程：负责把 EngineCore 的输出通过 ZMQ 发送给客户端
                target=self.process_output_sockets,
                args=(
                    addresses.outputs,
                    addresses.coordinator_output,
                    self.engine_index,
                ),
                daemon=True,
            )
            self.output_thread.start()

            # Don't complete handshake until DP coordinator ready message is   #等待DP Coordinator发送READY消息，确认所有组件就绪 如果等待超时或输入线程死亡，则抛出异常
            # received.
            while not ready_event.wait(timeout=10):
                if not input_thread.is_alive():
                    raise RuntimeError("Input socket thread died during startup")
                assert addresses.coordinator_input is not None
                logger.info("Waiting for READY message from DP Coordinator...")

    @contextmanager
    def _perform_handshakes(
        self,
        handshake_address: str,
        identity: bytes,
        local_client: bool,
        vllm_config: VllmConfig,
        client_handshake_address: str | None,
    ) -> Generator[EngineZmqAddresses, None, None]:
        """
        Perform startup handshakes.

        For DP=1 or offline mode, this is with the colocated front-end process.

        For DP>1 with internal load-balancing this is with the shared front-end
        process which may reside on a different node.

        For DP>1 with external or hybrid load-balancing, two handshakes are
        performed:
            - With the rank 0 front-end process which retrieves the
              DP Coordinator ZMQ addresses and DP process group address.
            - With the colocated front-end process which retrieves the
              client input/output socket addresses.
        with the exception of the rank 0 and colocated engines themselves which
        don't require the second handshake.

        Here, "front-end" process can mean the process containing the engine
        core client (which is the API server process in the case the API
        server is not scaled out), OR the launcher process running the
        run_multi_api_server() function in serve.py.
        """
        input_ctx = zmq.Context()
        is_local = local_client and client_handshake_address is None
        headless = not local_client
        handshake = self._perform_handshake(
            input_ctx,
            handshake_address,
            identity,
            is_local,
            headless,
            vllm_config,
            vllm_config.parallel_config,
        )
        if client_handshake_address is None:
            with handshake as addresses:
                yield addresses
        else:
            assert local_client
            local_handshake = self._perform_handshake(
                input_ctx, client_handshake_address, identity, True, False, vllm_config
            )
            with handshake as addresses, local_handshake as client_addresses:
                addresses.inputs = client_addresses.inputs
                addresses.outputs = client_addresses.outputs
                yield addresses

        # Update config which may have changed from the handshake
        vllm_config.__post_init__()

    @contextmanager
    def _perform_handshake(
        self,
        ctx: zmq.Context,
        handshake_address: str,
        identity: bytes,
        local_client: bool,
        headless: bool,
        vllm_config: VllmConfig,
        parallel_config_to_update: ParallelConfig | None = None,
    ) -> Generator[EngineZmqAddresses, None, None]:
        with make_zmq_socket(
            ctx,
            handshake_address,
            zmq.DEALER,
            identity=identity,
            linger=5000,
            bind=False,
        ) as handshake_socket:
            # Register engine with front-end.
            addresses = self.startup_handshake(
                handshake_socket, local_client, headless, parallel_config_to_update
            )
            yield addresses

            # Send ready message.
            num_gpu_blocks = vllm_config.cache_config.num_gpu_blocks
            # We pass back the coordinator stats update address here for the
            # external LB case for our colocated front-end to use (coordinator
            # only runs with rank 0).
            dp_stats_address = self.frontend_stats_publish_address

            # Include config hash for DP configuration validation
            ready_msg = {
                "status": "READY",
                "local": local_client,
                "headless": headless,
                "num_gpu_blocks": num_gpu_blocks,
                "dp_stats_address": dp_stats_address,
            }
            if vllm_config.parallel_config.data_parallel_size > 1:
                ready_msg["parallel_config_hash"] = (
                    vllm_config.parallel_config.compute_hash()
                )

            handshake_socket.send(msgspec.msgpack.encode(ready_msg))

    @staticmethod
    def startup_handshake(
        handshake_socket: zmq.Socket,
        local_client: bool,
        headless: bool,
        parallel_config: ParallelConfig | None = None,
    ) -> EngineZmqAddresses:
        # Send registration message.
        handshake_socket.send(
            msgspec.msgpack.encode(
                {
                    "status": "HELLO",
                    "local": local_client,
                    "headless": headless,
                }
            )
        )

        # Receive initialization message.
        logger.debug("Waiting for init message from front-end.")
        if not handshake_socket.poll(timeout=HANDSHAKE_TIMEOUT_MINS * 60_000):
            raise RuntimeError(
                "Did not receive response from front-end "
                f"process within {HANDSHAKE_TIMEOUT_MINS} "
                f"minutes"
            )
        init_bytes = handshake_socket.recv()
        init_message: EngineHandshakeMetadata = msgspec.msgpack.decode(
            init_bytes, type=EngineHandshakeMetadata
        )
        logger.debug("Received init message: %s", init_message)

        if parallel_config is not None:
            for key, value in init_message.parallel_config.items():
                setattr(parallel_config, key, value)

        return init_message.addresses

    @staticmethod
    def run_engine_core(*args, dp_rank: int = 0, local_dp_rank: int = 0, **kwargs):
        """Launch EngineCore busy loop in background process."""

        # Signal handler used for graceful termination.                         优雅退出相关,标记是否已经收到退出信号,避免重复处理
        # SystemExit exception is only raised once to allow this and worker     设计目的:SIGTERM / SIGINT 可能触发多次  只希望触发一次 SystemExit，让流程干净退出
        # processes to terminate without error
        shutdown_requested = False

        # Ensure we can serialize transformer config after spawning
        maybe_register_config_serialize_by_value()

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested:                                           # 第一次收到信号 → 触发 SystemExit
                shutdown_requested = True
                raise SystemExit()

        # Either SIGTERM or SIGINT will terminate the engine_core
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        engine_core: EngineCoreProc | None = None
        try:
            parallel_config: ParallelConfig = kwargs["vllm_config"].parallel_config
            if parallel_config.data_parallel_size > 1 or dp_rank > 0:
                set_process_title("EngineCore", f"DP{dp_rank}")
                decorate_logs()                                                  #给日志增加 rank 前缀（区分不同 worker）
                # Set data parallel rank for this engine process.                # 设置当前进程的 DP rank 信息
                parallel_config.data_parallel_rank = dp_rank
                parallel_config.data_parallel_rank_local = local_dp_rank
                engine_core = DPEngineCoreProc(*args, **kwargs)                  # 创建 Data Parallel 专用 EngineCore
            else:                                                                # 单卡 / 非 DP 情况
                set_process_title("EngineCore")
                decorate_logs()
                engine_core = EngineCoreProc(*args, **kwargs)                    #创建普通EngineCore

            engine_core.run_busy_loop()

        except SystemExit:
            logger.debug("EngineCore exiting.")
            raise
        except Exception as e:
            if engine_core is None:
                logger.exception("EngineCore failed to start.")
            else:
                logger.exception("EngineCore encountered a fatal error.")
                engine_core._send_engine_dead()
            raise e
        finally:
            if engine_core is not None:
                engine_core.shutdown()

    def _init_data_parallel(self, vllm_config: VllmConfig):
        pass

    def run_busy_loop(self):
        """Core busy loop of the EngineCore.EngineCore的核心循环 它负责不断地接受请求-执行推理-输出结果，是整个异步引擎的驱动器"""

        # Loop until process is sent a SIGINT or SIGTERM
        while True:
            # 1) Poll the input queue until there is work to do.
            self._process_input_queue()
            # 2) Step the engine core and return the outputs.
            self._process_engine_step()

    def _process_input_queue(self):
        """Exits when an engine step needs to be performed."""
        #以同步的方式从Engine输入队列Input_queue中获取新的请求，并在随后用_handle_client_request进一步放入该请求并做进一步的调度
        waited = False
        while (
            not self.engines_running
            and not self.scheduler.has_requests()
            and not self.batch_queue
        ):
            if self.input_queue.empty():
                # Drain aborts queue; all aborts are also processed via input_queue.
                with self.aborts_queue.mutex:
                    self.aborts_queue.queue.clear()
                if logger.isEnabledFor(DEBUG):
                    logger.debug("EngineCore waiting for work.")
                    waited = True
            req = self.input_queue.get()            #req = (<EngineCoreRequestType.ADD: b'\x00'>, (<vllm.v1.request.Request object at 0x7f02943c7c70>, 0))。 就是一个tuple : (操作类型, 操作内容)
            self._handle_client_request(*req)       #元组第一部分是个枚举，表示对EngineCore发起的请求类型。第二部分又是个二元组（请求对象，一个数字）

        if waited:
            logger.debug("EngineCore loop active.")

        # Handle any more client requests.
        while not self.input_queue.empty():
            req = self.input_queue.get_nowait()
            self._handle_client_request(*req)

    def _process_engine_step(self) -> bool:
        """Called only when there are unfinished local requests.执行engine的单步处理  该函数仅在未完成的本地请求时才会被调用"""

        # Step the engine core.                                         调用Engine Core的核心step函数，执行一次调度+模型推理，返回2个值  outputs: 本次 step 产生的输出结果（dict，key 为 request_id）
        outputs, model_executed = self.step_fn()                        #model_executed: 本次 step 是否真正执行了模型 forward（即是否发生了 GPU 计算）
        # Put EngineCoreOutputs into the output queue.                  #将本次生成的输出结果放入输出队列，供上层（AsyncLLMEngine 或用户调用）消费
        for output in outputs.items() if outputs else ():
            self.output_queue.put_nowait(output)                        #使用put_nowait，非阻塞方式放入输出队列  流式的话这里放进去 终端就收到了
        # Post-step hook.
        self.post_step(model_executed)                                  #执行step完成后的钩子函数，通常用于清理、统计更新、日志记录等后处理工作

        # If no model execution happened but there are waiting requests   如果未发生模型执行，但仍有处于等待状态的请求（例如：正在等待远程 KV 缓存），
        # (e.g., WAITING_FOR_REMOTE_KVS), yield the GIL briefly to allow  则短暂释放GIL，以允许后台线程（如NIXL握手协议）取得进展。如果不这么做
        # background threads (like NIXL handshake) to make progress.      高频的轮询死循环可能会导致线程因资源匮乏而饿死
        # Without this, the tight polling loop can starve background threads.
        if not model_executed and self.scheduler.has_unfinished_requests():
            time.sleep(0.001)

        return model_executed

    def _handle_client_request(
        self, request_type: EngineCoreRequestType, request: Any
    ) -> None:
        """Dispatch request from client.
        处理从客户端（Frontend / API Server）发送过来的各种请求，并进行分发。
        这是 EngineCore 接收外部请求的主要入口函数，所有用户请求最终都会通过这里进行处理。
        """

        if request_type == EngineCoreRequestType.ADD:       # 添加新的推理请求（最常见的请求类型）# request 格式为 (req, request_wave)
            req, request_wave = request
            self.add_request(req, request_wave)
        elif request_type == EngineCoreRequestType.ABORT:   #取消指定的请求,用户主动取消、超时、或按 Ctrl+C 调试时也会走到这里
            self.abort_requests(request)
        elif request_type == EngineCoreRequestType.UTILITY: #工具类调用请求（例如获取metrics、模型信息、当前状态等）
            client_idx, call_id, method_name, args = request
            output = UtilityOutput(call_id)
            try:
                method = getattr(self, method_name)
                result = method(*self._convert_msgspec_args(method, args))
                output.result = UtilityResult(result)
            except BaseException as e:
                logger.exception("Invocation of %s method failed", method_name)
                output.failure_message = (
                    f"Call to {method_name} method failed: {str(e)}"
                )
            self.output_queue.put_nowait(                    #将工具调用结果放入输出队列，返回给客户端
                (client_idx, EngineCoreOutputs(utility_output=output))
            )
        elif request_type == EngineCoreRequestType.EXECUTOR_FAILED:
            raise RuntimeError("Executor failed.")           #Worker执行器发生严重错误时的通知，通常表示某个Worker进程崩溃
        else:
            logger.error(
                "Unrecognized input request type encountered: %s", request_type
            )

    @staticmethod
    def _convert_msgspec_args(method, args):
        """If a provided arg type doesn't match corresponding target method
        arg type, try converting to msgspec object."""
        if not args:
            return args
        arg_types = signature(method).parameters.values()
        assert len(args) <= len(arg_types)
        return tuple(
            msgspec.convert(v, type=p.annotation)
            if isclass(p.annotation)
            and issubclass(p.annotation, msgspec.Struct)
            and not isinstance(v, p.annotation)
            else v
            for v, p in zip(args, arg_types)
        )

    def _send_engine_dead(self):
        """Send EngineDead status to the EngineCoreClient."""

        # Put ENGINE_CORE_DEAD in the queue.
        self.output_queue.put_nowait(EngineCoreProc.ENGINE_CORE_DEAD)

        # Wait until msg sent by the daemon before shutdown.
        self.output_thread.join(timeout=5.0)
        if self.output_thread.is_alive():
            logger.fatal(
                "vLLM shutdown signal from EngineCore failed "
                "to send. Please report this issue."
            )
    #独立IO线程，同时监听多个输入socket,把收到的二进制消息解析成Python请求对象，再安全 非堵塞的送进engine的输入队列
    def process_input_sockets(
        self,
        input_addresses: list[str], #多个输入 socket 地址列表（要连接的客户端或服务端地址）
        coord_input_address: str | None,#协调器（coordinator）的地址，可选
        identity: bytes,                #这个线程/套接字的身份标识，用于 ROUTER/DEALER 异步通信
        ready_event: threading.Event,  #线程事件，用于通知外部“socket 已准备好”
    ):
        """Input socket IO thread."""

        # Msgpack serialization decoding. #初始化消息解码器
        add_request_decoder = MsgpackDecoder(EngineCoreRequest)  #专门解码EngineCoreRequest类型的请求
        generic_decoder = MsgpackDecoder()                  #通用解码器，可解码任意类型的请求

        #创建上下文和ExitStack
        #zmq.Context()：创建一个新的ZeroMQ上下文，用于管理套接字和 I/O 线程
        #ExitStack()：用于 自动管理多个资源（套接字、文件等），退出时自动关闭
        with ExitStack() as stack, zmq.Context() as ctx:
            #1.创建zmq套接字和客户端连接
            #stack.enter_context(...)：把 socket 注册到 ExitStack，确保线程结束时自动关闭
            #make_zmq_socket 返回的对象是一个 zmq.Socket 对象，每个socket已经连接到对应地址，类型是DEALER
            input_sockets = [
                stack.enter_context(
                    make_zmq_socket(ctx, input_address, zmq.DEALER, identity=identity, bind=False)
                )
                for input_address in input_addresses
            ]
            if coord_input_address is None:
                coord_socket = None
            else:
                coord_socket = stack.enter_context(
                    make_zmq_socket(
                        ctx,
                        coord_input_address,
                        zmq.XSUB,
                        identity=identity,
                        bind=False,
                    )
                )
                # Send subscription message to coordinator.
                coord_socket.send(b"\x01")

            # Register sockets with poller.
            poller = zmq.Poller()
            for input_socket in input_sockets:
                # Send initial message to each input socket - this is required
                # before the front-end ROUTER socket can send input messages
                # back to us.
                # 将套接字注册到poller中，让内核关注套接字的读写事件
                #为什么一上来要发一个空消息？原因ROUTER ↔ DEALER 通信时，ROUTER 不知道 DEALER 的 identity，直到 DEALER 先发过消息
                input_socket.send(b"")
                poller.register(input_socket, zmq.POLLIN)

            if coord_socket is not None:
                # Wait for ready message from coordinator.
                assert coord_socket.recv() == b"READY"
                poller.register(coord_socket, zmq.POLLIN)
            #通知外部线程自己就绪
            ready_event.set()
            #删除引用，避免泄露
            del ready_event
            while True:
                for input_socket, _ in poller.poll():
                    # (RequestType, RequestData)
                    #接收来自客户端的请求数据
                    type_frame, *data_frames = input_socket.recv_multipart(copy=False)
                    #将第一段bytes转成枚举类型 用于判断后续如何解码和处理
                    request_type = EngineCoreRequestType(bytes(type_frame.buffer))

                    # Deserialize the request data.
                    request: Any
                    if request_type == EngineCoreRequestType.ADD:
                        req: EngineCoreRequest = add_request_decoder.decode(data_frames)
                        try:
                            request = self.preprocess_add_request(req)
                        except Exception:
                            self._handle_request_preproc_error(req)
                            continue
                    else:
                        #其他类型 用通用解码器generic_decoder解码
                        request = generic_decoder.decode(data_frames)
                        #ABORT类型请求：放入aborts_queue, 保证尽快处理，但不破坏原队列顺序
                        if request_type == EngineCoreRequestType.ABORT:
                            # Aborts are added to *both* queues, allows us to eagerly
                            # process aborts while also ensuring ordering in the input
                            # queue to avoid leaking requests. This is ok because
                            # aborting in the scheduler is idempotent.
                            self.aborts_queue.put_nowait(request)
                    #放到engine的待处理队列中
                    # Push to input queue for core busy loop.
                    #为什么是put_nowait？ 因为这里是IO线程，不能阻塞，如果不用put_await 如果input_queue满了会堵塞
                    #put_nowait 如果有空间立即放进去，满了立刻抛出异常 绝不会等待
                    self.input_queue.put_nowait((request_type, request))

    def process_output_sockets(
        self,
        output_paths: list[str],
        coord_output_path: str | None,
        engine_index: int,
    ):
        """Output socket IO thread."""

        # Msgpack serialization encoding.
        encoder = MsgpackEncoder()
        # Send buffers to reuse.
        reuse_buffers: list[bytearray] = []
        # Keep references to outputs and buffers until zmq is finished
        # with them (outputs may contain tensors/np arrays whose
        # backing buffers were extracted for zero-copy send).
        pending = deque[tuple[zmq.MessageTracker, Any, bytearray]]()

        # We must set linger to ensure the ENGINE_CORE_DEAD
        # message is sent prior to closing the socket.
        with ExitStack() as stack, zmq.Context() as ctx:
            sockets = [
                stack.enter_context(
                    make_zmq_socket(ctx, output_path, zmq.PUSH, linger=4000)
                )
                for output_path in output_paths
            ]
            coord_socket = (
                stack.enter_context(
                    make_zmq_socket(
                        ctx, coord_output_path, zmq.PUSH, bind=False, linger=4000
                    )
                )
                if coord_output_path is not None
                else None
            )
            max_reuse_bufs = len(sockets) + 1

            while True:
                output = self.output_queue.get()
                if output == EngineCoreProc.ENGINE_CORE_DEAD:
                    for socket in sockets:
                        socket.send(output)
                    break
                assert not isinstance(output, bytes)
                client_index, outputs = output
                outputs.engine_index = engine_index

                if client_index == -1:
                    # Don't reuse buffer for coordinator message
                    # which will be very small.
                    assert coord_socket is not None
                    coord_socket.send_multipart(encoder.encode(outputs))
                    continue

                # Reclaim buffers that zmq is finished with.
                while pending and pending[-1][0].done:
                    reuse_buffers.append(pending.pop()[2])

                buffer = reuse_buffers.pop() if reuse_buffers else bytearray()
                buffers = encoder.encode_into(outputs, buffer)
                tracker = sockets[client_index].send_multipart(
                    buffers, copy=False, track=True
                )
                if not tracker.done:
                    ref = outputs if len(buffers) > 1 else None
                    pending.appendleft((tracker, ref, buffer))
                elif len(reuse_buffers) < max_reuse_bufs:
                    # Limit the number of buffers to reuse.
                    reuse_buffers.append(buffer)

    def _handle_request_preproc_error(self, request: EngineCoreRequest) -> None:
        """Log and return a request-scoped error response for exceptions raised
        from the add request preprocessing in the input socket processing thread.
        """
        logger.exception(
            "Unexpected error pre-processing request %s", request.request_id
        )
        self.output_queue.put_nowait(
            (
                request.client_index,
                EngineCoreOutputs(
                    engine_index=self.engine_index,
                    finished_requests={request.request_id},
                    outputs=[
                        EngineCoreOutput(
                            request_id=request.request_id,
                            new_token_ids=[],
                            finish_reason=FinishReason.ERROR,
                        )
                    ],
                ),
            )
        )


class DPEngineCoreProc(EngineCoreProc):
    """ZMQ-wrapper for running EngineCore in background process
    in a data parallel context."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        client_handshake_address: str | None = None,
    ):
        # Counts forward-passes of the model so that we can synchronize
        # finished with DP peers every N steps.
        self.step_counter = 0
        self.current_wave = 0
        self.last_counts = (0, 0)

        # Initialize the engine.
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        super().__init__(
            vllm_config,
            local_client,
            handshake_address,
            executor_class,
            log_stats,
            client_handshake_address,
            dp_rank,
        )

    def _init_data_parallel(self, vllm_config: VllmConfig):
        # Configure GPUs and stateless process group for data parallel.
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        dp_size = vllm_config.parallel_config.data_parallel_size
        local_dp_rank = vllm_config.parallel_config.data_parallel_rank_local

        assert dp_size > 1
        assert local_dp_rank is not None
        assert 0 <= local_dp_rank <= dp_rank < dp_size

        if vllm_config.kv_transfer_config is not None:
            # modify the engine_id and append the local_dp_rank to it to ensure
            # that the kv_transfer_config is unique for each DP rank.
            vllm_config.kv_transfer_config.engine_id = (
                f"{vllm_config.kv_transfer_config.engine_id}_dp{local_dp_rank}"
            )
            logger.debug(
                "Setting kv_transfer_config.engine_id to %s",
                vllm_config.kv_transfer_config.engine_id,
            )

        self.dp_rank = dp_rank
        self.dp_group = vllm_config.parallel_config.stateless_init_dp_group()

    def shutdown(self):
        super().shutdown()
        if dp_group := getattr(self, "dp_group", None):
            stateless_destroy_torch_distributed_process_group(dp_group)

    def add_request(self, request: Request, request_wave: int = 0):
        if self.has_coordinator and request_wave != self.current_wave:
            if request_wave > self.current_wave:
                self.current_wave = request_wave
            elif not self.engines_running:
                # Request received for an already-completed wave, notify
                # front-end that we need to start the next one.
                self.output_queue.put_nowait(
                    (-1, EngineCoreOutputs(start_wave=self.current_wave))
                )

        super().add_request(request, request_wave)

    def _handle_client_request(
        self, request_type: EngineCoreRequestType, request: Any
    ) -> None:
        if request_type == EngineCoreRequestType.START_DP_WAVE:
            new_wave, exclude_eng_index = request
            if exclude_eng_index != self.engine_index and (
                new_wave >= self.current_wave
            ):
                self.current_wave = new_wave
                if not self.engines_running:
                    logger.debug("EngineCore starting idle loop for wave %d.", new_wave)
                    self.engines_running = True
        else:
            super()._handle_client_request(request_type, request)

    def _maybe_publish_request_counts(self):
        if not self.publish_dp_lb_stats:
            return

        # Publish our request counts (if they've changed).
        counts = self.scheduler.get_request_counts()
        if counts != self.last_counts:
            self.last_counts = counts
            stats = SchedulerStats(
                *counts, step_counter=self.step_counter, current_wave=self.current_wave
            )
            self.output_queue.put_nowait((-1, EngineCoreOutputs(scheduler_stats=stats)))

    def run_busy_loop(self):
        """Core busy loop of the EngineCore for data parallel case."""

        # Loop until process is sent a SIGINT or SIGTERM
        while True:
            # 1) Poll the input queue until there is work to do.
            self._process_input_queue()

            # 2) Step the engine core.
            executed = self._process_engine_step()
            self._maybe_publish_request_counts()

            local_unfinished_reqs = self.scheduler.has_unfinished_requests()
            if not executed:
                if not local_unfinished_reqs and not self.engines_running:
                    # All engines are idle.
                    continue

                # We are in a running state and so must execute a dummy pass
                # if the model didn't execute any ready requests.
                self.execute_dummy_batch()

            # 3) All-reduce operation to determine global unfinished reqs.
            self.engines_running = self._has_global_unfinished_reqs(
                local_unfinished_reqs
            )

            if not self.engines_running:
                if self.dp_rank == 0 or not self.has_coordinator:
                    # Notify client that we are pausing the loop.
                    logger.debug(
                        "Wave %d finished, pausing engine loop.", self.current_wave
                    )
                    # In the coordinator case, dp rank 0 sends updates to the
                    # coordinator. Otherwise (offline spmd case), each rank
                    # sends the update to its colocated front-end process.
                    client_index = -1 if self.has_coordinator else 0
                    self.output_queue.put_nowait(
                        (
                            client_index,
                            EngineCoreOutputs(wave_complete=self.current_wave),
                        )
                    )
                # Increment wave count and reset step counter.
                self.current_wave += 1
                self.step_counter = 0

    def _has_global_unfinished_reqs(self, local_unfinished: bool) -> bool:
        # Optimization - only perform finish-sync all-reduce every 32 steps.
        self.step_counter += 1
        if self.step_counter % 32 != 0:
            return True

        return ParallelConfig.has_unfinished_dp(self.dp_group, local_unfinished)

    def reinitialize_distributed(
        self, reconfig_request: ReconfigureDistributedRequest
    ) -> None:
        stateless_destroy_torch_distributed_process_group(self.dp_group)
        self.shutdown()

        parallel_config = self.vllm_config.parallel_config
        old_dp_size = parallel_config.data_parallel_size
        parallel_config.data_parallel_size = reconfig_request.new_data_parallel_size
        if reconfig_request.new_data_parallel_rank != -1:
            parallel_config.data_parallel_rank = reconfig_request.new_data_parallel_rank
        # local rank specifies device visibility, it should not be changed
        assert (
            reconfig_request.new_data_parallel_rank_local
            == ReconfigureRankType.KEEP_CURRENT_RANK
        )
        parallel_config.data_parallel_master_ip = (
            reconfig_request.new_data_parallel_master_ip
        )
        parallel_config.data_parallel_master_port = (
            reconfig_request.new_data_parallel_master_port
        )
        if reconfig_request.new_data_parallel_rank != -2:
            self.dp_rank = parallel_config.data_parallel_rank
            self.dp_group = parallel_config.stateless_init_dp_group()
        reconfig_request.new_data_parallel_master_port = (
            parallel_config.data_parallel_master_port
        )

        self.model_executor.reinitialize_distributed(reconfig_request)
        if reconfig_request.new_data_parallel_size > old_dp_size:
            assert self.available_gpu_memory_for_kv_cache > 0
            # pass available_gpu_memory_for_kv_cache from existing
            # engine-cores to new engine-cores so they can directly
            # use it in _initialize_kv_caches() rather than profiling.
            ParallelConfig.sync_kv_cache_memory_size(
                self.dp_group, self.available_gpu_memory_for_kv_cache
            )
            # NOTE(yongji): newly joined workers require dummy_run even
            # CUDA graph is not used
            self.model_executor.collective_rpc("compile_or_warm_up_model")
        if (
            reconfig_request.new_data_parallel_rank
            == ReconfigureRankType.SHUTDOWN_CURRENT_RANK
        ):
            self.shutdown()
            logger.info("DPEngineCoreProc %s shutdown", self.dp_rank)
        else:
            logger.info(
                "Distributed environment reinitialized for DP rank %s", self.dp_rank
            )


class DPEngineCoreActor(DPEngineCoreProc):
    """
    Ray actor for running EngineCore in a data parallel context
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        addresses: EngineZmqAddresses,
        executor_class: type[Executor],
        log_stats: bool,
        dp_rank: int = 0,
        local_dp_rank: int = 0,
    ):
        self.addresses = addresses
        vllm_config.parallel_config.data_parallel_rank = dp_rank
        vllm_config.parallel_config.data_parallel_rank_local = local_dp_rank

        # Set CUDA_VISIBLE_DEVICES as early as possible in actor life cycle
        # NOTE: in MP we set CUDA_VISIBLE_DEVICES at process creation time,
        # and this cannot be done in the same way for Ray because:
        # 1) Ray manages life cycle of all ray workers (including
        # DPEngineCoreActor)
        # 2) Ray sets CUDA_VISIBLE_DEVICES based on num_gpus configuration
        # To bypass 2, we need to also set
        # RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES, but vLLM workers created
        # thereafter would have CUDA_VISIBLE_DEVICES set, which is sticky:
        # https://github.com/ray-project/ray/blob/e752fc319ddedd9779a0989b6d3613909bad75c9/python/ray/_private/worker.py#L456 # noqa: E501
        # This is problematic because when the vLLM worker (a Ray actor)
        # executes a task, it indexes into the sticky CUDA_VISIBLE_DEVICES
        # rather than directly using the GPU ID, potentially resulting in
        # index out of bounds error. See:
        # https://github.com/ray-project/ray/pull/40461/files#diff-31e8159767361e4bc259b6d9883d9c0d5e5db780fcea4a52ead4ee3ee4a59a78R1860 # noqa: E501
        # and get_accelerator_ids_for_accelerator_resource() in worker.py
        # of ray.
        self._set_visible_devices(vllm_config, local_dp_rank)

        super().__init__(vllm_config, local_client, "", executor_class, log_stats)

    def _set_visible_devices(self, vllm_config: VllmConfig, local_dp_rank: int):
        from vllm.platforms import current_platform

        if current_platform.is_xpu():
            pass
        else:
            device_control_env_var = current_platform.device_control_env_var                #CUDA 需要手动用环境变量“屏蔽设备”，而很多 XPU 运行时本身就已经把设备分配/隔离好了，所以这里不需要再做一遍。
            self._set_cuda_visible_devices(
                vllm_config, local_dp_rank, device_control_env_var
            )

    def _set_cuda_visible_devices(
        self, vllm_config: VllmConfig, local_dp_rank: int, device_control_env_var: str
    ):
        world_size = vllm_config.parallel_config.world_size
        # Set CUDA_VISIBLE_DEVICES or equivalent.
        try:
            value = get_device_indices(
                device_control_env_var, local_dp_rank, world_size
            )
            os.environ[device_control_env_var] = value
        except IndexError as e:
            raise Exception(
                f"Error setting {device_control_env_var}: "
                f"local range: [{local_dp_rank * world_size}, "
                f"{(local_dp_rank + 1) * world_size}) "
                f'base value: "{os.getenv(device_control_env_var)}"'
            ) from e

    @contextmanager
    def _perform_handshakes(
        self,
        handshake_address: str,
        identity: bytes,
        local_client: bool,
        vllm_config: VllmConfig,
        client_handshake_address: str | None,
    ):
        """
        For Ray, we don't need to actually perform handshake.
        All addresses information is known before the actor creation.
        Therefore, we simply yield these addresses.
        """
        yield self.addresses

    def wait_for_init(self):
        """
        Wait until the engine core is initialized.

        This is just an empty method. When ray.get() on this method
        (or any other method of the actor) returns, it is guaranteed
        that actor creation (i.e., __init__) is complete.
        """
        pass

    def run(self):
        """
        Run the engine core busy loop.
        """
        try:
            self.run_busy_loop()
        except SystemExit:                                              #捕获“正常退出”信号（如外部调用 sys.exit()）
            logger.debug("EngineCore exiting.")                         # 这里只做调试日志记录，然后继续向上抛出，让上层完成退出流程
            raise
        except Exception:                                               #捕获所有非 SystemExit 的异常（真正的错误）
            logger.exception("EngineCore encountered a fatal error.")   ## 记录完整异常堆栈，便于排查问题
            raise
        finally:
            self.shutdown()
