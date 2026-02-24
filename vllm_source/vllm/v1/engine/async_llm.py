# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import os
import socket
import time
import warnings
from collections.abc import AsyncGenerator, Iterable, Mapping
from copy import copy
from typing import Any, cast

import numpy as np
import torch

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.utils import _validate_truncation_size
from vllm.inputs import PromptType
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.outputs import PoolingRequestOutput, RequestOutput
from vllm.plugins.io_processors import get_io_processor
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.tasks import SupportedTask
from vllm.tokenizers import TokenizerLike, cached_tokenizer_from_config
from vllm.tracing import init_tracer
from vllm.transformers_utils.config import maybe_register_config_serialize_by_value
from vllm.usage.usage_lib import UsageContext
from vllm.utils.async_utils import cancel_task_threadsafe
from vllm.utils.collection_utils import as_list
from vllm.utils.math_utils import cdiv
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core_client import EngineCoreClient
from vllm.v1.engine.exceptions import EngineDeadError, EngineGenerateError
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.engine.output_processor import OutputProcessor, RequestOutputCollector
from vllm.v1.engine.parallel_sampling import ParentRequest
from vllm.v1.executor import Executor
from vllm.v1.metrics.loggers import (
    StatLoggerFactory,
    StatLoggerManager,
    load_stat_logger_plugin_factories,
)
from vllm.v1.metrics.prometheus import shutdown_prometheus
from vllm.v1.metrics.stats import IterationStats

logger = init_logger(__name__)


class AsyncLLM(EngineClient):
    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        use_cached_outputs: bool = False,
        log_requests: bool = True,
        start_engine_loop: bool = True,
        stat_loggers: list[StatLoggerFactory] | None = None,
        aggregate_engine_logging: bool = False,
        client_addresses: dict[str, str] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ) -> None:
        """
        Create an AsyncLLM.

        Args:
            vllm_config: global configuration.
            executor_class: an Executor impl, e.g. MultiprocExecutor.
            log_stats: Whether to log stats.
            usage_context: Usage context of the LLM.
            mm_registry: Multi-modal registry.
            use_cached_outputs: Whether to use cached outputs.
            log_requests: Whether to log requests.
            start_engine_loop: Whether to start the engine loop.
            stat_loggers: customized stat loggers for the engine.
                If not provided, default stat loggers will be used.
                PLEASE BE AWARE THAT STAT LOGGER IS NOT STABLE
                IN V1, AND ITS BASE CLASS INTERFACE MIGHT CHANGE.

        Returns:
            None
        """
        # Ensure we can serialize custom transformer configs
        maybe_register_config_serialize_by_value()

        self.model_config = vllm_config.model_config
        self.vllm_config = vllm_config
        self.observability_config = vllm_config.observability_config
        self.log_requests = log_requests

        custom_stat_loggers = list(stat_loggers or [])
        custom_stat_loggers.extend(load_stat_logger_plugin_factories())

        has_custom_loggers = bool(custom_stat_loggers)
        self.log_stats = log_stats or has_custom_loggers
        if not log_stats and has_custom_loggers:
            logger.info(
                "AsyncLLM created with log_stats=False, "
                "but custom stat loggers were found; "
                "enabling logging without default stat loggers."
            )

        if self.model_config.skip_tokenizer_init:
            tokenizer = None
        else:
            tokenizer = cached_tokenizer_from_config(self.model_config)

        self.input_processor = InputProcessor(self.vllm_config, tokenizer)
        self.io_processor = get_io_processor(
            self.vllm_config,
            self.model_config.io_processor_plugin,
        )

        # OutputProcessor (converts EngineCoreOutputs --> RequestOutput).
        self.output_processor = OutputProcessor(
            self.tokenizer,
            log_stats=self.log_stats,
            stream_interval=self.vllm_config.scheduler_config.stream_interval,
        )
        #如果用户配置了opentelemetry的traces导出地址，就为vllm的输出输出模块启用tracing功能，让它能把推理过程中的关键步骤（尤其是输出相关部分）上报到外部Observablity系统（如jaeger）
        #为什么vLLM要加这个？生产环境最关心 从收到请求到返回花了多久，prefill阶段卡在哪里，解码每秒多少token，有没有OOM 超时异常？ 不通请求之间有没有关联？
        #opentelemetry+otlp是目前最主流的标准化方案
        endpoint = self.observability_config.otlp_traces_endpoint
        if endpoint is not None:
            tracer = init_tracer("vllm.llm_engine", endpoint)
            self.output_processor.tracer = tracer

        # EngineCore (starts the engine in background process).
        self.engine_core = EngineCoreClient.make_async_mp_client(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=self.log_stats,
            client_addresses=client_addresses,
            client_count=client_count,
            client_index=client_index,
        )

        # Loggers.
        self.logger_manager: StatLoggerManager | None = None
        if self.log_stats:#如果开启了统计日志功能，就创建一个统计日志管理器
            self.logger_manager = StatLoggerManager(
                vllm_config=vllm_config,
                engine_idxs=self.engine_core.engine_ranks_managed,
                custom_stat_loggers=custom_stat_loggers,
                enable_default_loggers=log_stats,
                client_count=client_count,
                aggregate_engine_logging=aggregate_engine_logging,
            )
            self.logger_manager.log_engine_initialized()

        # Pause / resume state for async RL workflows.
        #它俩一起组成了一个异步安全的“暂停/恢复”控制机制，专门设计给异步强化学习（async RL）工作流使用。
        self._pause_cond = asyncio.Condition()
        self._paused = False

        self.output_handler: asyncio.Task | None = None
        try:
            # Start output handler eagerly if we are in the asyncio eventloop.
            asyncio.get_running_loop()
            self._run_output_handler()
        except RuntimeError:
            pass

        #作用是有条件地为AsyncLLM的前端（python侧）启用Pytorch Profier(torch.profiler)，专门收集CPU活动追踪，并把结果保存到指定目录，用于后续性能分析
        #只有满足2个条件才启用profiler
        if (
            vllm_config.profiler_config.profiler == "torch"
            and not vllm_config.profiler_config.ignore_frontend
        ):

            profiler_dir = vllm_config.profiler_config.torch_profiler_dir
            logger.info(
                "Torch profiler enabled. AsyncLLM CPU traces will be collected under %s",  # noqa: E501
                profiler_dir,
            )
            worker_name = f"{socket.gethostname()}_{os.getpid()}.async_llm"
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                ],
                with_stack=vllm_config.profiler_config.torch_profiler_with_stack,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    profiler_dir,
                    worker_name=worker_name,
                    use_gzip=vllm_config.profiler_config.torch_profiler_use_gzip,
                ),
            )
        else:
            self.profiler = None

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
        enable_log_requests: bool = False,
        aggregate_engine_logging: bool = False,
        disable_log_stats: bool = False,
        client_addresses: dict[str, str] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ) -> "AsyncLLM":
        # Create the LLMEngine.
        return cls(
            vllm_config=vllm_config,
            executor_class=Executor.get_class(vllm_config),
            start_engine_loop=start_engine_loop,
            stat_loggers=stat_loggers,
            log_requests=enable_log_requests,
            log_stats=not disable_log_stats,
            aggregate_engine_logging=aggregate_engine_logging,
            usage_context=usage_context,
            client_addresses=client_addresses,
            client_count=client_count,
            client_index=client_index,
        )

    @classmethod
    def from_engine_args(
        cls,
        engine_args: AsyncEngineArgs,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
    ) -> "AsyncLLM":
        """Create an AsyncLLM from the EngineArgs."""

        # Create the engine configs.
        vllm_config = engine_args.create_engine_config(usage_context)
        executor_class = Executor.get_class(vllm_config)

        # Create the AsyncLLM.
        return cls(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_requests=engine_args.enable_log_requests,
            log_stats=not engine_args.disable_log_stats,
            start_engine_loop=start_engine_loop,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
        )

    def __del__(self):
        self.shutdown()

    def shutdown(self):
        """Shutdown, cleaning up the background proc and IPC."""

        shutdown_prometheus()

        if engine_core := getattr(self, "engine_core", None):
            engine_core.shutdown()

        handler = getattr(self, "output_handler", None)
        if handler is not None:
            cancel_task_threadsafe(handler)

    async def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return await self.engine_core.get_supported_tasks_async()

    async def add_request(
        self,
        request_id: str,
        prompt: EngineCoreRequest | PromptType,
        params: SamplingParams | PoolingParams,
        arrival_time: float | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
        prompt_text: str | None = None,
    ) -> RequestOutputCollector:
        """Add new request to the AsyncLLM."""

        if self.errored:
            raise EngineDeadError()

        is_pooling = isinstance(params, PoolingParams)

        # Convert Input --> Request.
        if isinstance(prompt, EngineCoreRequest):
            request = prompt
            if request_id != request.request_id:
                logger.warning_once(
                    "AsyncLLM.add_request() was passed a request_id parameter that "
                    "does not match the EngineCoreRequest.request_id attribute. The "
                    "latter will be used, and the former will be ignored."
                )
        else:
            assert prompt_text is None
            request = self.input_processor.process_inputs(
                request_id,
                prompt,
                params,
                arrival_time,
                lora_request,
                tokenization_kwargs,
                trace_headers,
                priority,
                data_parallel_rank,
            )
            if isinstance(prompt, str):
                prompt_text = prompt
            elif isinstance(prompt, Mapping):
                prompt_text = cast(str | None, prompt.get("prompt"))

        self.input_processor.assign_request_id(request)

        # Create a new output collector for the request.
        #每个请求都会对应一个独立的实例queue,这个queue就是后续用户async for out in response拉取结果的地方
        queue = RequestOutputCollector(params.output_kind, request.request_id)

        # Use cloned params that may have been updated in process_inputs()
        params = request.params

        #单输出模式（n=1或pooling任务）
        if is_pooling or params.n == 1:
            await self._add_request(request, prompt_text, None, 0, queue)
            return queue

        parent_params = params
        assert isinstance(parent_params, SamplingParams)

        #多输出模式（n>1）
        # Fan out child requests (for n>1).
        parent_request = ParentRequest(request)
        for idx in range(parent_params.n):
            request_id, child_params = parent_request.get_child_info(idx)
            child_request = request if idx == parent_params.n - 1 else copy(request)
            child_request.request_id = request_id
            child_request.sampling_params = child_params
            await self._add_request(
                child_request, prompt_text, parent_request, idx, queue
            )
        return queue
    #真正把一个请求加入调度和处理流程的核心方法
    async def _add_request(
        self,
        request: EngineCoreRequest,
        prompt: str | None,
        parent_req: ParentRequest | None,
        index: int,
        queue: RequestOutputCollector,
    ):
        """
        负责把用户请求同时注册到2个关键组件：
        1.当前进程的OutputProcessor（输出处理器，负责合并、格式化、推送到用户stream）
        2.EngineCore（推理引擎核心，负责调度、预填充、解码、kv cache管理等）
        一句话总结：
        把一个EngineCoreRequest（已经处理好的请求）同时投递给输出处理器和引擎核心，让请求开始被真正调度和执行，
        同时把输出收集器（queue）绑定上去，方便后续用户拉取结果

        用户调用 generate() / chat()
            ↓
        准备 EngineCoreRequest + RequestOutputCollector
            ↓
        调用 _add_request(request, prompt, parent, index, queue)
            ├── 1. output_processor.add_request(...)   # 当前进程：注册输出路径
            └── 2. await engine_core.add_request_async(...)  # 跨进程：把请求发给推理核心
            ↓
        引擎开始调度 → 产生输出 → OutputProcessor 处理 → put 到 queue
            ↓
        用户 async for out in response:   # 从 queue.get() 拉取


        为啥2个add_request一个异步 一个同步？
        AsyncLLM / AsyncEngine（前端，运行在主进程，通常是 API Server 进程）
        负责接收 HTTP/gRPC 请求、预处理输入、创建 RequestOutputCollector、协调输出流
        用 asyncio 事件循环处理并发

        OutputProcessor（也在主进程里，和 AsyncLLM 同进程）
        负责把 EngineCore 发回的原始输出（EngineCoreOutputs）转成用户友好的 RequestOutput
        处理合并（n>1）、格式化、logprobs、detokenize 等
        把结果 put 到每个请求的 collector queue

        EngineCore（运行在独立进程，通常是多进程/多 GPU 布局）
        真正的推理核心：调度器、连续批处理、预填充、解码、KV cache 管理、GPU forward
        性能敏感、重计算部分，独立进程避免阻塞主进程

        主进程（AsyncLLM + OutputProcessor）和 EngineCore 进程之间通过 IPC（如 multiprocessing.Queue、ZeroMQ、Ray 等）通信

       一个同步一个异步的原因是OutputProcessor和AsyncLLM在同一个进程，只是普通的方法调用（内存操作、字典插入、列表append等）
       EngineCore在另一个独立进程，需要通过IPC（如队列put）把request对象序列化/发送出去

        """
        # Add the request to OutputProcessor (this process).
        self.output_processor.add_request(request, prompt, parent_req, index, queue)

        # Add the EngineCoreRequest to EngineCore (separate process).
        await self.engine_core.add_request_async(request)

        if self.log_requests:
            logger.info("Added request %s.", request.request_id)

    # TODO: we should support multiple prompts in one call, as you
    # can do with LLM.generate. So that for multi-prompt completion
    # requests we don't need to send multiple messages to core proc,
    # and so we don't need multiple streams which then get
    # re-multiplexed in the API server anyhow.
    #目前这个接口一次只能处理一个prompt， 理想目标是一次请求船多个prompt，
    async def generate(
        self,
        prompt: EngineCoreRequest | PromptType,
        sampling_params: SamplingParams,
        request_id: str,
        *,
        prompt_text: str | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
    ) -> AsyncGenerator[RequestOutput, None]:
        """
        Main function called by the API server to kick off a request
            * 1) Making an AsyncStream corresponding to the Request.
            * 2) Processing the Input.
            * 3) Adding the Request to the Detokenizer.
            * 4) Adding the Request to the EngineCore (separate process).

        A separate output_handler loop runs in a background AsyncIO task,
        pulling outputs from EngineCore and putting them into the
        per-request AsyncStream.

        The caller of generate() iterates the returned AsyncGenerator,
        returning the RequestOutput back to the caller.
        数文档字符串，说明这是API服务器调用的主要函数，流程包括：
        创建与请求对应的异步流
        处理输入
        将请求添加到去分词器
        将请求添加到独立的 EngineCore 进程
        一个单独的 output_handler循环在后台运行
        从 EngineCore 拉取输出并放入每个请求的异步流中
        调用者通过迭代异步生成器获取 RequestOutput
        """

        if (
            self.vllm_config.cache_config.kv_sharing_fast_prefill
            and sampling_params.prompt_logprobs
        ):
            #如果开启了kv-sharing-fast-prefill（一种加速prefill的优化，但是用户要logprobs->直接报错）
            #因为fast prefill会牺牲prompt logprobs的准确性，因为是跳过计算或简化计算
            raise ValueError(
                "--kv-sharing-fast-prefill produces incorrect logprobs for "
                "prompt tokens, please disable it when the requests need "
                "prompt logprobs"
            )

        q: RequestOutputCollector | None = None
        try:
            # We start the output_handler on the first call to generate() so
            # we can call __init__ before the event loop, which enables us
            # to handle startup failure gracefully in the OpenAI server.
            #第一次调用 generate() 时才启动 output_handler 协程
            #这个协程一直在后台运行：从 EngineCore 拉取原始输出 → OutputProcessor 处理 → put 到各个请求的 queue
            #为什么延迟启动？因为 __init__ 可能在同步上下文调用，先启动 handler 能更好地处理启动失败
            self._run_output_handler()

            # Wait until generation is resumed if the engine is paused.
            #等待引擎回复 如果被暂停（比如RL训练时PAUSE），就挂起等待resume
            async with self._pause_cond:
                await self._pause_cond.wait_for(lambda: not self._paused)

            if tokenization_kwargs is None:
                tokenization_kwargs = {}
                truncate_prompt_tokens = sampling_params.truncate_prompt_tokens

                _validate_truncation_size(
                    self.model_config.max_model_len,
                    truncate_prompt_tokens,
                    tokenization_kwargs,
                )

            #返回的q就是这个请求专属的输出收集器
            q = await self.add_request(
                request_id,
                prompt,
                sampling_params,
                lora_request=lora_request,
                tokenization_kwargs=tokenization_kwargs,
                trace_headers=trace_headers,
                priority=priority,
                data_parallel_rank=data_parallel_rank,
                prompt_text=prompt_text,
            )

            # The output_handler task pushes items into the queue.
            # This task pulls from the queue and yields to caller.
            finished = False
            while not finished:
                # Note: drain queue without await if possible (avoids
                # task switching under load which helps performance).
                out = q.get_nowait() or await q.get()

                # Note: both OutputProcessor and EngineCore handle their
                # own request cleanup based on finished.
                finished = out.finished
                assert isinstance(out, RequestOutput)
                yield out

        # If the request is disconnected by the client, generate()
        # is cancelled or the generator is garbage collected. So,
        # we abort the request if we end up here.
        #浏览器关掉、客户端取消请求 → 立即 abort 请求，释放 KV cache 等资源
        except (asyncio.CancelledError, GeneratorExit):
            if q is not None:
                await self.abort(q.request_id, internal=True)
            if self.log_requests:
                logger.info("Request %s aborted.", request_id)
            raise

        # Engine is dead. Do not abort since we shut down.
        #引擎挂了 → 不 abort(因为已经死了)，直接抛给用户
        except EngineDeadError:
            if self.log_requests:
                logger.info("Request %s failed (engine dead).", request_id)
            raise

        # Request validation error.
        #参数校验错误
        except ValueError:
            if self.log_requests:
                logger.info("Request %s failed (bad request).", request_id)
            raise

        # Unexpected error in the generate() task (possibly recoverable).
        #其他意外错误
        except Exception as e:
            if q is not None:
                await self.abort(q.request_id, internal=True)
            if self.log_requests:
                logger.info("Request %s failed.", request_id)
            raise EngineGenerateError() from e

    def _run_output_handler(self):
        """Background loop: pulls from EngineCore and pushes to AsyncStreams."""

        if self.output_handler is not None:
            return

        # Ensure that the task doesn't have a circular ref back to the AsyncLLM
        # object, or else it won't be garbage collected and cleaned up properly.
        #确保task不会通过循环引用指向asyncLLM对象，否则它讲无法被垃圾回收并正确清理
        engine_core = self.engine_core
        output_processor = self.output_processor
        log_stats = self.log_stats
        logger_manager = self.logger_manager
        input_processor = self.input_processor

        async def output_handler():
            """
            一个异步无限循环，是vllm异步引擎中负责从EngineCore拉取原始输出->处理->分发给用户queue的核心后台任务
            """
            try:
                while True: #永不结束的循环，只要引擎活着，它就一直在后台运行
                    # 1) Pull EngineCoreOutputs from the EngineCore.
                    outputs = await engine_core.get_output_async() #从EngieCore中异步拉取一批已经完成的输出 EngineCoreOutputs
                    num_outputs = len(outputs.outputs) #这次拉取到的输出数量（每个output对应一个请求的一次生成chunk）

                    #如果开启了统计日志（log_stats=True）且有输出，就创建一个新的 IterationStats 对象，用于记录本次迭代的各种指标（TTFT、TPOT、KV cache 使用等）
                    iteration_stats = (
                        IterationStats() if (log_stats and num_outputs) else None
                    )

                    # Split outputs into chunks of at most
                    # VLLM_V1_OUTPUT_PROC_CHUNK_SIZE, so that we don't block the
                    # event loop for too long.
                    #翻译：将输出拆分成不超过VLLM_V1_OUTPUT_PROC_CHUNK_SIZE的小块，这样就不会长时间阻塞事件循环
                    if num_outputs <= envs.VLLM_V1_OUTPUT_PROC_CHUNK_SIZE:
                        slices = (outputs.outputs,)
                    else:
                        slices = np.array_split(
                            outputs.outputs,
                            cdiv(num_outputs, envs.VLLM_V1_OUTPUT_PROC_CHUNK_SIZE),
                        )

                    #把一批模型输出分成若干小块，一块一块地处理。每处理一块，就主动让出event loop一次，防止系统卡死
                    #同时，如果发现有请求因为stop tokens就提前结束，就立刻通知引擎终止它们
                    for i, outputs_slice in enumerate(slices):
                        # 2) Process EngineCoreOutputs.
                        #process_outputs实际做了 把模型原始输出->转换成用户可读的请求结果+控制信号
                        #里面通常包括detokenize ;stop string 判断; EOS判断; 生成RequestOutput;判断哪些请求可以中止
                        processed_outputs = output_processor.process_outputs(
                            outputs_slice, outputs.timestamp, iteration_stats
                        )
                        # NOTE: RequestOutputs are pushed to their queues.
                        #异步下，RequestOutpt已经被put到queue了，所以processed_outputs.request_outputs 应该是空的（同步模式 LLMEngine 才会收集到 list）
                        assert not processed_outputs.request_outputs

                        # Allow other asyncio tasks to run between chunks
                        #非常关键的让出控制权操作，在处理完小 chunk 后，主动 await asyncio.sleep(0)（相当于 yield），让事件循环有机会调度其他任务（新请求进来、HTTP 响应等）
                        if i + 1 < len(slices):
                            await asyncio.sleep(0)

                        # 3) Abort any reqs that finished due to stop strings.
                        await engine_core.abort_requests_async(processed_outputs.reqs_to_abort)

                    output_processor.update_scheduler_stats(outputs.scheduler_stats)

                    # 4) Logging.
                    # TODO(rob): make into a coroutine and launch it in
                    # background thread once Prometheus overhead is non-trivial.
                    if logger_manager:
                        logger_manager.record(
                            engine_idx=outputs.engine_index,
                            scheduler_stats=outputs.scheduler_stats,
                            iteration_stats=iteration_stats,
                            mm_cache_stats=input_processor.stat_mm_cache(),
                        )
            except Exception as e:
                logger.exception("AsyncLLM output_handler failed.")
                output_processor.propagate_error(e)

        self.output_handler = asyncio.create_task(output_handler())

    async def abort(
        self, request_id: str | Iterable[str], internal: bool = False
    ) -> None:
        """Abort RequestId in OutputProcessor and EngineCore."""

        request_ids = (
            (request_id,) if isinstance(request_id, str) else as_list(request_id)
        )
        all_request_ids = self.output_processor.abort_requests(request_ids, internal)
        await self.engine_core.abort_requests_async(all_request_ids)

        if self.log_requests:
            logger.info("Aborted request(s) %s.", ",".join(request_ids))

    async def pause_generation(
        self,
        *,
        wait_for_inflight_requests: bool = False, #是否等待当前正在飞的请求，自然完成后再暂停
        clear_cache: bool = True,#暂停后是否清空kv cache和prefix cache
    ) -> None:
        """
        Pause generation to allow model weight updates.

        New generation/encoding requests are blocked until resume.

        Args:
            wait_for_inflight_requests: When ``True`` waits for in-flight
                requests to finish before pausing. When ``False`` (default),
                immediately aborts any in-flight requests.
            clear_cache: Whether to clear KV cache and prefix cache after
                draining. Set to ``False`` to preserve cache for faster resume.
                Default is ``True`` (clear caches).
        在需要更新权重时，安全地暂停所有正在进行的生成任务，让系统进入”静止“状态，以便后续进行权重更新或资源释放。

        """

        async with self._pause_cond:
            if self._paused:
                return
            self._paused = True

        if not wait_for_inflight_requests:
            request_ids = list(self.output_processor.request_states.keys())
            if request_ids:
                await self.abort(request_ids, internal=True)

        # Wait for running requests to drain before clearing cache.
        if self.output_processor.has_unfinished_requests():
            await self.output_processor.wait_for_requests_to_drain()

        # Clear cache
        if clear_cache:
            await self.reset_prefix_cache()
            await self.reset_mm_cache()

    async def resume_generation(self) -> None:
        """Resume generation after :meth:`pause_generation`."""

        async with self._pause_cond:
            self._paused = False
            self._pause_cond.notify_all()  # Wake up all waiting requests

    async def is_paused(self) -> bool:
        """Return whether the engine is currently paused."""

        async with self._pause_cond:
            return self._paused

    async def encode(
        self,
        prompt: PromptType,
        pooling_params: PoolingParams,
        request_id: str,
        lora_request: LoRARequest | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        truncate_prompt_tokens: int | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> AsyncGenerator[PoolingRequestOutput, None]:
        """
        Main function called by the API server to kick off a request
            * 1) Making an AsyncStream corresponding to the Request.
            * 2) Processing the Input.
            * 3) Adding the Request to the EngineCore (separate process).

        A separate output_handler loop runs in a background AsyncIO task,
        pulling outputs from EngineCore and putting them into the
        per-request AsyncStream.

        The caller of generate() iterates the returned AsyncGenerator,
        returning the RequestOutput back to the caller.

        NOTE: truncate_prompt_tokens is deprecated in v0.14.
        TODO: Remove truncate_prompt_tokens in v0.15.
        """

        q: RequestOutputCollector | None = None
        try:
            # We start the output_handler on the first call to generate() so
            # we can call __init__ before the event loop, which enables us
            # to handle startup failure gracefully in the OpenAI server.
            self._run_output_handler()

            # Respect pause state before accepting new requests.
            async with self._pause_cond:
                await self._pause_cond.wait_for(lambda: not self._paused)

            if tokenization_kwargs is None:
                tokenization_kwargs = {}

            if truncate_prompt_tokens is not None:
                warnings.warn(
                    "The `truncate_prompt_tokens` parameter in `AsyncLLM.encode()` "
                    "is deprecated and will be removed in v0.15. "
                    "Please use `pooling_params.truncate_prompt_tokens` instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )

            _validate_truncation_size(
                self.model_config.max_model_len,
                pooling_params.truncate_prompt_tokens,
                tokenization_kwargs,
            )

            q = await self.add_request(
                request_id,
                prompt,
                pooling_params,
                lora_request=lora_request,
                tokenization_kwargs=tokenization_kwargs,
                trace_headers=trace_headers,
                priority=priority,
            )

            # The output_handler task pushes items into the queue.
            # This task pulls from the queue and yields to caller.
            finished = False
            while not finished:
                # Note: drain queue without await if possible (avoids
                # task switching under load which helps performance).
                out = q.get_nowait() or await q.get()
                assert isinstance(out, PoolingRequestOutput)
                # Note: both OutputProcessor and EngineCore handle their
                # own request cleanup based on finished.
                finished = out.finished
                yield out

        # If the request is disconnected by the client, generate()
        # is cancelled. So, we abort the request if we end up here.
        except asyncio.CancelledError:
            if q is not None:
                await self.abort(q.request_id, internal=True)
            if self.log_requests:
                logger.info("Request %s aborted.", request_id)
            raise

        # Engine is dead. Do not abort since we shut down.
        except EngineDeadError:
            if self.log_requests:
                logger.info("Request %s failed (engine dead).", request_id)
            raise

        # Request validation error.
        except ValueError:
            if self.log_requests:
                logger.info("Request %s failed (bad request).", request_id)
            raise

        # Unexpected error in the generate() task (possibly recoverable).
        except Exception as e:
            if q is not None:
                await self.abort(q.request_id, internal=True)
            if self.log_requests:
                logger.info("Request %s failed.", request_id)
            raise EngineGenerateError() from e

    @property
    def tokenizer(self) -> TokenizerLike | None:
        return self.input_processor.tokenizer

    async def get_tokenizer(self) -> TokenizerLike:
        if self.tokenizer is None:
            raise ValueError(
                "Unable to get tokenizer because `skip_tokenizer_init=True`"
            )

        return self.tokenizer

    async def is_tracing_enabled(self) -> bool:
        return self.observability_config.otlp_traces_endpoint is not None  # type: ignore

    async def do_log_stats(self) -> None:
        if self.logger_manager:
            self.logger_manager.log()

    async def check_health(self) -> None:
        logger.debug("Called check_health.")
        if self.errored:
            raise self.dead_error

    async def start_profile(self) -> None:
        coros = [self.engine_core.profile_async(True)]
        if self.profiler is not None:
            coros.append(asyncio.to_thread(self.profiler.start))
        await asyncio.gather(*coros)

    async def stop_profile(self) -> None:
        coros = [self.engine_core.profile_async(False)]
        if self.profiler is not None:
            coros.append(asyncio.to_thread(self.profiler.stop))
        await asyncio.gather(*coros)

    async def reset_mm_cache(self) -> None:
        self.input_processor.clear_mm_cache()
        await self.engine_core.reset_mm_cache_async()

    async def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        return await self.engine_core.reset_prefix_cache_async(
            reset_running_requests, reset_connector
        )

    async def sleep(self, level: int = 1) -> None:
        await self.reset_prefix_cache()
        await self.engine_core.sleep_async(level)

        if self.logger_manager is not None:
            self.logger_manager.record_sleep_state(1, level)

    async def wake_up(self, tags: list[str] | None = None) -> None:
        await self.engine_core.wake_up_async(tags)

        if self.logger_manager is not None:
            self.logger_manager.record_sleep_state(0, 0)

    async def is_sleeping(self) -> bool:
        return await self.engine_core.is_sleeping_async()

    async def add_lora(self, lora_request: LoRARequest) -> bool:
        """Load a new LoRA adapter into the engine for future requests."""
        return await self.engine_core.add_lora_async(lora_request)

    async def remove_lora(self, lora_id: int) -> bool:
        """Remove an already loaded LoRA adapter."""
        return await self.engine_core.remove_lora_async(lora_id)

    async def list_loras(self) -> set[int]:
        """List all registered adapters."""
        return await self.engine_core.list_loras_async()

    async def pin_lora(self, lora_id: int) -> bool:
        """Prevent an adapter from being evicted."""
        return await self.engine_core.pin_lora_async(lora_id)

    async def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
    ):
        """
        Perform a collective RPC call to the given path.
        """
        return await self.engine_core.collective_rpc_async(
            method, timeout, args, kwargs
        )

    async def wait_for_requests_to_drain(self, drain_timeout: int = 300):
        """Wait for all requests to be drained."""
        start_time = time.time()
        while time.time() - start_time < drain_timeout:
            if not self.engine_core.dp_engines_running():
                logger.info("Engines are idle, requests have been drained")
                return

            logger.info("Engines are still running, waiting for requests to drain...")
            await asyncio.sleep(1)  # Wait 1 second before checking again

        raise TimeoutError(
            f"Timeout reached after {drain_timeout} seconds "
            "waiting for requests to drain."
        )

    async def scale_elastic_ep(
        self, new_data_parallel_size: int, drain_timeout: int = 300
    ):
        """
        Scale up or down the data parallel size by adding or removing
        engine cores.
        Args:
            new_data_parallel_size: The new number of data parallel workers
            drain_timeout:
                Maximum time to wait for requests to drain (seconds)
        """
        old_data_parallel_size = self.vllm_config.parallel_config.data_parallel_size
        if old_data_parallel_size == new_data_parallel_size:
            logger.info(
                "Data parallel size is already %s, skipping scale",
                new_data_parallel_size,
            )
            return
        logger.info(
            "Waiting for requests to drain before scaling up to %s engines...",
            new_data_parallel_size,
        )
        await self.wait_for_requests_to_drain(drain_timeout)
        logger.info(
            "Requests have been drained, proceeding with scale to %s engines",
            new_data_parallel_size,
        )
        await self.engine_core.scale_elastic_ep(new_data_parallel_size)
        self.vllm_config.parallel_config.data_parallel_size = new_data_parallel_size

        # recreate stat loggers
        if new_data_parallel_size > old_data_parallel_size and self.log_stats:
            # TODO(rob): fix this after talking with Ray team.
            # This resets all the prometheus metrics since we
            # unregister during initialization. Need to understand
            # the intended behavior here better.
            self.logger_manager = StatLoggerManager(
                vllm_config=self.vllm_config,
                engine_idxs=list(range(new_data_parallel_size)),
                custom_stat_loggers=None,
            )

    @property
    def is_running(self) -> bool:
        # Is None before the loop is started.
        return self.output_handler is None or not self.output_handler.done()

    @property
    def is_stopped(self) -> bool:
        return self.errored

    @property
    def errored(self) -> bool:
        return self.engine_core.resources.engine_dead or not self.is_running

    @property
    def dead_error(self) -> BaseException:
        return EngineDeadError()
