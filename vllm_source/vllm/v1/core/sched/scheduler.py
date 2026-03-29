# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
import time
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from vllm import envs
from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorMetadata,
    ECConnectorRole,
)
from vllm.distributed.ec_transfer.ec_connector.factory import ECConnectorFactory
from vllm.distributed.kv_events import EventPublisherFactory, KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.encoder_cache_manager import (
    EncoderCacheManager,
    EncoderDecoderCacheManager,
    compute_encoder_budget,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.output import (
    CachedRequestData,
    GrammarOutput,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.core.sched.utils import check_stop, remove_all
from vllm.v1.engine import EngineCoreEventType, EngineCoreOutput, EngineCoreOutputs
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.perf import ModelMetrics, PerfStats
from vllm.v1.metrics.stats import (
    PrefixCacheStats,
    SchedulerStats,
)
from vllm.v1.outputs import DraftTokenIds, KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.utils import record_function_or_nullcontext

logger = init_logger(__name__)


class Scheduler(SchedulerInterface):
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,  #KV cache 的层级规格与 block 信息
        structured_output_manager: StructuredOutputManager,#管理模型输出结构化信息的对象
        block_size: int, #KV cache block 的 token 数
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY, #多模态注册表，用于处理 multimodal 输入。
        include_finished_set: bool = False,#是否记录每步完成的 request id
        log_stats: bool = False, #是否收集日志/性能指标
    ) -> None:
        # ========== 全局配置 ==========
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config             #调度策略配置（batch_size,策略等）
        self.cache_config = vllm_config.cache_config                     #kv cache缓存配置
        self.lora_config = vllm_config.lora_config                       #lora相关配置
        #========= KV Cache ==========
        #KV cache 是推理性能的核心（避免重复计算 attention）
        self.kv_cache_config = kv_cache_config
        #KV cache 事件（比如命中率、evict 等）
        self.kv_events_config = vllm_config.kv_events_config

        # ========== 并行相关 ==========
        # 包括 tensor parallel / pipeline parallel / data parallel
        self.parallel_config = vllm_config.parallel_config

        # ========== 日志 / 监控 ==========
        self.log_stats = log_stats  #是否记录调度过程中统计信息
        self.observability_config = vllm_config.observability_config

        #kv cache监控器（统计命中率、使用量等）
        self.kv_metrics_collector: KVCacheMetricsCollector | None = None
        if self.observability_config.kv_cache_metrics:
            self.kv_metrics_collector = KVCacheMetricsCollector(
                self.observability_config.kv_cache_metrics_sample,
            )
        # ========== 结构化输出 ==========
        # 用于支持 JSON schema / grammar decoding
        self.structured_output_manager = structured_output_manager
        # ========== 模型类型 ==========
        # 是否是 encoder-decoder（比如 T5 / 多模态模型）
        self.is_encoder_decoder = vllm_config.model_config.is_encoder_decoder

        # include_finished_set controls whether a separate set of finished         include_finished_set 控制：是否额外维护一个“已完成请求ID集合”
        # request ids should be included in the EngineCoreOutputs returned         这个集合会在 update_from_outputs() 返回的 EngineCoreOutputs 中带出来
        # by update_from_outputs(). This is currently used in the multi-engine     主要用于多引擎（multi-engine）场景下，快速跟踪请求生命周期（谁结束了）
        # case to track request lifetimes efficiently.
        self.finished_req_ids_dict: dict[int, set[str]] | None = (
            defaultdict(set) if include_finished_set else None
        )
        # 上一轮调度(step)中，被安排执行的 request id 集合    用途：避免重复调度 / 做差分调度 / 做调度优化
        self.prev_step_scheduled_req_ids: set[str] = set()

        # =========================
        #   调度约束（非常核心）
        # =========================
        # Scheduling constraints.
        #同时最多运行多少个请求（类似并发请求上限）  ，防止现存爆炸或batch太大导致性能下降
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        # 一次调度最多允许的 token 数 ， 控制GPU一次forward的工作量（吞吐vs 延迟的关键参数）
        self.max_num_scheduled_tokens = self.scheduler_config.max_num_batched_tokens

        #模型最大支持的序列长度（prompt + 生成）
        self.max_model_len = vllm_config.model_config.max_model_len

        #是否开启 KV cache 事件通知（用于监控 / profiling / debug）
        self.enable_kv_cache_events = (  #是否启用 KV cache 事件通知。
            self.kv_events_config is not None
            and self.kv_events_config.enable_kv_cache_events
        )
        # =========================
        #   KV Cache 远程传输（高级特性）
        # =========================
        # KV Connector：用于跨机器 / 跨进程共享 KV cache
        # 👉 典型用途：
        #    - Prefill/Decode 分离（P/D disaggregation）
        #    - KV cache offload（显存不够时转移）
        # Create KVConnector for the Scheduler. Note that each Worker
        # will have a corresponding KVConnector with Role=WORKER.
        # KV Connector pushes/pull of remote KVs for P/D and offloading.
        self.connector = None
        # 用于统计 prefix cache 命中率等指标（可观测性）
        self.connector_prefix_cache_stats: PrefixCacheStats | None = None
        # 当 KV 加载失败时是否重新计算（fallback 策略）  重新算一遍（安全但慢）  False = 直接失败
        self.recompute_kv_load_failures = True

        if self.vllm_config.kv_transfer_config is not None:
            #当前不支持 encoder-decoder 模型（比如 T5）使用 KV connector
            assert not self.is_encoder_decoder, (
                "Encoder-decoder models are not currently supported with KV connectors"
            )
            # 创建 KV Connector（Scheduler 侧）  Worker 侧也会有一个对应的 connector
            self.connector = KVConnectorFactory.create_connector(
                config=self.vllm_config,
                role=KVConnectorRole.SCHEDULER,
                kv_cache_config=self.kv_cache_config,
            )
            # 如果开启日志统计，则初始化 prefix cache 统计器
            if self.log_stats:
                self.connector_prefix_cache_stats = PrefixCacheStats()
            # KV 加载失败策略（配置项）
            kv_load_failure_policy = (
                self.vllm_config.kv_transfer_config.kv_load_failure_policy
            )
            # 是否在 KV load 失败时重新计算
            self.recompute_kv_load_failures = kv_load_failure_policy == "recompute"

        # =========================
        #   KV Cache 事件发布（监控用）
        # =========================
        #用于发布kv cache的生命周期事件（创建、命中、释放等）
        #常用于性能分析、监控系统、DEBUG
        self.kv_event_publisher = EventPublisherFactory.create( #发布 KV cache 事件。
            self.kv_events_config,
            self.parallel_config.data_parallel_rank,
        )

        # =========================
        #   Encoder Cache（多模态 / 编码器模型）
        # =========================
        # EC = Encoder Cache（和 KV cache 类似，但用于 encoder 输出）
        self.ec_connector = None  #ec_connector用于 encoder cache 的远程传输。
        if self.vllm_config.ec_transfer_config is not None:
            #创建Encoder Cache的远程传输connector
            self.ec_connector = ECConnectorFactory.create_connector(
                config=self.vllm_config, role=ECConnectorRole.SCHEDULER
            )
        # =========================
        #   KV Cache 基础信息
        # =========================
        # GPU 上可用的 KV cache block 数量（核心资源） 每个 block 可以存一段 token 的 KV（注意力缓存）
        num_gpu_blocks = self.cache_config.num_gpu_blocks
        # 必须保证有 KV cache，否则模型无法做高效推理
        assert num_gpu_blocks is not None and num_gpu_blocks > 0


        # 每个 KV block 里包含多少 token，block_size 越大：管理简单但可能浪费  block_size 越小：更灵活但管理开销大
        self.block_size = block_size

        # Decode 阶段的 context parallel（上下文并行），用于生成阶段（一个 token 一个 token 推）
        self.dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size

        # Prefill 阶段的 context parallel 用于 prompt 一次性处理（prefill）
        self.pcp_world_size = vllm_config.parallel_config.prefill_context_parallel_size

        # =========================
        #   请求管理（核心数据结构）
        # =========================
        # request_id -> Request 对象  存所有“系统里存在的请求”（无论在排队还是运行）
        self.requests: dict[str, Request] = {}

        # =========================
        #   调度策略（非常关键）
        # =========================
        # 根据配置创建调度策略（比如 FCFS / priority / fairness）
        try:
            self.policy = SchedulingPolicy(self.scheduler_config.policy)
        except ValueError as e:
            raise ValueError(
                f"Unknown scheduling policy: {self.scheduler_config.policy}"
            ) from e

        # =========================
        #   请求队列（核心中的核心）
        # =========================
        # waiting 队列（等待调度的请求） 本质：还没上 GPU 的请求 根据 policy 决定是 FIFO / 优先级队列 / 其他策略
        self.waiting = create_request_queue(self.policy)

        # running 列表（正在 GPU 上执行的请求） 这些请求已经占用了 KV cache + GPU 资源
        self.running: list[Request] = []

        # =========================
        #   生命周期管理（非常重要）
        # =========================
        # 在“上一轮 step → 当前 step”之间完成的请求  👉 用途：通知 Worker 可以释放 KV cache / 显存，👉 每个 step 结束后会清空
        # The request IDs that are finished in between the previous and the
        # current steps. This is used to notify the workers about the finished
        # requests so that they can free the cached states for those requests.
        # This is flushed at the end of each scheduling step.
        self.finished_req_ids: set[str] = set()         #本步完成的请求

        # =========================
        #   KV Connector（远程 KV 相关）
        # =========================
        #正在接收kv cache的请求（异步过程），比如从别的机器加载KV（prefill 结果复用）,
        # KV Connector: requests in process of async KV loading or recving
        self.finished_recving_kv_req_ids: set[str] = set()

        #接收KV失败的请求->后续可能需要fallback（重新计算）
        self.failed_recving_kv_req_ids: set[str] = set()

        # =========================
        #   Encoder 相关配置
        # =========================

        # Encoder-related.
        # Calculate encoder cache size if applicable   计算 encoder cache 的大小（如果模型有 encoder）
        # NOTE: For now we use the same budget for both compute and space. 注意：目前对 encoder，我们在计算时同时用同样的预算限制计算和缓存空间
        # This can be changed when we make encoder cache for embedding caching 以后如果为跨请求的 embedding 缓存单独设置 encoder cache，可以调整这个逻辑。
        # across requests.                                                      比如说是T5等模型
        encoder_compute_budget, encoder_cache_size = compute_encoder_budget(
            model_config=vllm_config.model_config,
            scheduler_config=vllm_config.scheduler_config,
            mm_registry=mm_registry,
        )

        # NOTE(woosuk): Here, "encoder" includes the vision encoder (and
        # projector if needed) for MM models as well as encoder-decoder
        # transformers. 计算预算（最多允许处理多少encoder token）-> 防止encoder输入过大导致算力/显存爆炸
        self.max_num_encoder_input_tokens = encoder_compute_budget

        #encoder cache管理器， 用来缓存encoder的输出（类似KV cache， 但用于encoder）
        # NOTE: For the models without encoder (e.g., text-only models),
        # the encoder cache will not be initialized because cache size is 0
        # for these models.
        self.encoder_cache_manager = (
            EncoderDecoderCacheManager(cache_size=encoder_cache_size)
            if self.is_encoder_decoder
            else EncoderCacheManager(cache_size=encoder_cache_size)
        )
        #对于encoder-decoder模型，预分配encoder最大token数，为什么？像 Whisper 这种模型：输入会被 pad 到固定最大长度， cross-attention需要提前分配好kv 空间
        # For encoder-decoder models, allocate the maximum number of tokens for Cross
        # Attn blocks, as for Whisper its input is always padded to the maximum length.
        # TODO (NickLucche): Generalize to models with variable-length encoder inputs.  #未来支持可变长度encoder
        self._num_encoder_max_input_tokens = (
            MULTIMODAL_REGISTRY.get_encdec_max_encoder_len(vllm_config.model_config)
        )

        # =========================
        #   推测解码（Speculative Decoding）
        # =========================
        speculative_config = vllm_config.speculative_config
        #是否使用EAGLE
        self.use_eagle = False
        #speculative token数
        self.num_spec_tokens = self.num_lookahead_tokens = 0
        if speculative_config:
            self.num_spec_tokens = speculative_config.num_speculative_tokens
            if speculative_config.use_eagle():
                self.use_eagle = True
                self.num_lookahead_tokens = self.num_spec_tokens

        #负责管理transformer的kv cache （注意力缓存）
        # Create the KV cache manager.
        self.kv_cache_manager = KVCacheManager(
            #kv cache的结构配置
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            enable_caching=self.cache_config.enable_prefix_caching,
            use_eagle=self.use_eagle,
            log_stats=self.log_stats,
            enable_kv_cache_events=self.enable_kv_cache_events,
            dcp_world_size=self.dcp_world_size,
            pcp_world_size=self.pcp_world_size,
            hash_block_size=self.block_size,
            metrics_collector=self.kv_metrics_collector,
        )
        # =========================
        #   并行执行模式
        # =========================
        # 是否启用 Pipeline Parallel（流水线并行）
        self.use_pp = self.parallel_config.pipeline_parallel_size > 1

        # 是否使用 V2 版本的 ModelRunner（新执行引擎）通常更高效，但可能是实验性特性
        self.use_v2_model_runner = envs.VLLM_USE_V2_MODEL_RUNNER

        # =========================
        #   性能指标（Observability）
        # =========================
        self.perf_metrics: ModelMetrics | None = None  #性能指标

        # 如果开启日志 + 开启 MFU 指标
        if self.log_stats and vllm_config.observability_config.enable_mfu_metrics:
            self.perf_metrics = ModelMetrics(vllm_config)

    def schedule(self) -> SchedulerOutput:
        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.     调度器没有prefill阶段和 decode阶段，每个请求只是维护2个关键数值。
        # Each request just has the num_computed_tokens and                     每个请求只维护2个关键数值：num_computed_tokens：当前已经被计算/处理的 token 数量
        # num_tokens_with_spec. num_tokens_with_spec =       m_tokens_with_spec：该请求“理论上应该已经处理到的”总 token 数量
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).  #num_tokens_with_spec = len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids)
        # At each step, the scheduler tries to assign tokens to the requests    在每个步骤中，调度器会尽量给每个请求分配token，
        # so that each request's num_computed_tokens can catch up its           目标是让每个请求的num_computed_tokens逐步追上它的num_tokens_with_spec
        # num_tokens_with_spec. This is general enough to cover                 这种统一的设计足够够用，能够自然覆盖以下场景：
        # chunked prefills, prefix caching, speculative decoding,               分块与填充；前缀缓存；推测解码；未来可能提出的Jump decoding优化
        # and the "jump decoding" optimization in the future.

        # =========================
        # 一些容器：记录这一轮调度结果
        # =========================
        scheduled_new_reqs: list[Request] = []       # 新加入运行的请求
        scheduled_resumed_reqs: list[Request] = []   # 被抢占后恢复的请求
        scheduled_running_reqs: list[Request] = []   # 原本就在运行的请求
        preempted_reqs: list[Request] = []           # 被抢占踢出去的请求

        req_to_new_blocks: dict[str, KVCacheBlocks] = {}                #记录每个请求本轮分配了多少新的kv cache block
        num_scheduled_tokens: dict[str, int] = {}                       #每个请求这轮分到多少 token
        token_budget = self.max_num_scheduled_tokens                    #本轮GPU最多能计算多少token？
        # Encoder-related.
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_compute_budget = self.max_num_encoder_input_tokens
        # Spec decode-related. 投机解码相关
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}

        # For logging.
        scheduled_timestamp = time.monotonic()

        # First, schedule the RUNNING requests.
        # =========================
        # 🟢 第一阶段：处理已经在运行的请求（running）
        # =========================
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]               #self.running 是本轮已经在 decode 的请求列表。
            #num_prompt_tokens：prompt token 数量; num_computed_tokens:已经真正计算过 forward 的 token 总数（包含 prompt + 所有已经接受的生成 token + 当前这一步 draft 出来的占位 token）
            #num_output_placeholders:当前这一批次里，这个请求被分配了多少个输出占位符（也就是 draft 了多少个新 token，包括可能被 reject 的那些）
            #max_tokens: 用户设置的最大生成长度（不包括 prompt）
            # -------------------------
            # 🚫 判断这个请求是否已经“跑够了”
            # -------------------------
            if (
                request.num_output_placeholders > 0  #最坏情况本轮draft所有token都被reject 但是至少一个还是被接受的
                # This is (num_computed_tokens + 1) - (num_output_placeholders - 1).
                # Since output placeholders are also included in the computed tokens
                # count, we subtract (num_output_placeholders - 1) to remove any draft
                # tokens, so that we can be sure no further steps are needed even if
                # they are all rejected.
                #这里的计算是 (num_computed_tokens + 1) - (num_output_placeholders - 1)。由于由于输出占位符（output placeholders）也被算在已生成 token 数（num_computed_tokens）里面，
                #由于输出占位符（output placeholders）也被算在已生成 token 数（num_computed_tokens）里面，  这样即使这些占位符最终被拒绝，也可以确保这个请求本轮不需要再生成新的 token。
                #另一个1哪来的？
                #
                and request.num_computed_tokens + 2 - request.num_output_placeholders
                >= request.num_prompt_tokens + request.max_tokens
            ):
                # Async scheduling: Avoid scheduling an extra step when we are sure that #当我们确定前一步已经达到request.max_tokens 时，就不再为这个请求调度额外的一步
                # the previous step has reached request.max_tokens. We don't schedule
                # partial draft tokens since this prevents uniform decode optimizations.
                req_index += 1
                continue
            # -------------------------
            # 🎯 计算：这个请求还需要多少 token
            # -------------------------
            num_new_tokens = (
                request.num_tokens_with_spec
                + request.num_output_placeholders
                - request.num_computed_tokens
            )
            # 限制长 prompt（避免一次吃太多）
            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold

            #不能超过当前剩余预算
            num_new_tokens = min(num_new_tokens, token_budget)

            #不能超过模型最大长度
            # Make sure the input position does not exceed the max model len.
            # This is necessary when using spec decoding.  #如果使用 spec decoding，那么需要确保输入位置不超过最大模型长度
            num_new_tokens = min(
                num_new_tokens, self.max_model_len - 1 - request.num_computed_tokens
            )
            # -------------------------
            # 🧠 多模态：是否需要先跑 encoder
            # -------------------------
            # Schedule encoder inputs. #如果是带图的请求，看看能不能先拍encoder计算，如果排了，可能会再把num_new_tokens减小
            encoder_inputs_to_schedule = None
            external_load_encoder_input: list[int] = []
            new_encoder_compute_budget = encoder_compute_budget
            if request.has_encoder_inputs:
                (
                    encoder_inputs_to_schedule,
                    num_new_tokens,
                    new_encoder_compute_budget,
                    external_load_encoder_input,
                ) = self._try_schedule_encoder_inputs(
                    request,
                    request.num_computed_tokens,
                    num_new_tokens,
                    encoder_compute_budget,
                    shift_computed_tokens=1 if self.use_eagle else 0,
                )
            # 如果算下来不能跑了 → 跳过
            if num_new_tokens == 0: #如果经过上面一堆裁剪后变成 0（预算不够、encoder 没排上、已经到顶等）
                # The request cannot be scheduled because one of the following
                # reasons:
                # 1. No new tokens to schedule. This may happen when
                #    (1) PP>1 and we have already scheduled all prompt tokens
                #    but they are not finished yet.
                #    (2) Async scheduling and the request has reached to either
                #    its max_total_tokens or max_model_len.
                # 2. The encoder budget is exhausted.
                # 3. The encoder cache is exhausted.
                # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                # we do not strictly follow the FCFS scheduling policy and
                # allow the lower-priority requests to be scheduled.
                req_index += 1
                continue

            # -------------------------
            # 🧱 尝试分配 KV cache
            # -------------------------
            # Schedule newly needed KV blocks for the request. 尝试分配kv 块
            with record_function_or_nullcontext("schedule: allocate_slots"):
                while True:
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_new_tokens,
                        num_lookahead_tokens=self.num_lookahead_tokens,
                    )

                    if new_blocks is not None:
                        # The request can be scheduled.
                        break  # 分配成功-> 跳出循环

                    # 分配失败-> 抢占优先级最低的请求
                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request.
                    if self.policy == SchedulingPolicy.PRIORITY:
                        preempted_req = max(
                            self.running,
                            key=lambda r: (r.priority, r.arrival_time), #优先级最差（数值最大，通常优先级 0 是最高）的请求。
                        )                                               #如果优先级一样，就找最晚来的（arrival_time 最大的），因为欺负新来的比中断一个快跑完的老请求代价小。
                        self.running.remove(preempted_req)
                        # 如果这个请求已经在本轮调度里，要回滚资源#从运行列表踢走
                        if preempted_req in scheduled_running_reqs:
                            scheduled_running_reqs.remove(preempted_req)
                            token_budget += num_scheduled_tokens[       #归还token额度
                                preempted_req.request_id
                            ]
                            req_to_new_blocks.pop(preempted_req.request_id) #删掉块分配记录
                            num_scheduled_tokens.pop(preempted_req.request_id)
                            scheduled_spec_decode_tokens.pop(
                                preempted_req.request_id, None
                            )
                            preempted_encoder_inputs = scheduled_encoder_inputs.pop(
                                preempted_req.request_id, None
                            )
                            if preempted_encoder_inputs:
                                # Restore encoder compute budget if the preempted
                                # request had encoder inputs scheduled in this step.
                                num_embeds_to_restore = sum(
                                    preempted_req.get_num_encoder_embeds(i)
                                    for i in preempted_encoder_inputs
                                )
                                encoder_compute_budget += num_embeds_to_restore
                            req_index -= 1                              #重点！由于 running 列表变短了，索引要回退 1 位
                    else:   #如果没开PRIORITY，则FIFO策略，直接踢最后一个
                        preempted_req = self.running.pop()

                    self._preempt_request(preempted_req, scheduled_timestamp)
                    preempted_reqs.append(preempted_req)
                    if preempted_req == request: #连自己都被抢占了 说明真没位置了
                        # No more request to preempt. Cannot schedule this request.
                        break

            if new_blocks is None:  #实在分配不到，这轮调度到此为止吧
                # Cannot schedule this request.
                break

            # Schedule the request.
            # -------------------------
            # ✅ 成功调度这个请求
            # -------------------------
            scheduled_running_reqs.append(request)                         #这个request本轮会执行
            req_to_new_blocks[request.request_id] = new_blocks             #记录分配的块
            num_scheduled_tokens[request.request_id] = num_new_tokens      #记录本轮要计算多少token
            token_budget -= num_new_tokens
            req_index += 1

            # -------------------------
            # 🤖 speculative decoding 处理
            # -------------------------
            # Speculative decode related.
            if request.spec_token_ids:
                num_scheduled_spec_tokens = (                               #计算本轮接受多少spec.token
                    num_new_tokens
                    + request.num_computed_tokens
                    - request.num_tokens
                    - request.num_output_placeholders
                )
                if num_scheduled_spec_tokens > 0:
                    # Trim spec_token_ids list to num_scheduled_spec_tokens.
                    del request.spec_token_ids[num_scheduled_spec_tokens:]
                    scheduled_spec_decode_tokens[request.request_id] = (
                        request.spec_token_ids
                    )
                # New spec tokens will be set in `update_draft_token_ids` before the
                # next step when applicable.
                request.spec_token_ids = []

            # -------------------------
            # 🧠 encoder cache 分配
            # -------------------------
            # Encoder-related.
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request.request_id] = (            #记录本轮要计算的encoder输入
                    encoder_inputs_to_schedule
                )
                # Allocate the encoder cache.
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)         #给这些 encoder 输入 分配缓存。因为encoder输出通常被decoder多次使用
                encoder_compute_budget = new_encoder_compute_budget
            if external_load_encoder_input:                                 #处理外部加载的encoder输入
                for i in external_load_encoder_input:
                    self.encoder_cache_manager.allocate(request, i)
                    if self.ec_connector is not None:
                        self.ec_connector.update_state_after_alloc(request, i)

        # Record the LoRAs in scheduled_running_reqs
        scheduled_loras: set[int] = set()
        if self.lora_config:
            scheduled_loras = set(
                req.lora_request.lora_int_id
                for req in scheduled_running_reqs
                if req.lora_request and req.lora_request.lora_int_id > 0
            )
            assert len(scheduled_loras) <= self.lora_config.max_loras

        # Use a temporary RequestQueue to collect requests that need to be    #暂时挂起机制，在排队大厅找一个临时休息区，把那些现在还不满足上场条件的‘VIP 客户’先请过去坐一会儿，等这一轮安排完了，再把他们插回到队列的最前面。”
        # skipped and put back at the head of the waiting queue later
        skipped_waiting_requests = create_request_queue(self.policy)

        # =========================
        # 🟡 第二阶段：处理 waiting 队列
        # =========================
        # Next, schedule the WAITING requests.
        if not preempted_reqs:                                                #如果本轮没发生抢占，一旦发生抢占，说明这一轮很紧张了（连正在跑的都要抢） 就别引入新请求了
            while self.waiting and token_budget > 0:
                if len(self.running) == self.max_num_running_reqs:            #如果当前已经在跑的请求数量 = 系统允许的最大并发数 → 就别再从 waiting 里拉新请求进来了
                    break

                request = self.waiting.peek_request()

                # KVTransfer: skip request if still waiting for remote kvs.  如果这个request还在等 远程KV CACHE
                if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:     #这个request的kv cache在远程节点，比如GPU1生成了一半，GPU2接着算，KV CACHE必须从GPU1传到GPU2
                    is_ready = self._update_waiting_for_remote_kv(request)     #远程kv cache到了没有？
                    if is_ready:
                        request.status = RequestStatus.WAITING
                    else:
                        logger.debug(
                            "%s is still in WAITING_FOR_REMOTE_KVS state.",
                            request.request_id,
                        )
                        self.waiting.pop_request()                             #如果还没到 放进skip队列
                        skipped_waiting_requests.prepend_request(request)
                        continue
                #🟡 处理“结构化输出还没准备好”的请求
                # Skip request if the structured output request is still waiting
                # for FSM compilation.
                if request.status == RequestStatus.WAITING_FOR_FSM:              #如果这个request还在等待FSM（有限状态机）编译完成 背景就是结构化输出不能直接用，必须先编译成FSM（有限状态机）,用于限制模型生成的token，FSM准备号之前，这个request不能开始推理。
                    structured_output_req = request.structured_output_request
                    if structured_output_req and structured_output_req.grammar:
                        request.status = RequestStatus.WAITING
                    else:
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                # Check that adding the request still respects the max_loras
                # constraint.
                if (
                    self.lora_config
                    and request.lora_request
                    and (
                        len(scheduled_loras) == self.lora_config.max_loras
                        and request.lora_request.lora_int_id not in scheduled_loras
                    )
                ):
                    # Scheduling would exceed max_loras, skip.
                    self.waiting.pop_request()
                    skipped_waiting_requests.prepend_request(request)
                    continue

                num_external_computed_tokens = 0                #远程kv cache对应的token数
                load_kv_async = False                           #是否需要异步加载远程Kv

                # Get already-cached tokens. 获取该请求已计算过（cached）的token数量和对应block。该部分主要用于prefix caching(前缀缓存优化)，避免重复计算相同的prompt前缀
                if request.num_computed_tokens == 0:            #这
                    # Get locally-cached tokens.
                    new_computed_blocks, num_new_local_computed_tokens = ( #
                        self.kv_cache_manager.get_computed_blocks(request)
                    )

                    # Get externally-cached tokens if using a KVConnector. 在处理请求的kv cache时，除了检查本地GPU内存里的缓存（prefix caching），还要额外查询外部缓存（比如远程服务器、共享存储、LMcache、 mooncake、vast系统）
                    if self.connector is not None:                         #kv connector（外部缓存连接器），常用于PD分离架构、kv cache跨界点共享，kv offloading（把 KV 缓存卸载到 CPU/远程存储）， LMCache、NIXL 等第三方 KV 缓存系统
                        ext_tokens, load_kv_async = (
                            self.connector.get_num_new_matched_tokens( #这个函数会问远程系统：这个 request 的 prompt有多少 token 已经算过？
                                request, num_new_local_computed_tokens
                            )
                        )

                        if ext_tokens is None:                         #远程系统暂时无法确定 KV 是否匹配,可能远程cache查询还没完成，或者prefix hash还没算完 则不调度这个request
                            # The request cannot be scheduled because
                            # the KVConnector couldn't determine
                            # the number of matched tokens.
                            self.waiting.pop_request()                          #把这个请求从waiting队列中拿出来
                            skipped_waiting_requests.prepend_request(request)   #塞到被跳过队列的最前面，下次优先调度
                            continue                                            #这次没查到 下次就能查到了？LMCache 文档里明确提到：get_num_new_matched_tokens可以故意返回 None，目的是让 vLLM 先去处理其他请求，同时在后台重叠进行这个请求的 I/O（加载 KV、存储 KV、计算哈希等）。

                        request.num_external_computed_tokens = ext_tokens #远程 KV 查询成功：
                        num_external_computed_tokens = ext_tokens

                    # Total computed tokens (local + external).总共已经算过多少 token
                    num_computed_tokens = (
                        num_new_local_computed_tokens + num_external_computed_tokens   #kv cache还能拼接？
                    )
                else: #另一个分支，继续执行/回复执行
                    # KVTransfer: WAITING reqs have num_computed_tokens > 0
                    # after async KV recvs are completed.
                    new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens

                #接下来初始化 encoder 调度变量：
                encoder_inputs_to_schedule = None
                external_load_encoder_input = []                        #从外部加载 encoder cache
                new_encoder_compute_budget = encoder_compute_budget

                if load_kv_async:
                    # KVTransfer: loading remote KV, do not allocate for new work.如果需要异步加载远程 KV（load_kv_async 为 True），说明外部 KV 正在传输中，此时我们**不能**为这个请求分配新的计算工作（不能进行 prefill）
                    assert num_external_computed_tokens > 0 ## 原因：KV 还没完全就位，不能开始计算新的 token，处理方式：把 num_new_tokens 设为 0，后面就不会给它分配 token budget
                    num_new_tokens = 0
                else:
                    # Number of tokens to be scheduled.        #计算本次调度还需要为该请求处理多少个新token
                    # We use `request.num_tokens` instead of   #使用 request.num_tokens（总 token 数）而不是 request.num_prompt_to,是为了兼容被抢占后恢复（resumed）的请求 —— 这类请求可能已经生成了部分 output tokenskens
                    # `request.num_prompt_tokens` to consider the resumed
                    # requests, which have output tokens.
                    num_new_tokens = request.num_tokens - num_computed_tokens   #第一次的时候这不就prompt长度-0吗
                    threshold = self.scheduler_config.long_prefill_token_threshold #如果配置了 long_prefill_token_threshold（长 prefill 阈值），并且当前需要计算的 token 超过该阈值，则进行截断，防止一次 prefill 太长导致延迟抖动
                    if 0 < threshold < num_new_tokens:
                        num_new_tokens = threshold

                    # chunked prefill has to be enabled explicitly to allow  如果没有显式开启chunked prefill，且本次需要处理的token数超过了当前剩余token budget,则直接break，停止继续调度后续waiting请求
                    # pooling requests to be chunked
                    if (
                        not self.scheduler_config.enable_chunked_prefill
                        and num_new_tokens > token_budget
                    ):
                        # If chunked_prefill is disabled,
                        # we can stop the scheduling here.
                        break

                    num_new_tokens = min(num_new_tokens, token_budget)
                    assert num_new_tokens > 0

                    # Schedule encoder inputs. 为请求中的Encoder输入进行调度
                    if request.has_encoder_inputs:
                        (
                            encoder_inputs_to_schedule,
                            num_new_tokens,
                            new_encoder_compute_budget,
                            external_load_encoder_input,
                        ) = self._try_schedule_encoder_inputs(
                            request,
                            num_computed_tokens,
                            num_new_tokens,
                            encoder_compute_budget,
                            shift_computed_tokens=1 if self.use_eagle else 0,
                        )
                        if num_new_tokens == 0:
                            # The request cannot be scheduled.
                            break

                # Handles an edge case when P/D Disaggregation       这段代码专门处理一个边缘情况  当同时启用PD分离和投机解码。
                # is used with Spec Decoding where an                decode节点在分配kv cache block时可能会多分配一个block，导致本地（decode节点）的block数量与远程（prefill节点）实际的block数量不一致
                # extra block gets allocated which                   因此需要计算一个effective_lookahead_tokens 来修正block分配数量，避免mismatch问题
                # creates a mismatch between the number
                # of local and remote blocks.
                effective_lookahead_tokens = (                      #lookahead tokens = 预留未来 decode 的 KV cache 空间
                    0 if request.num_computed_tokens == 0 else self.num_lookahead_tokens #如果是全新请求（num_computed_tokens == 0），说明还在 prefill 阶段，此时不需要预留 lookahead tokens（因为还没开始 decode）
                )                                                   # 如果请求已经计算过 token（处于 decode 阶段或被抢占后恢复），则使用配置的 num_lookahead_tokens（通常是 speculative decoding 的 draft token 数量）

                num_encoder_tokens = (
                    self._num_encoder_max_input_tokens
                    if self.is_encoder_decoder and request.has_encoder_inputs
                    else 0
                )
                #尝试为当前request分配kv cache block(物理槽位)，这是scheduler中非常核心的异步，决定这个请求是否能成功加入运行队列
                new_blocks = self.kv_cache_manager.allocate_slots(  #尝试给 request 分配 KV cache block
                    request,
                    num_new_tokens,                                            #本次需要重新计算的token数量（可能是prompt或待生成的output）
                    num_new_computed_tokens=num_new_local_computed_tokens,     #以下为各种辅助信息，用于精确计算需要分配多少block，以及如何复用已有的block
                    new_computed_blocks=new_computed_blocks,                   #本地已经命中的kv blocks
                    num_lookahead_tokens=effective_lookahead_tokens,
                    num_external_computed_tokens=num_external_computed_tokens, # 外部 KV Connector（如远程、LMCache）命中的 token 数
                    delay_cache_blocks=load_kv_async,                          # 如果为 True，表示 KV 正在异步加载中，暂时不立即分配 block（用于 KVTransfer）
                    num_encoder_tokens=num_encoder_tokens,                     # Encoder-Decoder 模型中 Encoder 部分的 token 数
                )
                #如果分配失败，说明当前剩余 KV cache 不足，无法为该请求分配足够的 block
                if new_blocks is None:
                    # The request cannot be scheduled.# 本轮无法调度该请求，直接跳出 waiting 队列的调度循环
                    break

                # KVTransfer: the connector uses this info to determine         如果启用了 KV Connector（例如 P/D 分离、LMCache、Offloading 等外部 KV 系统）
                # if a load is needed. Note that                                则在 block 分配完成后，通知 connector 更新状态。connector 会根据这些信息判断：
                # This information is used to determine if a load is            是否需要从远程/外部异步加载 KV cache；如何映射本地 block 与远程 block；是否需要触发后续的 KV 数据传输
                # needed for this request.
                if self.connector is not None:
                    self.connector.update_state_after_alloc(
                        request,
                        self.kv_cache_manager.get_blocks(request.request_id),
                        num_external_computed_tokens,
                    )

                # Request was already popped from self.waiting
                # unless it was re-added above due to new_blocks being None.
                request = self.waiting.pop_request()                    # 从 waiting 队列中取出当前要处理的请求（正常情况下在这里真正 pop）
                if load_kv_async:                                       #接下来处理是异步KV加载
                    # If loading async, allocate memory and put request # 如果 KV 需要异步从远程加载（load_kv_async 为 True）：1. 虽然 block 已经分配成功，但实际 KV 数据还没传输过来
                    # into the WAITING_FOR_REMOTE_KV state.             # 2. 不能立即把请求放入 running 队列执行计算 3. 因此要把请求暂时放入“等待远程 KV”状态
                    skipped_waiting_requests.prepend_request(request)   # 把该请求重新塞回到 skipped_waiting_requests 的最前面，下次调度时会优先检查它，看异步加载是否已经完成）
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS #更新请求状态为 WAITING_FOR_REMOTE_KVS  示这个请求正在等待远程/外部 KV cache 的数据到达
                    continue                                            ## 直接 continue，跳过本轮剩余的调度逻辑（不把请求加入 running 队列）

                self._update_connector_prefix_cache_stats(request)     #把当前这个 request 的 Prefix Cache（前缀缓存）命中情况，汇报给外部 KV Connector，用于更新和记录各种缓存统计指标。

                self.running.append(request)                            #加入running队列，此时请求已经成功分配了 KV cache block，可以开始执行 prefill 或继续 decode
                if self.log_stats:                                      #如果开启了统计日志，则记录该请求被成功调度的时刻，用于后续TTFT，调度延迟等性能指标
                    request.record_event(
                        EngineCoreEventType.SCHEDULED, scheduled_timestamp
                    )
                if request.status == RequestStatus.WAITING:             #根据请求当前的状态进行分类统计
                    scheduled_new_reqs.append(request)                  #新请求首次被调度
                elif request.status == RequestStatus.PREEMPTED:         #之前被抢占后，现在回复调度。这类请求通常是之前已经在运行，但是因资源不足被踢回waiting的请求
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(f"Invalid request status: {request.status}")

                if self.lora_config and request.lora_request:
                    scheduled_loras.add(request.lora_request.lora_int_id)

                #将该请求当前分配的所有kv blocks记录下来，req_to_new_blocks是一个字典，用于在本次调度结束后，把request_id与其对应的physical blocks映射关系传递给后续的执行器
                req_to_new_blocks[request.request_id] = (
                    self.kv_cache_manager.get_blocks(request.request_id)
                )
                num_scheduled_tokens[request.request_id] = num_new_tokens #记录该请求本次实际调度的token数量，用于后续统计throughout / token使用情况等
                token_budget -= num_new_tokens                            #本次调度后，剩余的token budget
                request.status = RequestStatus.RUNNING                    #将请求状态设置为RUNNING
                request.num_computed_tokens = num_computed_tokens         #更新该请求已经计算完成的token数量，后续如果该请求被抢占或继续decode，都会基于这个值继续计算
                # Count the number of prefix cached tokens.
                if request.num_cached_tokens < 0:                         #记录该请求通过prefix caching（前缀缓存）命中的token数量，值在第一次记录设置（num_cached_tokens，通常设置为1）
                    request.num_cached_tokens = num_computed_tokens       #num_cached_tokens 这个字段记录的是 “这个请求的 prompt 前缀中，通过 Prefix Caching 命中的 token 数量”，它只对 prompt（预填充阶段）有意义，而且只需要记录一次（通常是第一次成功调度该请求的时候）
                # Encoder-related.                                        #后续 decode 阶段不需要再记录：当请求进入 decode（生成 output tokens）阶段后，num_computed_tokens 会继续增加（每生成一个 token 就 +1），但这些新增的 token 不是通过 prefix caching 命中的，而是正常计算出来的。
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request.request_id] = (
                        encoder_inputs_to_schedule
                    )
                    # Allocate the encoder cache.
                    for i in encoder_inputs_to_schedule:
                        self.encoder_cache_manager.allocate(request, i)
                    encoder_compute_budget = new_encoder_compute_budget
                # Allocate for external load encoder cache
                if external_load_encoder_input:
                    for i in external_load_encoder_input:
                        self.encoder_cache_manager.allocate(request, i)
                        if self.ec_connector is not None:
                            self.ec_connector.update_state_after_alloc(request, i)
        # Put back any skipped requests at the head of the waiting queue  之前被暂时跳过的请求（skipped_waiting_requests）重新放回 waiting 队列的最前面
        #前边调度过程中，很多 request 可能会因为以下原因被暂时跳过：外部 KV Connector 查询返回 None（还没准备好）；oad_kv_async = True（正在异步加载远程 KV）；Encoder 输入调度失败；allocate_slots() 分配 block 失败
        if skipped_waiting_requests:
            self.waiting.prepend_requests(skipped_waiting_requests)  #这些被跳过的请求会被放入 skipped_waiting_requests 中，在调度循环结束前，把它们重新加回到 waiting 队列头部，确保下一次调度时能优先尝试这些请求

        # ====================== 调度约束检查（Scheduling Constraints Validation） ======================
        # 检查本轮调度是否满足各种约束条件，用于调试和保证调度逻辑正确性
        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())    #本轮计算多少token,（所有成功调度的请求的 num_new_tokens 之和）
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens #断言：本轮实际调度的token数不能超过配置的最大值

        assert token_budget >= 0                                           #token_budeget剩余必须>=0，正常情况下应该成立
        assert len(self.running) <= self.max_num_running_reqs              #running 队列中的请求数量不能超过允许的最大并发请求数
        # Since some requests in the RUNNING queue may not be scheduled in       #注意：running 队列中可能存在一些请求在本轮调度中**没有被调度**（例如正在等待下一轮、或被 chunked 等）
        # this step, the total number of scheduled requests can be smaller than  # 因此本轮实际被调度的请求数量（新请求 + 恢复请求 + 本轮继续运行的请求）应当是 running 队列的子集
        # len(self.running).
        assert len(scheduled_new_reqs) + len(scheduled_resumed_reqs) + len(  #本轮被调度执行的request一定是running queue的子集
            scheduled_running_reqs
        ) <= len(self.running)

        # Get the longest common prefix among all requests in the running queue.        #寻找 running 队列中所有请求的最长公共前缀（Longest Common Prefix）
        # This can be potentially used for cascade attention.                           #这个值主要用于 **Cascade Attention**（级联注意力）等高级优化技术，通过识别多个请求之间共享的最长公共 prompt 前缀，可以进一步减少重复的注意力计算，# 从而在高并发场景下提升性能。
        num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)      # 注意：当前 vLLM 版本中，这个变量虽然被计算出来了，但**尚未被实际使用**在核心计算路径中，# 属于为未来优化（Cascade Attention / Prefix-aware scheduling）预留的功能。
        with record_function_or_nullcontext("schedule: get_num_common_prefix_blocks"):
            if self.running:
                any_request = self.running[0]
                num_common_prefix_blocks = (
                    self.kv_cache_manager.get_num_common_prefix_blocks(
                        any_request.request_id
                    )
                )

        # Construct the scheduler output.构造本次调度（scheduler）的最终输出结果  这部分会把本轮成功调度的请求整理成后续执行器（Model Runner / Executor）所需的数据格式
        if self.use_v2_model_runner:
            # 如果使用的是 v2 Model Runner（vLLM v1 新架构），需要特殊处理：
            # 把新请求（scheduled_new_reqs）和被抢占后恢复的请求（scheduled_resumed_reqs）合并在一起
            # 统一当作 “新请求” 处理（v2 架构中对 resumed 请求的处理方式有所不同）
            scheduled_new_reqs = scheduled_new_reqs + scheduled_resumed_reqs
            #清空scheduled_resumed_reqs列表，因为已经合并到new_reqs中
            scheduled_resumed_reqs = []
            # 构造 NewRequestData 对象列表（v2 版本需要更多信息）
            new_reqs_data = [
                NewRequestData.from_request(
                    req,
                    req_to_new_blocks[req.request_id].get_block_ids(),
                    req._all_token_ids,
                )
                for req in scheduled_new_reqs
            ]
        else:
            # 使用传统的 v1 Model Runner 时，处理方式更简单
            # 只处理 scheduled_new_reqs（新请求），resumed 请求在其他地方单独处理
            new_reqs_data = [
                NewRequestData.from_request(
                    req, req_to_new_blocks[req.request_id].get_block_ids()
                )
                for req in scheduled_new_reqs
            ]
        #使用性能记录上下文，方便profiler分析这个函数的耗时
        with record_function_or_nullcontext("schedule: make_cached_request_data"):
            # 构造「已经在 running 队列中」的请求数据（cached request）
            # 包括：正在运行的请求（scheduled_running_reqs）和被抢占后恢复的请求（scheduled_resumed_reqs）
            # 这个函数会为它们准备好后续 Model Runner 需要的数据结构
            cached_reqs_data = self._make_cached_request_data(
                scheduled_running_reqs,
                scheduled_resumed_reqs,
                num_scheduled_tokens,
                scheduled_spec_decode_tokens,
                req_to_new_blocks,
            )

        # Record the request ids that were scheduled in this step.# 记录本轮被成功调度的所有 request_id，用于下一轮调度时参考（例如判断请求是否连续调度等）
        self.prev_step_scheduled_req_ids.clear()
        self.prev_step_scheduled_req_ids.update(num_scheduled_tokens.keys())

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,                           #本轮新调度的请求数据（v2架构中包含resumed）
            scheduled_cached_reqs=cached_reqs_data,                     #本轮继续运行的已缓存请求数据
            num_scheduled_tokens=num_scheduled_tokens,                  #每个请求本次调度的token数量
            total_num_scheduled_tokens=total_num_scheduled_tokens,      #本轮总调度的token数
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,  #投机解码相关信息
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,          # running 队列中最长公共前缀 block 数量（用于未来 Cascade Attention
            preempted_req_ids={req.request_id for req in preempted_reqs},# 本轮被抢占的请求 ID 列表
            # finished_req_ids is an existing state in the scheduler,   ## finished_req_ids 是调度器中已存在的状态，并非本轮新调度产生的 它包含在上一步和当前步之间已经完成（finished）的请求 ID
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
        )

        # NOTE(Kuntai): this function is designed for multiple purposes:  #如果启用了 KV Connector（P/D 分离、LMCache、Offloading 等外部 KV 系统），则构建连接器元数据
        # 1. Plan the KV cache store                                        该函数有多个作用：规划 KV cache 的存储和传输；把所有 KV load / save 操作打包成一个不透明对象；清空 connector 的内部临时状态
        # 2. Wrap up all the KV cache load / save ops into an opaque object
        # 3. Clear the internal states of the connector
        if self.connector is not None:
            meta: KVConnectorMetadata = self.connector.build_connector_meta(
                scheduler_output
            )
            scheduler_output.kv_connector_metadata = meta

        # Build the connector meta for ECConnector
        if self.ec_connector is not None:
            ec_meta: ECConnectorMetadata = self.ec_connector.build_connector_meta(
                scheduler_output
            )
            scheduler_output.ec_connector_metadata = ec_meta

        #更新内部状态 包括：清理临时数据、更新各种统计信息、处理 finished 请求、更新 prefix cache 统计等
        with record_function_or_nullcontext("schedule: update_after_schedule"):
            self._update_after_schedule(scheduler_output)
        return scheduler_output# 返回本次调度的最终结果，供 Engine Core / Executor 使用

    def _preempt_request(
        self,
        request: Request,
        timestamp: float,
    ) -> None:
        """Preempt a request and put it back to the waiting queue.  抢占正在运行的请求，然后把它重新放回等待队列

        NOTE: The request should be popped from the running queue outside of this
        method.
        """
        assert request.status == RequestStatus.RUNNING, (
            "Only running requests can be preempted"
        )
        self.kv_cache_manager.free(request)
        self.encoder_cache_manager.free(request)
        request.status = RequestStatus.PREEMPTED
        request.num_computed_tokens = 0
        request.num_preemptions += 1
        if self.log_stats:
            request.record_event(EngineCoreEventType.PREEMPTED, timestamp)

        # Put the request back to the waiting queue.
        self.waiting.prepend_request(request)

    def _update_after_schedule(
        self,
        scheduler_output: SchedulerOutput,
    ) -> None:
        # Advance the number of computed tokens for the request AFTER
        # the request is scheduled.
        # 1. The scheduler_output of the current step has to include the
        #    original number of scheduled tokens to determine input IDs.
        # 2. Advance the number of computed tokens here allowing us to
        #    schedule the prefill request again immediately in the next
        #    scheduling step.
        # 3. If some tokens (e.g. spec tokens) are rejected later, the number of
        #    computed tokens will be adjusted in update_from_output.
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            request = self.requests[req_id]
            request.num_computed_tokens += num_scheduled_token

            # NOTE: _free_encoder_inputs relies on num_computed_tokens, which
            # may be updated again in _update_from_output for speculative
            # decoding. However, it is safe to call the method here because
            # encoder inputs are always part of the prompt, not the output,
            # and thus are unaffected by speculative decoding.
            if request.has_encoder_inputs:
                self._free_encoder_inputs(request)

        # Clear the finished request IDs.
        # NOTE: We shouldn't do self.finished_req_ids.clear() here because
        # it will also affect the scheduler output.
        self.finished_req_ids = set()

    def _make_cached_request_data(
        self,
        running_reqs: list[Request],
        resumed_reqs: list[Request],
        num_scheduled_tokens: dict[str, int],
        spec_decode_tokens: dict[str, list[int]],
        req_to_new_blocks: dict[str, KVCacheBlocks],
    ) -> CachedRequestData:
        req_ids: list[str] = []
        new_token_ids: list[list[int]] = []
        new_block_ids: list[tuple[list[int], ...] | None] = []
        all_token_ids: dict[str, list[int]] = {}
        num_computed_tokens: list[int] = []
        num_output_tokens: list[int] = []
        resumed_req_ids = set()

        num_running_reqs = len(running_reqs)
        for idx, req in enumerate(itertools.chain(running_reqs, resumed_reqs)):
            req_id = req.request_id
            req_ids.append(req_id)
            if self.use_pp:
                # When using PP, the scheduler sends the sampled tokens back,
                # because there's no direct communication between the first-
                # stage worker and the last-stage worker. Otherwise, we don't
                # need to send the sampled tokens back because the model runner
                # will cache them.
                num_tokens = num_scheduled_tokens[req_id] - len(
                    spec_decode_tokens.get(req_id, ())
                )
                token_ids = req.all_token_ids[
                    req.num_computed_tokens : req.num_computed_tokens + num_tokens
                ]
                new_token_ids.append(token_ids)
            scheduled_in_prev_step = req_id in self.prev_step_scheduled_req_ids
            if idx >= num_running_reqs:
                assert not scheduled_in_prev_step
                resumed_req_ids.add(req_id)
            if not scheduled_in_prev_step:
                all_token_ids[req_id] = req.all_token_ids.copy()
            new_block_ids.append(
                req_to_new_blocks[req_id].get_block_ids(allow_none=True)
            )
            num_computed_tokens.append(req.num_computed_tokens)
            num_output_tokens.append(
                req.num_output_tokens + req.num_output_placeholders
            )

        return CachedRequestData(
            req_ids=req_ids,
            resumed_req_ids=resumed_req_ids,
            new_token_ids=new_token_ids,
            all_token_ids=all_token_ids,
            new_block_ids=new_block_ids,
            num_computed_tokens=num_computed_tokens,
            num_output_tokens=num_output_tokens,
        )

    def _try_schedule_encoder_inputs(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
        encoder_compute_budget: int,
        shift_computed_tokens: int = 0,
    ) -> tuple[list[int], int, int, list[int]]:
        """
        Determine which encoder inputs need to be scheduled in the current step,
        and update `num_new_tokens` and encoder token budget accordingly.

        An encoder input will be scheduled if:
        - Its output tokens overlap with the range of tokens being computed
        in this step, i.e.,
        [num_computed_tokens, num_computed_tokens + num_new_tokens).
        - It is not already computed and stored in the encoder cache.
        - It is not exist on remote encoder cache (via ECConnector)
        - There is sufficient encoder token budget to process it.
        - The encoder cache has space to store it.

        If an encoder input cannot be scheduled due to cache or budget
        limitations, the method adjusts `num_new_tokens` to schedule only the
        decoder tokens up to just before the unschedulable encoder input.

        Note that num_computed_tokens includes both locally cached
        blocks and externally cached blocks (via KVConnector).
        """
        if num_new_tokens == 0 or not request.has_encoder_inputs:
            return [], num_new_tokens, encoder_compute_budget, []
        encoder_inputs_to_schedule: list[int] = []
        mm_features = request.mm_features
        assert mm_features is not None
        assert len(mm_features) > 0
        external_load_encoder_input = []

        # Check remote cache first
        if self.ec_connector is not None:
            remote_cache_has_item = self.ec_connector.has_caches(request)
        # NOTE: since scheduler operates on the request level (possibly with
        # multiple encoder inputs per request), we need to create temporary
        # trackers for accounting at the encoder input level.
        mm_hashes_to_schedule = set()
        num_embeds_to_schedule = 0
        for i, mm_feature in enumerate(mm_features):
            start_pos = mm_feature.mm_position.offset
            num_encoder_tokens = mm_feature.mm_position.length
            num_encoder_embeds = mm_feature.mm_position.get_num_embeds

            # The encoder output is needed if the two ranges overlap:
            # [num_computed_tokens, num_computed_tokens + num_new_tokens) and
            # [start_pos, start_pos + num_encoder_tokens)
            if (
                start_pos
                >= num_computed_tokens + num_new_tokens + shift_computed_tokens
            ):
                # The encoder input is not needed in this step.
                break

            if self.is_encoder_decoder and num_computed_tokens > 0:
                assert start_pos == 0, (
                    "Encoder input should be processed at the beginning of "
                    "the sequence when encoder-decoder models are used."
                )
                # Encoder input has already been computed
                # The calculation here is a bit different. We don't turn encoder
                # output into tokens that get processed by the decoder and
                # reflected in num_computed_tokens. Instead, start_pos reflects
                # the position where we need to ensure we calculate encoder
                # inputs. This should always be 0 to ensure we calculate encoder
                # inputs before running the decoder.  Once we've calculated some
                # decoder tokens (num_computed_tokens > 0), then we know we
                # already calculated encoder inputs and can skip here.
                continue
            elif start_pos + num_encoder_tokens <= num_computed_tokens:
                # The encoder input is already computed and stored
                # in the decoder's KV cache.
                continue

            if not self.is_encoder_decoder:
                # We are not using the encoder cache for encoder-decoder models,
                # yet.
                if request.mm_features[i].identifier in mm_hashes_to_schedule:
                    # The same encoder input has already been scheduled in the
                    # current step.
                    continue

                if self.encoder_cache_manager.check_and_update_cache(request, i):
                    # The encoder input is already computed and cached from a
                    # previous step.
                    continue

            # If no encoder input chunking is allowed, we do not want to
            # partially schedule a multimodal item. If the scheduled range would
            # only cover part of the mm input, roll back to before the mm item.
            if (
                self.scheduler_config.disable_chunked_mm_input
                and num_computed_tokens < start_pos
                and (num_computed_tokens + num_new_tokens)
                < (start_pos + num_encoder_tokens)
            ):
                num_new_tokens = start_pos - num_computed_tokens
                break
            if not self.encoder_cache_manager.can_allocate(
                request, i, encoder_compute_budget, num_embeds_to_schedule
            ):
                # The encoder cache is full or the encoder budget is exhausted.
                # NOTE(woosuk): We assume that the encoder input tokens should
                # be processed altogether, as the encoder usually uses
                # bidirectional attention.
                if num_computed_tokens + shift_computed_tokens < start_pos:
                    # We only schedule the decoder tokens just before the
                    # encoder input.
                    num_new_tokens = start_pos - (
                        num_computed_tokens + shift_computed_tokens
                    )
                else:
                    # Because of prefix caching, num_computed_tokens is greater
                    # than start_pos even though its encoder input is not
                    # available. In this case, we can't schedule any token for
                    # the request in this step.
                    num_new_tokens = 0
                break

            # Calculate the number of embeddings to schedule in the current range
            # of scheduled encoder placholder tokens.
            start_idx_rel = max(0, num_computed_tokens - start_pos)
            end_idx_rel = min(
                num_encoder_tokens, num_computed_tokens + num_new_tokens - start_pos
            )
            curr_embeds_start, curr_embeds_end = (
                mm_feature.mm_position.get_embeds_indices_in_range(
                    start_idx_rel,
                    end_idx_rel,
                )
            )
            # There's no embeddings in the current range of encoder placeholder tokens
            # so we can skip the encoder input.
            if curr_embeds_end - curr_embeds_start == 0:
                continue

            if self.ec_connector is not None and remote_cache_has_item[i]:
                mm_hashes_to_schedule.add(request.mm_features[i].identifier)
                external_load_encoder_input.append(i)
                num_embeds_to_schedule += num_encoder_embeds
                continue

            num_embeds_to_schedule += num_encoder_embeds
            encoder_compute_budget -= num_encoder_embeds
            mm_hashes_to_schedule.add(request.mm_features[i].identifier)
            encoder_inputs_to_schedule.append(i)

        return (
            encoder_inputs_to_schedule,
            num_new_tokens,
            encoder_compute_budget,
            external_load_encoder_input,
        )

    def get_grammar_bitmask(
        self,
        scheduler_output: SchedulerOutput,
    ) -> GrammarOutput | None:
        # Collect list of scheduled request ids that use structured output.
        # The corresponding rows of the bitmask will be in this order.
        # PERF: in case of chunked prefill,
        # request might not include any new tokens.
        # Therefore, we might introduce some additional
        # cycle to fill in the bitmask, which could be a big no-op.
        structured_output_request_ids = [
            req_id
            for req_id in scheduler_output.num_scheduled_tokens
            if (req := self.requests.get(req_id)) and req.use_structured_output
        ]
        if not structured_output_request_ids:
            return None

        bitmask = self.structured_output_manager.grammar_bitmask(
            self.requests,
            structured_output_request_ids,
            scheduler_output.scheduled_spec_decode_tokens,
        )
        return GrammarOutput(structured_output_request_ids, bitmask)

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        sampled_token_ids = model_runner_output.sampled_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        pooler_outputs = model_runner_output.pooler_output
        num_nans_in_logits = model_runner_output.num_nans_in_logits
        kv_connector_output = model_runner_output.kv_connector_output
        cudagraph_stats = model_runner_output.cudagraph_stats

        perf_stats: PerfStats | None = None
        if self.perf_metrics and self.perf_metrics.is_enabled():
            perf_stats = self.perf_metrics.get_step_perf_stats_per_gpu(scheduler_output)

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        spec_decoding_stats: SpecDecodingStats | None = None
        kv_connector_stats: KVConnectorStats | None = (
            kv_connector_output.kv_connector_stats if kv_connector_output else None
        )
        if kv_connector_stats and self.connector:
            kv_stats = self.connector.get_kv_connector_stats()
            if kv_stats:
                kv_connector_stats = kv_connector_stats.aggregate(kv_stats)

        failed_kv_load_req_ids = None
        if kv_connector_output and kv_connector_output.invalid_block_ids:
            # These blocks contain externally computed tokens that failed to
            # load. Identify affected requests and adjust their computed token
            # count to trigger recomputation of the invalid blocks.
            failed_kv_load_req_ids = self._handle_invalid_blocks(
                kv_connector_output.invalid_block_ids
            )

        # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
        # the below loop can be a performance bottleneck. We should do our best
        # to avoid expensive operations inside the loop.
        stopped_running_reqs: set[Request] = set()
        stopped_preempted_reqs: set[Request] = set()
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0
            if failed_kv_load_req_ids and req_id in failed_kv_load_req_ids:
                # skip failed or rescheduled requests from KV load failure
                continue
            request = self.requests.get(req_id)
            if request is None:
                # The request is already finished. This can happen if the
                # request is aborted while the model is executing it (e.g.,
                # in pipeline parallelism).
                continue

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = (
                sampled_token_ids[req_index] if sampled_token_ids else []
            )

            scheduled_spec_token_ids = (
                scheduler_output.scheduled_spec_decode_tokens.get(req_id)
            )
            if scheduled_spec_token_ids:
                num_draft_tokens = len(scheduled_spec_token_ids)
                num_accepted = len(generated_token_ids) - 1
                num_rejected = num_draft_tokens - num_accepted
                # num_computed_tokens represents the number of tokens
                # processed in the current step, considering scheduled
                # tokens and rejections. If some tokens are rejected,
                # num_computed_tokens is decreased by the number of rejected
                # tokens.
                if request.num_computed_tokens > 0:
                    request.num_computed_tokens -= num_rejected
                # If async scheduling, num_output_placeholders also includes
                # the scheduled spec tokens count and so is similarly adjusted.
                if request.num_output_placeholders > 0:
                    request.num_output_placeholders -= num_rejected
                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens=num_draft_tokens,
                    num_accepted_tokens=num_accepted,
                )

            stopped = False
            new_logprobs = None
            new_token_ids = generated_token_ids
            pooler_output = pooler_outputs[req_index] if pooler_outputs else None
            kv_transfer_params = None
            status_before_stop = request.status

            # Check for stop and update request status.
            if new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(
                    request, new_token_ids
                )
            elif request.pooling_params and pooler_output is not None:
                # Pooling stops as soon as there is output.
                request.status = RequestStatus.FINISHED_STOPPED
                stopped = True

            if stopped:
                kv_transfer_params = self._free_request(request)
                if status_before_stop == RequestStatus.RUNNING:
                    stopped_running_reqs.add(request)
                else:
                    stopped_preempted_reqs.add(request)

            # Extract sample logprobs if needed.
            if (
                request.sampling_params is not None
                and request.sampling_params.logprobs is not None
                and logprobs
            ):
                new_logprobs = logprobs.slice_request(req_index, len(new_token_ids))

            if new_token_ids and self.structured_output_manager.should_advance(request):
                struct_output_request = request.structured_output_request
                assert struct_output_request is not None
                assert struct_output_request.grammar is not None
                struct_output_request.grammar.accept_tokens(req_id, new_token_ids)

            if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                request.num_nans_in_logits = num_nans_in_logits[req_id]

            # Get prompt logprobs for this request.
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if new_token_ids or pooler_output is not None or kv_transfer_params:
                # Add EngineCoreOutput for this Request.
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=request.get_finished_reason(),
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        pooling_output=pooler_output,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        kv_transfer_params=kv_transfer_params,
                        trace_headers=request.trace_headers,
                        num_cached_tokens=request.num_cached_tokens,
                        num_nans_in_logits=request.num_nans_in_logits,
                    )
                )
            else:
                # Invariant: EngineCore returns no partial prefill outputs.
                assert not prompt_logprobs_tensors

        # Remove the stopped requests from the running and waiting queues.
        if stopped_running_reqs:
            self.running = remove_all(self.running, stopped_running_reqs)
        if stopped_preempted_reqs:
            # This is a rare case and unlikely to impact performance.
            self.waiting.remove_requests(stopped_preempted_reqs)

        if failed_kv_load_req_ids and not self.recompute_kv_load_failures:
            requests = [self.requests[req_id] for req_id in failed_kv_load_req_ids]
            self.finish_requests(failed_kv_load_req_ids, RequestStatus.FINISHED_ERROR)
            for request in requests:
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=request.request_id,
                        new_token_ids=[],
                        finish_reason=request.get_finished_reason(),
                        events=request.take_events(),
                        trace_headers=request.trace_headers,
                        num_cached_tokens=request.num_cached_tokens,
                    )
                )

        # KV Connector: update state for finished KV Transfers.
        if kv_connector_output:
            self._update_from_kv_xfer_finished(kv_connector_output)

        # collect KV cache events from KV cache manager
        events = self.kv_cache_manager.take_events()

        # collect KV cache events from connector
        if self.connector is not None:
            connector_events = self.connector.take_events()
            if connector_events:
                if events is None:
                    events = list(connector_events)
                else:
                    events.extend(connector_events)

        # publish collected KV cache events
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)

        # Create EngineCoreOutputs for all clients that have requests with
        # outputs in this step.
        engine_core_outputs = {
            client_index: EngineCoreOutputs(outputs=outs)
            for client_index, outs in outputs.items()
        }

        finished_req_ids = self.finished_req_ids_dict
        if finished_req_ids:
            # Include ids of requests that finished since last outputs
            # were sent.
            for client_index, finished_set in finished_req_ids.items():
                # Set finished request set in EngineCoreOutputs for this client.
                if (eco := engine_core_outputs.get(client_index)) is not None:
                    eco.finished_requests = finished_set
                else:
                    engine_core_outputs[client_index] = EngineCoreOutputs(
                        finished_requests=finished_set
                    )
            finished_req_ids.clear()

        if (
            stats := self.make_stats(
                spec_decoding_stats, kv_connector_stats, cudagraph_stats, perf_stats
            )
        ) is not None:
            # Return stats to only one of the front-ends.
            if (eco := next(iter(engine_core_outputs.values()), None)) is None:
                # We must return the stats even if there are no request
                # outputs this step.
                engine_core_outputs[0] = eco = EngineCoreOutputs()
            eco.scheduler_stats = stats

        return engine_core_outputs

    def _update_request_with_output(
        self,
        request: Request,
        new_token_ids: list[int],
    ) -> tuple[list[int], bool]:
        # Append generated tokens and check for stop. Note that if
        # a request is still being prefilled, we expect the model runner
        # to return empty token ids for the request.
        stopped = False
        for num_new, output_token_id in enumerate(new_token_ids, 1):
            request.append_output_token_ids(output_token_id)

            # Check for stop and update request state.
            # This must be called before we make the EngineCoreOutput.
            stopped = check_stop(request, self.max_model_len)
            if stopped:
                del new_token_ids[num_new:]  # Trim new tokens if needed.
                break
        return new_token_ids, stopped

    def _free_encoder_inputs(self, request: Request) -> None:
        cached_encoder_input_ids = self.encoder_cache_manager.get_cached_input_ids(
            request
        )
        # OPTIMIZATION: Avoid list(set) if the set is empty.
        if not cached_encoder_input_ids:
            return

        # Here, we use list(set) to avoid modifying the set while iterating
        # over it.
        for input_id in list(cached_encoder_input_ids):
            mm_feature = request.mm_features[input_id]
            start_pos = mm_feature.mm_position.offset
            num_tokens = mm_feature.mm_position.length
            if self.is_encoder_decoder and request.num_computed_tokens > 0:
                # With Whisper, as soon as we've generated a single token,
                # we know we're done with the encoder input. Cross Attention
                # KVs have been calculated and cached already.
                self.encoder_cache_manager.free_encoder_input(request, input_id)
            elif start_pos + num_tokens <= request.num_computed_tokens:
                # The encoder output is already processed and stored
                # in the decoder's KV cache.
                self.encoder_cache_manager.free_encoder_input(request, input_id)

    def update_draft_token_ids(
        self,
        draft_token_ids: DraftTokenIds,
    ) -> None:
        for req_id, spec_token_ids in zip(
            draft_token_ids.req_ids,
            draft_token_ids.draft_token_ids,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request may have been finished. Skip.
                continue

            # Add newly generated spec token ids to the request.
            if self.structured_output_manager.should_advance(request):
                metadata = request.structured_output_request
                request.spec_token_ids = metadata.grammar.validate_tokens(  # type: ignore[union-attr]
                    spec_token_ids
                )
            else:
                request.spec_token_ids = spec_token_ids

    def get_request_counts(self) -> tuple[int, int]:
        """Returns (num_running_reqs, num_waiting_reqs)."""
        return len(self.running), len(self.waiting)

    def add_request(self, request: Request) -> None:
        self.waiting.add_request(request)
        self.requests[request.request_id] = request
        if self.log_stats:
            request.record_event(EngineCoreEventType.QUEUED)

    def finish_requests(
        self,
        request_ids: str | Iterable[str],
        finished_status: RequestStatus,
    ) -> None:
        """Handles the finish signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.
        """
        assert RequestStatus.is_finished(finished_status) #finished_status value eg. <RequestStatus.FINISHED_ABORTED: 8>
        if isinstance(request_ids, str):
            request_ids = (request_ids,)
        else:
            request_ids = set(request_ids)

        running_requests_to_remove = set()
        waiting_requests_to_remove = []
        valid_requests = []

        # First pass: collect requests to remove from queues
        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # Invalid request ID.
                continue

            valid_requests.append(request)
            if request.status == RequestStatus.RUNNING:
                running_requests_to_remove.add(request)
            else:
                waiting_requests_to_remove.append(request)

        # Remove all requests from queues at once for better efficiency
        if running_requests_to_remove:
            self.running = remove_all(self.running, running_requests_to_remove)
        if waiting_requests_to_remove:
            self.waiting.remove_requests(waiting_requests_to_remove)

        # Second pass: set status and free requests
        for request in valid_requests:
            request.status = finished_status
            self._free_request(request)

    def _free_request(self, request: Request) -> dict[str, Any] | None:
        assert request.is_finished()

        delay_free_blocks, kv_xfer_params = self._connector_finished(request)
        self.encoder_cache_manager.free(request)
        request_id = request.request_id
        self.finished_req_ids.add(request_id)
        if self.finished_req_ids_dict is not None:
            self.finished_req_ids_dict[request.client_index].add(request_id)

        if not delay_free_blocks:
            self._free_blocks(request)

        return kv_xfer_params

    def _free_blocks(self, request: Request):
        assert request.is_finished()
        self.kv_cache_manager.free(request)
        del self.requests[request.request_id]

    def get_num_unfinished_requests(self) -> int:
        return len(self.waiting) + len(self.running)

    def has_finished_requests(self) -> bool:
        return len(self.finished_req_ids) > 0

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        """Reset the KV prefix cache.

        If reset_running_requests is True, all the running requests will be
        preempted and moved to the waiting queue.
        Otherwise, this method will only reset the KV prefix cache when there
        is no running requests taking KV cache.
        """
        if reset_running_requests:
            # For logging.
            timestamp = time.monotonic()
            # Invalidate all the current running requests KV's by pushing them to
            # the waiting queue. In this case, we can reduce the ref count of all
            # the kv blocks to 0 and thus we can make sure the reset is successful.
            # Preempt in reverse order so the requests will be added back to the
            # running queue in FIFO order.
            while self.running:
                request = self.running.pop()
                self._preempt_request(request, timestamp)
                # NOTE(zhuohan): For async scheduling, we need to discard the latest
                # output token on the fly to avoid a redundant repetitive output token.
                request.num_output_placeholders = 0
                request.discard_latest_async_tokens = True

            # Clear scheduled request ids cache. Since we are forcing preemption
            # + resumption in the same step, we must act as if these requests were
            # not scheduled in the prior step. They will be flushed from the
            # persistent batch in the model runner.
            self.prev_step_scheduled_req_ids.clear()

        reset_successful = self.kv_cache_manager.reset_prefix_cache()
        if reset_running_requests and not reset_successful:
            raise RuntimeError(
                "Failed to reset KV cache even when all the running requests are "
                "preempted and moved to the waiting queue. This is likely due to "
                "the presence of running requests waiting for remote KV transfer, "
                "which is not supported yet."
            )

        if reset_connector:
            reset_successful = self.reset_connector_cache() and reset_successful

        return reset_successful

    def reset_connector_cache(self) -> bool:
        if self.connector is None:
            logger.warning("reset_connector called but no KV connector is configured.")
            return False

        if self.connector.reset_cache() is False:
            return False

        if self.log_stats:
            assert self.connector_prefix_cache_stats is not None
            self.connector_prefix_cache_stats.reset = True

        return True

    def make_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None = None,
        kv_connector_stats: KVConnectorStats | None = None,
        cudagraph_stats: CUDAGraphStat | None = None,
        perf_stats: PerfStats | None = None,
    ) -> SchedulerStats | None:
        if not self.log_stats:
            return None
        prefix_cache_stats = self.kv_cache_manager.make_prefix_cache_stats()
        assert prefix_cache_stats is not None
        connector_prefix_cache_stats = self._make_connector_prefix_cache_stats()
        eviction_events = (
            self.kv_metrics_collector.drain_events()
            if self.kv_metrics_collector is not None
            else []
        )
        spec_stats = spec_decoding_stats
        connector_stats_payload = (
            kv_connector_stats.data if kv_connector_stats else None
        )
        return SchedulerStats(
            num_running_reqs=len(self.running),
            num_waiting_reqs=len(self.waiting),
            kv_cache_usage=self.kv_cache_manager.usage,
            prefix_cache_stats=prefix_cache_stats,
            connector_prefix_cache_stats=connector_prefix_cache_stats,
            kv_cache_eviction_events=eviction_events,
            spec_decoding_stats=spec_stats,
            kv_connector_stats=connector_stats_payload,
            cudagraph_stats=cudagraph_stats,
            perf_stats=perf_stats,
        )

    def make_spec_decoding_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None,
        num_draft_tokens: int,
        num_accepted_tokens: int,
    ) -> SpecDecodingStats | None:
        if not self.log_stats:
            return None
        if spec_decoding_stats is None:
            spec_decoding_stats = SpecDecodingStats.new(self.num_spec_tokens)
        spec_decoding_stats.observe_draft(
            num_draft_tokens=num_draft_tokens, num_accepted_tokens=num_accepted_tokens
        )
        return spec_decoding_stats

    def shutdown(self) -> None:
        if self.kv_event_publisher:
            self.kv_event_publisher.shutdown()
        if self.connector is not None:
            self.connector.shutdown()

    ########################################################################
    # KV Connector Related Methods
    ########################################################################

    def _update_connector_prefix_cache_stats(self, request: Request) -> None:
        if self.connector_prefix_cache_stats is None:
            return

        self.connector_prefix_cache_stats.record(
            num_tokens=request.num_tokens,
            num_hits=request.num_external_computed_tokens,
            preempted=request.num_preemptions > 0,
        )

    def _make_connector_prefix_cache_stats(self) -> PrefixCacheStats | None:
        if self.connector_prefix_cache_stats is None:
            return None
        stats = self.connector_prefix_cache_stats
        self.connector_prefix_cache_stats = PrefixCacheStats()
        return stats

    def get_kv_connector(self) -> KVConnectorBase_V1 | None:
        return self.connector

    def _connector_finished(
        self, request: Request
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Invoke the KV connector request_finished() method if applicable.

        Returns optional kv transfer parameters to be included with the
        request outputs.
        """
        if self.connector is None:
            return False, None

        # Free any out-of-window prefix blocks before we hand the block table to
        # the connector.
        self.kv_cache_manager.remove_skipped_blocks(
            request_id=request.request_id,
            total_computed_tokens=request.num_tokens,
        )

        block_ids = self.kv_cache_manager.get_block_ids(request.request_id)

        if not isinstance(self.connector, SupportsHMA):
            # NOTE(Kuntai): We should deprecate this code path after we enforce
            # all connectors to support HMA.
            # Hybrid memory allocator should be already turned off for this
            # code path, but let's double-check here.
            assert len(self.kv_cache_config.kv_cache_groups) == 1
            return self.connector.request_finished(request, block_ids[0])

        return self.connector.request_finished_all_groups(request, block_ids)

    def _update_waiting_for_remote_kv(self, request: Request) -> bool:
        """
        KV Connector: check if the request_id is finished_recving.

        The finished_recving_kv_req_ids list is populated
        on the previous steps()'s update_from_output based
        on the worker side connector.

        When the kv transfer is ready, we cache the blocks
        and the request state will be moved back to WAITING from
        WAITING_FOR_REMOTE_KV.
        """
        assert self.connector is not None
        if request.request_id not in self.finished_recving_kv_req_ids:
            return False

        if request.request_id in self.failed_recving_kv_req_ids:
            # Request had KV load failures; num_computed_tokens was already
            # updated in _update_requests_with_invalid_blocks
            if request.num_computed_tokens:
                # Cache any valid computed tokens.
                self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)
            else:
                # No valid computed tokens, release allocated blocks.
                # There may be a local cache hit on retry.
                self.kv_cache_manager.free(request)

            self.failed_recving_kv_req_ids.remove(request.request_id)
        else:
            # Now that the blocks are ready, actually cache them.
            (block_ids,) = self.kv_cache_manager.get_block_ids(request.request_id)
            num_computed_tokens = len(block_ids) * self.block_size
            # Handle the case where num request tokens less than one block.
            num_computed_tokens = min(num_computed_tokens, request.num_tokens)
            if num_computed_tokens == request.num_tokens:
                num_computed_tokens -= 1
            # This will cache the blocks iff caching is enabled.
            self.kv_cache_manager.cache_blocks(request, num_computed_tokens)

            # Update the request state for scheduling.
            request.num_computed_tokens = num_computed_tokens

        # Return that we are ready.
        self.finished_recving_kv_req_ids.remove(request.request_id)
        return True

    def _update_from_kv_xfer_finished(self, kv_connector_output: KVConnectorOutput):
        """
        KV Connector: update the scheduler state based on the output.

        The Worker side connectors add finished_recving and
        finished_sending reqs to the output.
        * if finished_sending: free the blocks
        # if finished_recving: add to state so we can
            schedule the request during the next step.
        """

        if self.connector is not None:
            self.connector.update_connector_output(kv_connector_output)

        # KV Connector:: update recv and send status from last step.
        for req_id in kv_connector_output.finished_recving or ():
            logger.debug("Finished recving KV transfer for request %s", req_id)
            self.finished_recving_kv_req_ids.add(req_id)
        for req_id in kv_connector_output.finished_sending or ():
            logger.debug("Finished sending KV transfer for request %s", req_id)
            assert req_id in self.requests
            self._free_blocks(self.requests[req_id])

    def _update_requests_with_invalid_blocks(
        self,
        requests: Iterable[Request],
        invalid_block_ids: set[int],
        evict_blocks: bool = True,
    ) -> tuple[set[str], int, set[int]]:
        """
        Identify and update requests affected by invalid KV cache blocks.

        This method scans the given requests, detects those with invalid blocks
        and adjusts their `num_computed_tokens` to the longest valid prefix.
        For observability, it also accumulates the total number of tokens that
        will need to be recomputed across all affected requests.

        Args:
            requests: The set of requests to scan for invalid blocks.
            invalid_block_ids: IDs of invalid blocks.
            evict_blocks: Whether to collect blocks for eviction (False for
                async requests which aren't cached yet).

        Returns:
            tuple:
                - affected_req_ids (set[str]): IDs of requests impacted by
                invalid blocks.
                - total_affected_tokens (int): Total number of tokens that must
                be recomputed across all affected requests.
                - blocks_to_evict (set[int]): Block IDs to evict from cache,
                including invalid blocks and downstream dependent blocks.
        """
        affected_req_ids: set[str] = set()
        total_affected_tokens = 0
        blocks_to_evict: set[int] = set()
        # If a block is invalid and shared by multiple requests in the batch,
        # these requests must be rescheduled, but only the first will recompute
        # it. This set tracks blocks already marked for recomputation.
        marked_invalid_block_ids: set[int] = set()
        for request in requests:
            is_affected = False
            marked_invalid_block = False
            req_id = request.request_id
            # TODO (davidb): add support for hybrid memory allocator
            (req_block_ids,) = self.kv_cache_manager.get_block_ids(req_id)
            # We iterate only over blocks that may contain externally computed
            # tokens
            if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                # Async loading. If num_computed_tokens is set it implies we
                # already processed some block failures for it in a prior step
                req_num_computed_tokens = (
                    request.num_computed_tokens
                    if req_id in self.failed_recving_kv_req_ids
                    else len(req_block_ids) * self.block_size
                )
            else:
                # Sync loading. num_computed_tokens includes new tokens
                req_num_computed_tokens = request.num_cached_tokens

            req_num_computed_blocks = (
                req_num_computed_tokens + self.block_size - 1
            ) // self.block_size
            for idx, block_id in zip(range(req_num_computed_blocks), req_block_ids):
                if block_id not in invalid_block_ids:
                    continue

                is_affected = True

                if block_id in marked_invalid_block_ids:
                    # This invalid block is shared with a previous request
                    # and was already marked for recomputation.
                    # This means this request can still consider this block
                    # as computed when rescheduled.
                    # Currently this only applies to sync loading; Async
                    # loading does not yet support block sharing
                    continue

                marked_invalid_block_ids.add(block_id)

                if marked_invalid_block:
                    # This request has already marked an invalid block for
                    # recomputation and updated its num_computed_tokens.
                    continue

                marked_invalid_block = True
                # Truncate the computed tokens at the first failed block
                request.num_computed_tokens = idx * self.block_size
                num_affected_tokens = (
                    req_num_computed_tokens - request.num_computed_tokens
                )
                total_affected_tokens += num_affected_tokens
                request.num_external_computed_tokens -= num_affected_tokens
                # collect invalid block and all downstream dependent blocks
                if evict_blocks:
                    blocks_to_evict.update(req_block_ids[idx:])

            if is_affected:
                if not marked_invalid_block:
                    # All invalid blocks of this request are shared with
                    # previous requests and will be recomputed by them.
                    # Revert to considering only cached tokens as computed.
                    # Currently this only applies to sync loading; Async
                    # loading does not yet support block sharing
                    total_affected_tokens += (
                        request.num_computed_tokens - request.num_cached_tokens
                    )
                    request.num_computed_tokens = request.num_cached_tokens

                affected_req_ids.add(request.request_id)

        return affected_req_ids, total_affected_tokens, blocks_to_evict

    def _handle_invalid_blocks(self, invalid_block_ids: set[int]) -> set[str]:
        """
        Handle requests affected by invalid KV cache blocks.

        Returns:
            Set of affected request IDs to skip in update_from_output main loop.
        """
        should_fail = not self.recompute_kv_load_failures

        # handle async KV loads (not cached yet, evict_blocks=False)
        async_load_reqs = (
            req
            for req in self.waiting
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        )
        async_failed_req_ids, num_failed_tokens, _ = (
            self._update_requests_with_invalid_blocks(
                async_load_reqs, invalid_block_ids, evict_blocks=False
            )
        )

        total_failed_requests = len(async_failed_req_ids)
        total_failed_tokens = num_failed_tokens

        # handle sync loads (may be cached, collect blocks for eviction)
        sync_failed_req_ids, num_failed_tokens, sync_blocks_to_evict = (
            self._update_requests_with_invalid_blocks(
                self.running, invalid_block_ids, evict_blocks=True
            )
        )

        total_failed_requests += len(sync_failed_req_ids)
        total_failed_tokens += num_failed_tokens

        if not total_failed_requests:
            return set()

        # evict invalid blocks and downstream dependent blocks from cache
        # only when not using recompute policy (where blocks will be recomputed
        # and reused by other requests sharing them)
        if sync_blocks_to_evict and not self.recompute_kv_load_failures:
            self.kv_cache_manager.evict_blocks(sync_blocks_to_evict)

        if should_fail:
            all_failed_req_ids = async_failed_req_ids | sync_failed_req_ids
            logger.error(
                "Failing %d request(s) due to KV load failure "
                "(failure_policy=fail, %d tokens affected). Request IDs: %s",
                total_failed_requests,
                total_failed_tokens,
                all_failed_req_ids,
            )
            return all_failed_req_ids

        logger.warning(
            "Recovered from KV load failure: "
            "%d request(s) rescheduled (%d tokens affected).",
            total_failed_requests,
            total_failed_tokens,
        )

        # Mark async requests with KV load failures for retry once loading completes
        self.failed_recving_kv_req_ids |= async_failed_req_ids
        # Return sync affected IDs to skip in update_from_output
        return sync_failed_req_ids
