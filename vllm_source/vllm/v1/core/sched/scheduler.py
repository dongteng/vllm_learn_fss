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
        kv_cache_config: KVCacheConfig,                                                #KV cache 的层级规格与 block 信息
        structured_output_manager: StructuredOutputManager,
        block_size: int,                                                               #KV cache block 的 token 数
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,                         #多模态注册表,用于处理 multimodal 输入.
        include_finished_set: bool = False,                                            #是否记录每步完成的 request id
        log_stats: bool = False,                                                       #是否收集日志/性能指标
    ) -> None:
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config                           #调度策略配置(batch_size,策略等)
        self.cache_config = vllm_config.cache_config                                   #kv cache缓存配置
        self.lora_config = vllm_config.lora_config                                     #lora相关配置
        self.kv_cache_config = kv_cache_config
        self.kv_events_config = vllm_config.kv_events_config

        self.parallel_config = vllm_config.parallel_config

        self.log_stats = log_stats                                                     #是否记录调度过程中统计信息
        self.observability_config = vllm_config.observability_config

        self.kv_metrics_collector: KVCacheMetricsCollector | None = None               #kv cache监控器(统计命中率、使用量等)
        if self.observability_config.kv_cache_metrics:
            self.kv_metrics_collector = KVCacheMetricsCollector(
                self.observability_config.kv_cache_metrics_sample,
            )
        
        self.structured_output_manager = structured_output_manager                     # 用于支持 JSON schema / grammar decoding
        
        self.is_encoder_decoder = vllm_config.model_config.is_encoder_decoder          # 是否是 encoder-decoder(比如 T5 / 多模态模型)

        # include_finished_set controls whether a separate set of finished             include_finished_set 控制：是否额外维护一个“已完成请求ID集合”
        # request ids should be included in the EngineCoreOutputs returned             这个集合会在 update_from_outputs() 返回的 EngineCoreOutputs 中带出来
        # by update_from_outputs(). This is currently used in the multi-engine         主要用于多引擎(multi-engine)场景下,快速跟踪请求生命周期(谁结束了)
        # case to track request lifetimes efficiently.
        self.finished_req_ids_dict: dict[int, set[str]] | None = (
            defaultdict(set) if include_finished_set else None
        )
        self.prev_step_scheduled_req_ids: set[str] = set()                             # 上一轮调度(step)中,被安排执行的 request id 集合    用途：避免重复调度 / 做差分调度 / 做调度优化

        # Scheduling constraints.                                                      同时最多运行多少个请求(类似并发请求上限)  ,防止现存爆炸或batch太大导致性能下降
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = self.scheduler_config.max_num_batched_tokens   #一次调度最多允许的 token 数 , 控制GPU一次forward的工作量(吞吐vs 延迟的关键参数)

        self.max_model_len = vllm_config.model_config.max_model_len                    #模型最大支持的序列长度(prompt + 生成)

        self.enable_kv_cache_events = (                                                ##是否开启 KV cache 事件通知(用于监控 / profiling / debug)
            self.kv_events_config is not None
            and self.kv_events_config.enable_kv_cache_events
        )
        # Create KVConnector for the Scheduler. Note that each Worker
        # will have a corresponding KVConnector with Role=WORKER.
        # KV Connector pushes/pull of remote KVs for P/D and offloading.
        self.connector = None
        self.connector_prefix_cache_stats: PrefixCacheStats | None = None              # 用于统计 prefix cache 命中率等指标(可观测性)
        self.recompute_kv_load_failures = True                                         ## 当 KV 加载失败时是否重新计算(fallback 策略)  重新算一遍(安全但慢)  False = 直接失败

        if self.vllm_config.kv_transfer_config is not None:
            assert not self.is_encoder_decoder, (                                      ##当前不支持 encoder-decoder 模型(比如 T5)使用 KV connector
                "Encoder-decoder models are not currently supported with KV connectors"
            )
            self.connector = KVConnectorFactory.create_connector(                      #创建 KV Connector(Scheduler 侧)  Worker 侧也会有一个对应的 connector
                config=self.vllm_config,
                role=KVConnectorRole.SCHEDULER,
                kv_cache_config=self.kv_cache_config,
            )
            
            if self.log_stats:                                                         #如果开启日志统计,则初始化 prefix cache 统计器
                self.connector_prefix_cache_stats = PrefixCacheStats()
            kv_load_failure_policy = (
                self.vllm_config.kv_transfer_config.kv_load_failure_policy
            )
            self.recompute_kv_load_failures = kv_load_failure_policy == "recompute"

        self.kv_event_publisher = EventPublisherFactory.create(                        #发布 KV cache 事件.
            self.kv_events_config,
            self.parallel_config.data_parallel_rank,
        )

        self.ec_connector = None                                                        #ec_connector用于 encoder cache 的远程传输.
        if self.vllm_config.ec_transfer_config is not None:
            self.ec_connector = ECConnectorFactory.create_connector(
                config=self.vllm_config, role=ECConnectorRole.SCHEDULER
            )
        num_gpu_blocks = self.cache_config.num_gpu_blocks
        assert num_gpu_blocks is not None and num_gpu_blocks > 0

        self.block_size = block_size

        self.dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        self.pcp_world_size = vllm_config.parallel_config.prefill_context_parallel_size

        self.requests: dict[str, Request] = {}

        try:
            self.policy = SchedulingPolicy(self.scheduler_config.policy)                ## 根据配置创建调度策略(比如 FCFS / priority / fairness)
        except ValueError as e:
            raise ValueError(
                f"Unknown scheduling policy: {self.scheduler_config.policy}"
            ) from e
        self.waiting = create_request_queue(self.policy)                                # waiting 队列(等待调度的请求)
        self.running: list[Request] = []                                                # running 列表(正在 GPU 上执行的请求) 这些请求已经占用了 KV cache + GPU 资源

        # The request IDs that are finished in between the previous and the
        # current steps. This is used to notify the workers about the finished
        # requests so that they can free the cached states for those requests.
        # This is flushed at the end of each scheduling step.
        self.finished_req_ids: set[str] = set()                                         #本步完成的请求

        # KV Connector: requests in process of async KV loading or recving
        self.finished_recving_kv_req_ids: set[str] = set()

        self.failed_recving_kv_req_ids: set[str] = set()                                #接收KV失败的请求->后续可能需要fallback(重新计算)

        # Encoder-related.
        # Calculate encoder cache size if applicable                                    计算 encoder cache 的大小(如果模型有 encoder)
        # NOTE: For now we use the same budget for both compute and space.              注意：目前对 encoder,我们在计算时同时用同样的预算限制计算和缓存空间
        # This can be changed when we make encoder cache for embedding caching          以后如果为跨请求的 embedding 缓存单独设置 encoder cache,可以调整这个逻辑.
        # across requests.                                                              比如说是T5等模型
        encoder_compute_budget, encoder_cache_size = compute_encoder_budget(
            model_config=vllm_config.model_config,
            scheduler_config=vllm_config.scheduler_config,
            mm_registry=mm_registry,
        )

        # NOTE(woosuk): Here, "encoder" includes the vision encoder (and                 计算预算(最多允许处理多少encoder token)-> 防止encoder输入过大导致算力/显存爆炸
        # projector if needed) for MM models as well as encoder-decoder
        # transformers.
        self.max_num_encoder_input_tokens = encoder_compute_budget

        # NOTE: For the models without encoder (e.g., text-only models),                 #encoder cache管理器, 用来缓存encoder的输出(类似KV cache, 但用于encoder)
        # the encoder cache will not be initialized because cache size is 0
        # for these models.
        self.encoder_cache_manager = (
            EncoderDecoderCacheManager(cache_size=encoder_cache_size)
            if self.is_encoder_decoder
            else EncoderCacheManager(cache_size=encoder_cache_size)
        )
        # For encoder-decoder models, allocate the maximum number of tokens for Cross   #对于encoder-decoder模型,预分配encoder最大token数,为什么？像 Whisper 这种模型：输入会被 pad 到固定最大长度, cross-attention需要提前分配好kv 空间
        # Attn blocks, as for Whisper its input is always padded to the maximum length.
        # TODO (NickLucche): Generalize to models with variable-length encoder inputs.  #未来支持可变长度encoder
        self._num_encoder_max_input_tokens = (
            MULTIMODAL_REGISTRY.get_encdec_max_encoder_len(vllm_config.model_config)
        )

        speculative_config = vllm_config.speculative_config
        self.use_eagle = False                                                          #是否启用eagle
        #speculative token数
        self.num_spec_tokens = self.num_lookahead_tokens = 0
        if speculative_config:
            self.num_spec_tokens = speculative_config.num_speculative_tokens
            if speculative_config.use_eagle():
                self.use_eagle = True
                self.num_lookahead_tokens = self.num_spec_tokens
                
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

        self.use_pp = self.parallel_config.pipeline_parallel_size > 1                    #是否启用pp

        self.use_v2_model_runner = envs.VLLM_USE_V2_MODEL_RUNNER

        self.perf_metrics: ModelMetrics | None = None  #性能指标

        # 如果开启日志 + 开启 MFU 指标
        if self.log_stats and vllm_config.observability_config.enable_mfu_metrics:
            self.perf_metrics = ModelMetrics(vllm_config)

    def schedule(self) -> SchedulerOutput:
        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.     调度器没有prefill阶段和 decode阶段,每个请求只是维护2个关键数值.
        # Each request just has the num_computed_tokens and                     每个请求只维护2个关键数值：num_computed_tokens：当前已经被计算/处理的 token 数量
        # num_tokens_with_spec. num_tokens_with_spec =       m_tokens_with_spec：该请求“理论上应该已经处理到的”总 token 数量
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).  num_tokens_with_spec = len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids)
        # At each step, the scheduler tries to assign tokens to the requests    在每个步骤中,调度器会尽量给每个请求分配token,
        # so that each request's num_computed_tokens can catch up its           目标是让每个请求的num_computed_tokens逐步追上它的num_tokens_with_spec
        # num_tokens_with_spec. This is general enough to cover                 这种统一的设计足够够用,能够自然覆盖以下场景：
        # chunked prefills, prefix caching, speculative decoding,               分块与填充;前缀缓存;推测解码;未来可能提出的Jump decoding优化
        # and the "jump decoding" optimization in the future.
                                                                                
        scheduled_new_reqs: list[Request] = []                                  # 本轮从waiting 队列中拉入运行态的请求(第一次被调度执行)
        scheduled_resumed_reqs: list[Request] = []                              # 之前被preempt(抢占过),这一轮重新恢复执行的请求
        scheduled_running_reqs: list[Request] = []                              # 本轮调度器提出running的请求
        preempted_reqs: list[Request] = []                                      # 

        req_to_new_blocks: dict[str, KVCacheBlocks] = {}                        #本轮为每个request新分配的kv cache block的具体信息
        num_scheduled_tokens: dict[str, int] = {}                               #每个 request在这一轮forward中要计算的token数
        token_budget = self.max_num_scheduled_tokens                            #本轮调度的“全局 token 预算”(所有 request 加起来的上限)
        # Encoder-related.
        scheduled_encoder_inputs: dict[str, list[int]] = {}                    #每个request本轮需要执行的encoder输入
        encoder_compute_budget = self.max_num_encoder_input_tokens             #encoder 侧的计算预算(类似 token_budget,但只针对 encoder)
        # Spec decode-related.                                                  
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}                #投机解码相关

        # For logging.
        scheduled_timestamp = time.monotonic()

        # First, schedule the RUNNING requests.                                         #第一阶段：优先处理已经在运行的请求(running),优先让已经跑的继续跑,新请求放后边
        req_index = 0                                                                   #核心思想是让已经在跑的请求继续跑(避免频繁切换 提高吞吐);新请求放后边
        while req_index < len(self.running) and token_budget > 0:                       #遍历running列表,同时收到budget的限制
            request = self.running[req_index]                                           #当前处理的request(已经在decode阶段)
            if (
                request.num_output_placeholders > 0                                     #跳过已确认完成的请求
                # This is (num_computed_tokens + 1) - (num_output_placeholders - 1).    这段比较绕,本质实在判断即使当前spec tokens全部被reject,是否已经达到max_tokens
                # Since output placeholders are also included in the computed tokens    推导含义:num_computed_tokens包含placeholder要减去draft token的影响(num_output_placeholders-1)
                # count, we subtract (num_output_placeholders - 1) to remove any draft  最终效果:如果已经理论上完成,就不要再调度它
                # tokens, so that we can be sure no further steps are needed even if
                # they are all rejected.
                and request.num_computed_tokens + 2 - request.num_output_placeholders
                >= request.num_prompt_tokens + request.max_tokens
            ):
                # Async scheduling: Avoid scheduling an extra step when we are sure that 当我们确定前一步已经达到request.max_tokens 时,就不再为这个请求调度额外的一步
                # the previous step has reached request.max_tokens. We don't schedule
                # partial draft tokens since this prevents uniform decode optimizations.
                req_index += 1
                continue
            num_new_tokens = (                                                          #计算本轮要处理多少token
                request.num_tokens_with_spec                                            #目标token(包含spec), 这个状态是谁更新来的?
                + request.num_output_placeholders                                       #draft token(尚未确认)
                - request.num_computed_tokens                                           #已经算过的
            )                                                                           #本质是还欠多少token没算  该公式本质是 目标+草稿-已完成=剩余工作量
            
            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens: 
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold     #如果这次要处理的token太多,就强行截断,只处理一小部分

            num_new_tokens = min(num_new_tokens, token_budget)                          #不能超过当前剩余预算
            
            # Make sure the input position does not exceed the max model len.           #不能超过模型最大长度
            # This is necessary when using spec decoding.                               #如果使用 spec decoding,那么需要确保输入位置不超过最大模型长度
            num_new_tokens = min(num_new_tokens, self.max_model_len - 1 - request.num_computed_tokens)
            
            # Schedule encoder inputs.                                                  #如果是带图的请求,看看能不能先拍encoder计算,如果排了,可能会再把num_new_tokens减小
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
            if num_new_tokens == 0:                                                     #如果经过上面一堆裁剪后变成 0(预算不够、encoder 没排上、已经到顶等)
                # The request cannot be scheduled because one of the following          #这个request无法调度 可能有以下原因：
                # reasons:
                # 1. No new tokens to schedule. This may happen when                        1.没有新的token可以调度
                #    (1) PP>1 and we have already scheduled all prompt tokens                       (1)当PP>1,我们已经把所有prompt token都调度出去了 但它们还没执行完成
                #    but they are not finished yet. 
                #    (2) Async scheduling and the request has reached to either                     (2)在异步调度(async scheduling)中,这个request已经达到了max_total_tokens或max_model_len的限制
                #    its max_total_tokens or max_model_len.
                # 2. The encoder budget is exhausted.                                       2.异步调度中,这个request已经达到了max_total_toens或max_model_len的限制
                # 3. The encoder cache is exhausted.
                # NOTE(woosuk): Here, by doing `continue` instead of `break`,               注意这里使用 continue 而不是 break,意味着我们不会严格遵守FCFS调度策略,而是允许优先级更低的request也被调度
                # we do not strictly follow the FCFS scheduling policy and
                # allow the lower-priority requests to be scheduled.
                req_index += 1
                continue
            
            # Schedule newly needed KV blocks for the request.                           #尝试分配kv 块
            with record_function_or_nullcontext("schedule: allocate_slots"):
                while True:                                                              #为啥要while true呢？因为真实流程是尝试分配->分配失败->踢掉一个request(释放资源)->再尝试分配->还不够？继续踢 ->再尝试
                    new_blocks = self.kv_cache_manager.allocate_slots(                   #尝试为当前request分配kv cache slot
                        request,
                        num_new_tokens,
                        num_lookahead_tokens=self.num_lookahead_tokens,
                    )

                    if new_blocks is not None:
                        # The request can be scheduled.                                  #分配成功-> 跳出循环
                        break  

                    # The request cannot be scheduled.                                   #分配失败:说明kv cache不够,需要腾位置,通过抢占(preempt)其他request来释放资源
                    # Preempt the lowest-priority request.
                    if self.policy == SchedulingPolicy.PRIORITY:
                        preempted_req = max(
                            self.running,
                            key=lambda r: (r.priority, r.arrival_time),                 #优先级最差(数值最大,通常优先级 0 是最高)的请求.
                        )                                                               #如果优先级一样,就找最晚来的(arrival_time 最大的),因为欺负新来的比中断一个快跑完的老请求代价小.
                        self.running.remove(preempted_req)                              #从running队列中移除(不再参与本轮执行)
                        
                        if preempted_req in scheduled_running_reqs:                     #如果这个request已经在本轮调度计划里,需要回滚它占用的资源,还在好奇num_scheduled_tokens里边没东西,实际上这玩意是遍历request的 ,到后边肯定有了
                            scheduled_running_reqs.remove(preempted_req)
                            token_budget += num_scheduled_tokens[                       #归还token额度
                                preempted_req.request_id
                            ]
                            req_to_new_blocks.pop(preempted_req.request_id)             #删掉刚刚给他分配的kv block
                            num_scheduled_tokens.pop(preempted_req.request_id)          #删除刚刚给它分配的kv block
                            scheduled_spec_decode_tokens.pop(                           #删除spec记录
                                preempted_req.request_id, None
                            )
                            preempted_encoder_inputs = scheduled_encoder_inputs.pop(    #encoder也要回滚
                                preempted_req.request_id, None
                            )
                            if preempted_encoder_inputs:
                                # Restore encoder compute budget if the preempted       如果这个 request 本轮已经占用了 encoder 预算,需要把预算还回去
                                # request had encoder inputs scheduled in this step.
                                num_embeds_to_restore = sum(
                                    preempted_req.get_num_encoder_embeds(i)
                                    for i in preempted_encoder_inputs
                                )
                                encoder_compute_budget += num_embeds_to_restore
                            req_index -= 1                                              #重点！由于 running 列表变短了,索引要回退 1 位
                    else:                                                               
                        preempted_req = self.running.pop()                              #如果没开PRIORITY,则FIFO策略,直接踢最后一个

                    self._preempt_request(preempted_req, scheduled_timestamp)           #执行真正的抢占逻辑(释放kv cache等资源)
                    preempted_reqs.append(preempted_req)
                    if preempted_req == request:                                        #连自己都被抢占了 说明真没位置了
                        # No more request to preempt. Cannot schedule this request.
                        break

            if new_blocks is None:                                                      #到这里说明已经尝试过allocate_slots,甚至已经抢占过其他request , 仍然无法为当前request分配足够的kv cache
                # Cannot schedule this request.                                         #就说明当前request本轮无法调度,直接结束这一轮调度循环
                break

            # Schedule the request.                                                     #能走到这里说明 当前request的kv cache已经成功分配,可以正式加入本轮执行计划
            scheduled_running_reqs.append(request)                                      #进入本轮集合容器里,此处只是计划 还没真正进
            req_to_new_blocks[request.request_id] = new_blocks                          #记录本轮分配的块
            num_scheduled_tokens[request.request_id] = num_new_tokens                   #记录本轮要计算多少token
            token_budget -= num_new_tokens
            req_index += 1

            # Speculative decode related.                                               #误区纠正:这里接受多少spec tokens不是在判断哪些是对的,只是在决定这一轮要提交多少个draft token去给大模型算
            if request.spec_token_ids:                                                  #num_computed_tokens:已计算token数  num_tokens：当前真实token数(已经“正式生成”的长度)     num_output_placeholders：输出占位token数(草稿token数量,还没验证)
                num_scheduled_spec_tokens = (                                           #spec_token_ids：草稿列表   num_new_tokens：本轮要推进的token数量(本轮要处理多少步)
                    num_new_tokens
                    + request.num_computed_tokens
                    - request.num_tokens
                    - request.num_output_placeholders
                )
                if num_scheduled_spec_tokens > 0:
                    # Trim spec_token_ids list to num_scheduled_spec_tokens.
                    del request.spec_token_ids[num_scheduled_spec_tokens:]              #
                    scheduled_spec_decode_tokens[request.request_id] = (
                        request.spec_token_ids
                    )
                # New spec tokens will be set in `update_draft_token_ids` before the
                # next step when applicable.
                request.spec_token_ids = []

            # Encoder-related.
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request.request_id] = (                        #记录本轮要计算的encoder输入
                    encoder_inputs_to_schedule
                )
                # Allocate the encoder cache.
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)                     #给这些 encoder 输入 分配缓存.因为encoder输出通常被decoder多次使用
                encoder_compute_budget = new_encoder_compute_budget
            if external_load_encoder_input:                                             #处理外部加载的encoder输入
                for i in external_load_encoder_input:
                    self.encoder_cache_manager.allocate(request, i)
                    if self.ec_connector is not None:
                        self.ec_connector.update_state_after_alloc(request, i)

        # Record the LoRAs in scheduled_running_reqs                                    记录本轮调度中用到的lora
        scheduled_loras: set[int] = set()
        if self.lora_config:
            scheduled_loras = set(
                req.lora_request.lora_int_id
                for req in scheduled_running_reqs
                if req.lora_request and req.lora_request.lora_int_id > 0
            )
            assert len(scheduled_loras) <= self.lora_config.max_loras
                                                                                       #用来临时存放本轮不能被调度的等待请求. 为什么需要这个临时的队列？在调度过程中,有些请求可能
        # Use a temporary RequestQueue to collect requests that need to be              因为以下原因暂时不能执行：显存不够、前缀缓存还没命中、lora切换限制、等待结构化输出grammar编译完成
        # skipped and put back at the head of the waiting queue later           
        skipped_waiting_requests = create_request_queue(self.policy)

        # Next, schedule the WAITING requests.                                          第二阶段：处理 waiting 队列
        if not preempted_reqs:                                                          #如果本轮没发生抢占,一旦发生抢占,说明这一轮很紧张了(连正在跑的都要抢) 就别引入新请求了
            while self.waiting and token_budget > 0:
                if len(self.running) == self.max_num_running_reqs:                      #如果当前已经在跑的请求数量 = 系统允许的最大并发数 → 就别再从 waiting 里拉新请求进来了
                    break

                request = self.waiting.peek_request()                                   #从waiting队列的最前面拿一出一个请求来看看,但先不把它真正拿出来(不移除)

                # KVTransfer: skip request if still waiting for remote kvs.             如果request在等远程kv cache,就暂时跳过它,request.status 是一个“状态机变量”,由 scheduler + KV 传输模块 + 执行阶段共同维护
                if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:              
                    is_ready = self._update_waiting_for_remote_kv(request)              #检查远程kv cache是否已经传输完成(是否可以在本机执行)
                    if is_ready:
                        request.status = RequestStatus.WAITING
                    else:
                        logger.debug(
                            "%s is still in WAITING_FOR_REMOTE_KVS state.",
                            request.request_id,
                        )
                        self.waiting.pop_request()                                      #如果还没到 放进skip队列,并且从waiting队列里弹出
                        skipped_waiting_requests.prepend_request(request)
                        continue
                # Skip request if the structured output request is still waiting        #处理“结构化输出还没准备好”的请求
                # for FSM compilation.
                if request.status == RequestStatus.WAITING_FOR_FSM:                     #如果这个request还在等待FSM(有限状态机)编译完成 背景就是结构化输出不能直接用,必须先编译成FSM(有限状态机),用于限制模型生成的token,FSM准备号之前,这个request不能开始推理.
                    structured_output_req = request.structured_output_request
                    if structured_output_req and structured_output_req.grammar:
                        request.status = RequestStatus.WAITING
                    else:
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                # Check that adding the request still respects the max_loras            lora约束检查
                # constraint.
                if (
                    self.lora_config
                    and request.lora_request                                            #当前是一个lora请求
                    and (
                        len(scheduled_loras) == self.lora_config.max_loras              #当前已经调度的lora数量已达到上限
                        and request.lora_request.lora_int_id not in scheduled_loras     #并且这个request使用的是一个新的lora
                    )
                ):
                    # Scheduling would exceed max_loras, skip.                          #放到跳过队列(不是丢弃) 下一轮仍会尝试调度
                    self.waiting.pop_request()
                    skipped_waiting_requests.prepend_request(request)
                    continue

                num_external_computed_tokens = 0                                        #远程kv cache对应的token数
                load_kv_async = False                                                   #是否需要异步加载远程Kv

                
                # Get already-cached tokens.                                            #只有在 num_computed_tokens == 0 时:才会执行一次 prefix cache 匹配(本地 + 远程，连续拼接)
                if request.num_computed_tokens == 0:                                    #一旦大于0,说明已经从cache回复过,或已经开始执行prefill decode,不再做prefix匹配
                    # Get locally-cached tokens.
                    new_computed_blocks, num_new_local_computed_tokens = (              #本地匹配永远是第一步
                        self.kv_cache_manager.get_computed_blocks(request)
                    )

                    # Get externally-cached tokens if using a KVConnector.              
                    if self.connector is not None:                                      #如果开启了kv connector(外部缓存连接器).ext_tokens：外部缓存命中的token数量 ; load_kv_async是否需要异步加载外部kv
                        ext_tokens, load_kv_async = (
                            self.connector.get_num_new_matched_tokens(                  #向远程系统查询:从本地命中结束的位置开始,还能匹配多少token
                                request, num_new_local_computed_tokens                  #======传入本地已命中数量,让外部缓存继续往后匹配,外部connector如何知道本地已经匹配了什么内容？也没把本地命中块传进去啊,hash是(前缀块+本块)的hash啊
                            )                                                           #哦 远程可以自己算前边的
                        )

                        if ext_tokens is None:                                          #返回None 可能是网络请求未返回,prefix hash没算完,远程kv正在加载
                            # The request cannot be scheduled because
                            # the KVConnector couldn't determine
                            # the number of matched tokens.
                            self.waiting.pop_request()                                  #把这个请求从waiting队列中拿出来
                            skipped_waiting_requests.prepend_request(request)           #塞到被跳过队列的最前面,下次优先调度
                            continue                                                    #这次没查到 下次就能查到了？LMCache 文档里明确提到：get_num_new_matched_tokens可以故意返回 None,目的是让 vLLM 先去处理其他请求,同时在后台重叠进行这个请求的 I/O(加载 KV、存储 KV、计算哈希等).

                        request.num_external_computed_tokens = ext_tokens               #记录远程命中token数
                        num_external_computed_tokens = ext_tokens

                    # Total computed tokens (local + external)                          .合并本地+远程命中
                    num_computed_tokens = (
                        num_new_local_computed_tokens + num_external_computed_tokens   #kv cache还能拼接？
                    )
                else:                                                                   
                    # KVTransfer: WAITING reqs have num_computed_tokens > 0            #已经有kv(非冷启动)
                    # after async KV recvs are completed.                              不再做prefix cache匹配,直接用已有的num_computed_tokens
                    new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks  #显式设置空block为空,为啥置空？置空不是在清空已有kv，而是在表达,这一轮没有通过prefix cache命中新产生的block
                    num_new_local_computed_tokens = 0                                  #设置本轮本地prefix命中为0
                    num_computed_tokens = request.num_computed_tokens                  #直接用request当前已有的kv进度

                encoder_inputs_to_schedule = None                                       #接下来初始化 encoder 调度变量：
                external_load_encoder_input = []                                        #从外部加载 encoder cache
                new_encoder_compute_budget = encoder_compute_budget

                if load_kv_async:
                    # KVTransfer: loading remote KV, do not allocate for new work.      如果需要异步加载远程 KV(load_kv_async 为 True),说明外部 KV 正在传输中,此时我们**不能**为这个请求分配新的计算工作(不能进行 prefill)
                    assert num_external_computed_tokens > 0                             
                    num_new_tokens = 0                                                  #本轮不给这个request分配任何token计算额度 等价于本轮只占位等待kv 不做计算
                else:
                    # Number of tokens to be scheduled.                                 #计算本次调度还需要为该请求处理多少个新token, 
                    # We use `request.num_tokens` instead of                            #使用 request.num_tokens(总 token 数)而不是 request.num_prompt_to,是为了兼容被抢占后恢复(resumed)的请求 —— 这类请求可能已经生成了部分 output tokenskens
                    # `request.num_prompt_tokens` to consider the resumed
                    # requests, which have output tokens.
                    num_new_tokens = request.num_tokens - num_computed_tokens           #num_tokens是已确认存在的token(prompt+已生成输出),num_computed_tokens是已经有kv cache的token,差值为还需要计算的token(prefill或补算)。decode 阶段，这个公式本质是在算：有多少“新生成但还没转成 KV 的 token”需要补算
                    threshold = self.scheduler_config.long_prefill_token_threshold      #如果配置了 long_prefill_token_threshold(长 prefill 阈值),并且当前需要计算的 token 超过该阈值,则进行截断,防止一次 prefill 太长导致延迟抖动
                    if 0 < threshold < num_new_tokens:                                  #decode阶段肯定不会触发吧 
                        num_new_tokens = threshold

                    # chunked prefill has to be enabled explicitly to allow             如果没有显式开启chunked prefill,且本次需要处理的token数超过了当前剩余token budget,
                    # pooling requests to be chunked                    
                    if (
                        not self.scheduler_config.enable_chunked_prefill
                        and num_new_tokens > token_budget
                    ):
                        # If chunked_prefill is disabled,
                        # we can stop the scheduling here.
                        break                                                           #则直接break,停止继续调度后续waiting请求

                    num_new_tokens = min(num_new_tokens, token_budget)
                    assert num_new_tokens > 0

                    # Schedule encoder inputs.                                          为请求中的Encoder输入进行调度
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

                # Handles an edge case when P/D Disaggregation                          
                # is used with Spec Decoding where an                                   
                # extra block gets allocated which                                      
                # creates a mismatch between the number
                # of local and remote blocks.
                effective_lookahead_tokens = (                                          
                    0 if request.num_computed_tokens == 0 else self.num_lookahead_tokens #
                )                                                                       

                num_encoder_tokens = (
                    self._num_encoder_max_input_tokens
                    if self.is_encoder_decoder and request.has_encoder_inputs
                    else 0
                )
                                                                                        #根据已有kv+本轮新增token+未来预留空间,计算需要多少block
                new_blocks = self.kv_cache_manager.allocate_slots(                      #应该还会顺手把满的块缓存掉
                    request,
                    num_new_tokens,                                                     #本轮需要新计算的token数,这些token当前还没有kv,需要分配空间写入kv,prefill:通常是prompt的一段;decode:通常是1(speculative时>1)
                    num_new_computed_tokens=num_new_local_computed_tokens,              #本地命中的token数
                    new_computed_blocks=new_computed_blocks,                            #本地已经命中的kv blocks
                    num_lookahead_tokens=effective_lookahead_tokens,
                    num_external_computed_tokens=num_external_computed_tokens,          # 外部 KV Connector(如远程、LMCache)命中的 token 数
                    delay_cache_blocks=load_kv_async,                                   # 如果为 True,表示 KV 正在异步加载中,暂时不立即分配 block(用于 KVTransfer)
                    num_encoder_tokens=num_encoder_tokens,                              # Encoder-Decoder 模型中 Encoder 部分的 token 数
                )
                                                                                        #如果分配失败,说明当前剩余 KV cache 不足,无法为该请求分配足够的 block
                if new_blocks is None:
                    # The request cannot be scheduled.                                  # 本轮无法调度该请求,直接跳出 waiting 队列的调度循环
                    break

                # KVTransfer: the connector uses this info to determine                 在kv block分配完成后,把本地kv覆盖范围同步给connector,用于推进远程kv 加载与状态机对齐,而不是决定是否加载kv
                # if a load is needed. Note that                                        这里只是告诉位置,而不是执行加载
                # This information is used to determine if a load is                    
                # needed for this request.
                if self.connector is not None:
                    self.connector.update_state_after_alloc(
                        request,
                        self.kv_cache_manager.get_blocks(request.request_id),
                        num_external_computed_tokens,
                    )

                # Request was already popped from self.waiting
                # unless it was re-added above due to new_blocks being None.
                request = self.waiting.pop_request()                                    #从 waiting 队列中取出当前要处理的请求(正常情况下在这里真正 pop)
                if load_kv_async:                                                       #接下来处理是异步KV加载
                    # If loading async, allocate memory and put request                 # 如果 KV 需要异步从远程加载(load_kv_async 为 True)：1. 虽然 block 已经分配成功,但实际 KV 数据还没传输过来
                    # into the WAITING_FOR_REMOTE_KV state.                             # 2. 不能立即把请求放入 running 队列执行计算 3. 因此要把请求暂时放入“等待远程 KV”状态
                    skipped_waiting_requests.prepend_request(request)                   # 把该请求重新塞回到 skipped_waiting_requests 的最前面,下次调度时会优先检查它,看异步加载是否已经完成)
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS               #更新请求状态为 WAITING_FOR_REMOTE_KVS  示这个请求正在等待远程/外部 KV cache 的数据到达
                    continue                                                            # 直接 continue,跳过本轮剩余的调度逻辑(不把请求加入 running 队列)

                self._update_connector_prefix_cache_stats(request)                      #把当前这个 request 的 Prefix Cache(前缀缓存)命中情况,汇报给外部 KV Connector,用于更新和记录各种缓存统计指标.

                self.running.append(request)                                            #加入running队列,此时请求已经成功分配了 KV cache block,可以开始执行 prefill 或继续 decode
                if self.log_stats:                                                      #如果开启了统计日志,则记录该请求被成功调度的时刻,用于后续TTFT,调度延迟等性能指标
                    request.record_event(
                        EngineCoreEventType.SCHEDULED, scheduled_timestamp
                    )
                if request.status == RequestStatus.WAITING:                             #根据请求当前的状态进行分类统计
                    scheduled_new_reqs.append(request)                                  #新请求首次被调度
                elif request.status == RequestStatus.PREEMPTED:                         #之前被抢占后,现在回复调度.这类请求通常是之前已经在运行,但是因资源不足被踢回waiting的请求
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(f"Invalid request status: {request.status}")

                if self.lora_config and request.lora_request:
                    scheduled_loras.add(request.lora_request.lora_int_id)

                #将该请求当前分配的所有kv blocks记录下来,req_to_new_blocks是一个字典,用于在本次调度结束后,把request_id与其对应的physical blocks映射关系传递给后续的执行器
                req_to_new_blocks[request.request_id] = (
                    self.kv_cache_manager.get_blocks(request.request_id)
                )
                num_scheduled_tokens[request.request_id] = num_new_tokens               #记录该请求本次实际调度的token数量,用于后续统计throughout / token使用情况等
                token_budget -= num_new_tokens                                          #本次调度后,剩余的token budget
                request.status = RequestStatus.RUNNING                                  #将请求状态设置为RUNNING
                request.num_computed_tokens = num_computed_tokens                       #更新该请求已经计算完成的token数量,后续如果该请求被抢占或继续decode,都会基于这个值继续计算
                # Count the number of prefix cached tokens.
                if request.num_cached_tokens < 0:                                       #记录该请求通过prefix caching(前缀缓存)命中的token数量,值在第一次记录设置(num_cached_tokens,通常设置为1)
                    request.num_cached_tokens = num_computed_tokens                     #num_cached_tokens 这个字段记录的是 “这个请求的 prompt 前缀中,通过 Prefix Caching 命中的 token 数量”,它只对 prompt(预填充阶段)有意义,而且只需要记录一次(通常是第一次成功调度该请求的时候)
                # Encoder-related.                                                      #后续 decode 阶段不需要再记录：当请求进入 decode(生成 output tokens)阶段后,num_computed_tokens 会继续增加(每生成一个 token 就 +1),但这些新增的 token 不是通过 prefix caching 命中的,而是正常计算出来的.
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
        # Put back any skipped requests at the head of the waiting queue                之前被暂时跳过的请求(skipped_waiting_requests)重新放回 waiting 队列的最前面
        #前边调度过程中,很多 request 可能会因为以下原因被暂时跳过：外部 KV Connector 查询返回 None(还没准备好);oad_kv_async = True(正在异步加载远程 KV);Encoder 输入调度失败;allocate_slots() 分配 block 失败
        if skipped_waiting_requests:
            self.waiting.prepend_requests(skipped_waiting_requests)                     #这些被跳过的请求会被放入 skipped_waiting_requests 中,在调度循环结束前,把它们重新加回到 waiting 队列头部,确保下一次调度时能优先尝试这些请求

        # ====================== 调度约束检查(Scheduling Constraints Validation) ====================== 检查本轮调度是否满足各种约束条件,用于调试和保证调度逻辑正确性
        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())                 #本轮计算多少token,(所有成功调度的请求的 num_new_tokens 之和)
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens              #断言：本轮实际调度的token数不能超过配置的最大值

        assert token_budget >= 0                                                        #token_budeget剩余必须>=0,正常情况下应该成立
        assert len(self.running) <= self.max_num_running_reqs                           #running 队列中的请求数量不能超过允许的最大并发请求数
        # Since some requests in the RUNNING queue may not be scheduled in               #注意：running 队列中可能存在一些请求在本轮调度中**没有被调度**(例如正在等待下一轮、或被 chunked 等)
        # this step, the total number of scheduled requests can be smaller than         # 因此本轮实际被调度的请求数量(新请求 + 恢复请求 + 本轮继续运行的请求)应当是 running 队列的子集
        # len(self.running).
        assert len(scheduled_new_reqs) + len(scheduled_resumed_reqs) + len(             #本轮被调度执行的request一定是running queue的子集
            scheduled_running_reqs
        ) <= len(self.running)

        # Get the longest common prefix among all requests in the running queue.        #寻找 running 队列中所有请求的最长公共前缀(Longest Common Prefix)
        # This can be potentially used for cascade attention.                           #这个值主要用于 **Cascade Attention**(级联注意力)等高级优化技术,通过识别多个请求之间共享的最长公共 prompt 前缀,可以进一步减少重复的注意力计算,# 从而在高并发场景下提升性能.
        num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)      # 注意：当前 vLLM 版本中,这个变量虽然被计算出来了,但**尚未被实际使用**在核心计算路径中,# 属于为未来优化(Cascade Attention / Prefix-aware scheduling)预留的功能.
        with record_function_or_nullcontext("schedule: get_num_common_prefix_blocks"):
            if self.running:
                any_request = self.running[0]
                num_common_prefix_blocks = (
                    self.kv_cache_manager.get_num_common_prefix_blocks(
                        any_request.request_id
                    )
                )

        # Construct the scheduler output.                                               构造本次调度(scheduler)的最终输出结果  这部分会把本轮成功调度的请求整理成后续执行器(Model Runner / Executor)所需的数据格式
        if self.use_v2_model_runner:
                                                                                        # 如果使用的是 v2 Model Runner(vLLM v1 新架构),需要特殊处理：
                                                                                        # 把新请求(scheduled_new_reqs)和被抢占后恢复的请求(scheduled_resumed_reqs)合并在一起
                                                                                        # 统一当作 “新请求” 处理(v2 架构中对 resumed 请求的处理方式有所不同)
            scheduled_new_reqs = scheduled_new_reqs + scheduled_resumed_reqs
            scheduled_resumed_reqs = []                                                  #清空scheduled_resumed_reqs列表,因为已经合并到new_reqs中
            # 构造 NewRequestData 对象列表(v2 版本需要更多信息)
            new_reqs_data = [
                NewRequestData.from_request(
                    req,
                    req_to_new_blocks[req.request_id].get_block_ids(),
                    req._all_token_ids,
                )
                for req in scheduled_new_reqs
            ]
        else:
                                                                                        # 使用传统的 v1 Model Runner 时,处理方式更简单
                                                                                        # 只处理 scheduled_new_reqs(新请求),resumed 请求在其他地方单独处理
            new_reqs_data = [
                NewRequestData.from_request(
                    req, req_to_new_blocks[req.request_id].get_block_ids()
                )
                for req in scheduled_new_reqs
            ]
                                                                                        #使用性能记录上下文,方便profiler分析这个函数的耗时
        with record_function_or_nullcontext("schedule: make_cached_request_data"):
                                                                                        # 构造「已经在 running 队列中」的请求数据(cached request)
                                                                                        # 包括：正在运行的请求(scheduled_running_reqs)和被抢占后恢复的请求(scheduled_resumed_reqs)
                                                                                        # 这个函数会为它们准备好后续 Model Runner 需要的数据结构
            cached_reqs_data = self._make_cached_request_data(
                scheduled_running_reqs,
                scheduled_resumed_reqs,
                num_scheduled_tokens,
                scheduled_spec_decode_tokens,
                req_to_new_blocks,
            )

        # Record the request ids that were scheduled in this step.                      # 记录本轮被成功调度的所有 request_id,用于下一轮调度时参考(例如判断请求是否连续调度等)
        self.prev_step_scheduled_req_ids.clear()
        self.prev_step_scheduled_req_ids.update(num_scheduled_tokens.keys())

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,                                           #本轮新调度的请求数据(v2架构中包含resumed)
            scheduled_cached_reqs=cached_reqs_data,                                     #本轮继续运行的已缓存请求数据
            num_scheduled_tokens=num_scheduled_tokens,                                  #每个请求本次调度的token数量
            total_num_scheduled_tokens=total_num_scheduled_tokens,                      #本轮总调度的token数
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,                  #投机解码相关信息
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,                          # running 队列中最长公共前缀 block 数量(用于未来 Cascade Attention
            preempted_req_ids={req.request_id for req in preempted_reqs},               # 本轮被抢占的请求 ID 列表
            # finished_req_ids is an existing state in the scheduler,   #               # finished_req_ids 是调度器中已存在的状态,并非本轮新调度产生的 它包含在上一步和当前步之间已经完成(finished)的请求 ID
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
        )

        # NOTE(Kuntai): this function is designed for multiple purposes:                #如果启用了 KV Connector(P/D 分离、LMCache、Offloading 等外部 KV 系统),则构建连接器元数据
        # 1. Plan the KV cache store                                                    该函数有多个作用：1规划 KV cache 存储：确定哪些 KV 需要存入远程/本地缓存，哪些需要从远程拉取
        # 2. Wrap up all the KV cache load / save ops into an opaque object                              2.封装操作：将所有复杂的 KV load (加载) / save (存储) 指令打包成一个不透明的 Metadata 对象，后续 Worker 只需根据此对象执行，无需关心具体的搬运逻辑
        # 3. Clear the internal states of the connector                                                  3. 清理状态：重置 connector 内部在匹配阶段产生的临时变量，为下一次迭代做准备。
        if self.connector is not None:
            meta: KVConnectorMetadata = self.connector.build_connector_meta(            # 根据调度器输出（包含刚刚分配好的 Block 布局）构建连接器元数据,这里的 build_connector_meta 实际上完成了“货架编号”与“搬运任务”的最终绑定
                scheduler_output
            )
            scheduler_output.kv_connector_metadata = meta

        # Build the connector meta for ECConnector                                      为 ECConnector (通常用于纠错码或弹性计算相关的 KV 传输) 构建元数据
        if self.ec_connector is not None:
            ec_meta: ECConnectorMetadata = self.ec_connector.build_connector_meta(
                scheduler_output
            )
            scheduler_output.ec_connector_metadata = ec_meta
        with record_function_or_nullcontext("schedule: update_after_schedule"):         #更新内部状态 包括：清理临时数据、更新各种统计信息、处理 finished 请求、更新 prefix cache 统计等
            self._update_after_schedule(scheduler_output)
        return scheduler_output                                                         #返回本次调度的最终结果,供 Engine Core / Executor 使用

    def _preempt_request(
        self,
        request: Request,
        timestamp: float,
    ) -> None:
        """Preempt a request and put it back to the waiting queue.                      #抢占 将request放回waiting队列

        NOTE: The request should be popped from the running queue outside of this
        method.
        """
        assert request.status == RequestStatus.RUNNING, (                               #只允许正在运行的request被抢占
            "Only running requests can be preempted"
        )
        self.kv_cache_manager.free(request)
        self.encoder_cache_manager.free(request)
        request.status = RequestStatus.PREEMPTED                                        #状态改为 PREEMPTED(被抢占)
        request.num_computed_tokens = 0                                                 #重置已计算token数,说明这个request之后需要从头重新计算,因为kv cache已经被释放,之前的计算结果没法复用了
        request.num_preemptions += 1                                                    #记录被抢占次数
        if self.log_stats:
            request.record_event(EngineCoreEventType.PREEMPTED, timestamp)              #如果开启日志,则记录一次PREEMPT时间

        # Put the request back to the waiting queue.
        self.waiting.prepend_request(request)

    def _update_after_schedule(         #本次调度结束后,更新请求的内部状态.这是scheduler完成一轮调度后的收尾工作
        self,
        scheduler_output: SchedulerOutput,
    ) -> None:
        # Advance the number of computed tokens for the request AFTER                #为什么要在调度完成后才更新num_computed_tokens?原因有3点：
        # the request is scheduled.                                                  1.当前步骤生成的scheduler_output必须使用调度前的num_computed_tokens以便正确构造input_ids(模型输入)
        # 1. The scheduler_output of the current step has to include the
        #    original number of scheduled tokens to determine input IDs.
        # 2. Advance the number of computed tokens here allowing us to               2.在这里提前num_computed_tokens,可以让同一个prefill请求在下一轮调度室,立即被识别为已经计算过部分token,从而支持连续调度(尤其是chunked prefill)
        #    schedule the prefill request again immediately in the next
        #    scheduling step.
        # 3. If some tokens (e.g. spec tokens) are rejected later, the number of     3.如果后续执行阶段发现某些token数被reject(例如投机解码中draft token被拒绝)会在update_from_output中再对num_computed_tokens进行修正
        #    computed tokens will be adjusted in update_from_output.
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            request = self.requests[req_id]
            request.num_computed_tokens += num_scheduled_token

            # NOTE: _free_encoder_inputs relies on num_computed_tokens, which        如果该请求带有 Encoder 输入(Encoder-Decoder 模型或多模态模型),
            # may be updated again in _update_from_output for speculative            则尝试释放 Encoder 输入占用的 cache.
            # decoding. However, it is safe to call the method here because          Encoder 输入属于 prompt 的一部分,不受 speculative decoding 影响,
            # encoder inputs are always part of the prompt, not the output,          所以在这里提前释放是安全的.
            # and thus are unaffected by speculative decoding.
            if request.has_encoder_inputs:
                self._free_encoder_inputs(request)

        # Clear the finished request IDs.                                            清空已经完成请求ID的集合
        # NOTE: We shouldn't do self.finished_req_ids.clear() here because           注意：不能直接使用 self.finished_req_ids.clear(),因为 scheduler_output 中引用了当前的 finished_req_ids(浅引用),
        # it will also affect the scheduler output.                                  清空会导致 scheduler_output 中的 finished_req_ids 也被清空,破坏已构造好的输出结果
        self.finished_req_ids = set()                                                #因此采用重新赋值空 set 的方式来清空.

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
        """
        在每次Engine step(调度+模型推理)完成后调用.
        作用：把模型刚刚生成的token(采样结果)更新到各个request对象中,检查哪些请求停止了,构造要返回给上层(AsyncLLMEngine/API Server)的输出
        """
        sampled_token_ids = model_runner_output.sampled_token_ids           #本次 step 每个 request 新采样的 token ids(通常 decode 时每个 req 1个)
        logprobs = model_runner_output.logprobs                             # 如果用户请求了 logprobs,这里有对应数据
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict     #prompt 阶段的 logprobs(较少使用)
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens        # dict: {req_id: 本次 step 为该请求调度了多少 tokens}
        pooler_outputs = model_runner_output.pooler_output                  # 用于 embedding / reward model / pooling 模式的输出
        num_nans_in_logits = model_runner_output.num_nans_in_logits         ## 记录 logits 中出现 NaN 的情况(用于 debug/监控)
        kv_connector_output = model_runner_output.kv_connector_output       #KV Cache 跨节点/设备传输相关输出(分布式场景)
        cudagraph_stats = model_runner_output.cudagraph_stats               #CUDA Graph统计信息(性能监控)

        #性能统计,如果开启了perf_metrics,记录本次step的GPU性能数据(时间、内存等)
        perf_stats: PerfStats | None = None
        if self.perf_metrics and self.perf_metrics.is_enabled():
            perf_stats = self.perf_metrics.get_step_perf_stats_per_gpu(scheduler_output)

        #最终返回的结果,key是client_index(多客户端)
        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        spec_decoding_stats: SpecDecodingStats | None = None                #投机解码统计
        
        #kv cache传输统计(分布式、kv offloading 、 disaggregated serving时使用)
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
            #某些kv block从外部加载失败(比如跨节点传输失败)
            #需要把这些请求标记为重新计算
            failed_kv_load_req_ids = self._handle_invalid_blocks(
                kv_connector_output.invalid_block_ids
            )

        #重要性能提醒：num_scheduled_tokens可能很大(上千),循环里尽量不要放昂贵操作
        # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
        # the below loop can be a performance bottleneck. We should do our best
        # to avoid expensive operations inside the loop.
        stopped_running_reqs: set[Request] = set()                             #本次正常运行中停止的请求(用于后续从 running 队列移除)
        stopped_preempted_reqs: set[Request] = set()                           #被抢占后停止的请求(较少见)
        
        #==================== 核心循环：逐个处理本次调度的请求 ==================
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0
            if failed_kv_load_req_ids and req_id in failed_kv_load_req_ids:
                # skip failed or rescheduled requests from KV load failure      # KV 加载失败的请求跳过,本次不处理(后面会标记为 error)
                continue
            request = self.requests.get(req_id)                                 ## 从 scheduler 维护的请求字典中拿到 Request 对象
            if request is None:
                # The request is already finished. This can happen if the
                # request is aborted while the model is executing it (e.g.,
                # in pipeline parallelism).
                #请求可能在模型执行期间被 abort(比如用户中途取消)
                continue

            req_index = model_runner_output.req_id_to_index[req_id]             #该请求在本次batch中的索引
            
            #本次实际生成的新token(通常1个,投机解码多个)
            generated_token_ids = (
                sampled_token_ids[req_index] if sampled_token_ids else []
            )
            # ------------------- Speculative Decoding(投机解码)处理 -------------------
            scheduled_spec_token_ids = (
                scheduler_output.scheduled_spec_decode_tokens.get(req_id)
            )
            if scheduled_spec_token_ids:
                num_draft_tokens = len(scheduled_spec_token_ids)                #draft model提前猜测了多少token
                num_accepted = len(generated_token_ids) - 1                     #主模型接收了多少个
                num_rejected = num_draft_tokens - num_accepted
                # num_computed_tokens represents the number of tokens
                # processed in the current step, considering scheduled
                # tokens and rejections. If some tokens are rejected,
                # num_computed_tokens is decreased by the number of rejected
                # tokens.
                #因为有rejection,需要回退 num_computed_tokens(已计算 token 数)
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
            # ------------------- 更新请求状态 -------------------
            stopped = False
            new_logprobs = None
            new_token_ids = generated_token_ids
            pooler_output = pooler_outputs[req_index] if pooler_outputs else None
            kv_transfer_params = None
            status_before_stop = request.status                                 # 记录停止前的状态(用于区分 running / preempted)

            # Check for stop and update request status.检查是否触发停止条件
            if new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(
                    request, new_token_ids
                )
            elif request.pooling_params and pooler_output is not None:          # Pooling 模型(embedding 类)只要有输出就停止
                # Pooling stops as soon as there is output.
                request.status = RequestStatus.FINISHED_STOPPED
                stopped = True

            if stopped:                                                         #释放KV cache等资源
                kv_transfer_params = self._free_request(request)
                if status_before_stop == RequestStatus.RUNNING:
                    stopped_running_reqs.add(request)
                else:
                    stopped_preempted_reqs.add(request)

            # Extract sample logprobs if needed.# ------------------- 提取 logprobs(如果用户请求了) -------------------
            if (
                request.sampling_params is not None
                and request.sampling_params.logprobs is not None
                and logprobs
            ):
                new_logprobs = logprobs.slice_request(req_index, len(new_token_ids))

            # ------------------- Structured Output(JSON / grammar)处理 -------------------
            if new_token_ids and self.structured_output_manager.should_advance(request):
                #把新生成的 token 喂给 grammar FSM(例如 JSON schema 校验)
                struct_output_request = request.structured_output_request
                assert struct_output_request is not None
                assert struct_output_request.grammar is not None
                struct_output_request.grammar.accept_tokens(req_id, new_token_ids)

            # ------------------- Prompt logprobs 处理 -------------------
            if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                request.num_nans_in_logits = num_nans_in_logits[req_id]

            # Get prompt logprobs for this request.# 只要有新 token、pooling 输出、或 KV 传输参数,就构造输出对象
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if new_token_ids or pooler_output is not None or kv_transfer_params:
                # Add EngineCoreOutput for this Request.
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,                                # 本次新增的 token(流式关键)
                        finish_reason=request.get_finished_reason(),                # stop / length / tool_calls 等
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        pooling_output=pooler_output,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),                               # tracing / 事件
                        kv_transfer_params=kv_transfer_params,  
                        trace_headers=request.trace_headers,
                        num_cached_tokens=request.num_cached_tokens,                # prefix caching 命中情况
                        num_nans_in_logits=request.num_nans_in_logits,
                    )
                )
            else:
                # Invariant: EngineCore returns no partial prefill outputs. # 不应该出现 partial prefill 输出(vLLM v1 的 invariant)
                assert not prompt_logprobs_tensors

        # Remove the stopped requests from the running and waiting queues.# ==================== 清理停止的请求 ====================
        if stopped_running_reqs:
            self.running = remove_all(self.running, stopped_running_reqs)            # 从 running 队列移除
        if stopped_preempted_reqs:
            # This is a rare case and unlikely to impact performance.
            self.waiting.remove_requests(stopped_preempted_reqs)                     # 从 waiting 队列移除(少见)

        #处理KV 加载失败的请求(标记为 error)
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

        # KV Connector: update state for finished KV Transfers.# 更新 KV Connector 完成的状态(分布式 KV 传输)
        if kv_connector_output:
            self._update_from_kv_xfer_finished(kv_connector_output)

        # collect KV cache events from KV cache manager # 收集并发布 KV cache 相关事件(用于监控、tracing)
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

        ## ==================== 构造最终返回给上层的 EngineCoreOutputs ====================
        # Create EngineCoreOutputs for all clients that have requests with
        # outputs in this step.
        engine_core_outputs = {
            client_index: EngineCoreOutputs(outputs=outs)
            for client_index, outs in outputs.items()
        }
        # 把最近完成的 request ids 也带上(用于清理)
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

        # 添加性能 / 统计信息(只返回给其中一个 frontend)
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
        """Handles the finish signal from outside the scheduler.  处理来自Scheduler外部的请求结束信号  是统一处理请求终止的核心入口 无论正常结束还是异常终止 

        For example, the API server can abort a request when the client
        disconnects.
        """
        assert RequestStatus.is_finished(finished_status) #finished_status value eg. <RequestStatus.FINISHED_ABORTED: 8>
        if isinstance(request_ids, str):
            request_ids = (request_ids,)
        else:
            request_ids = set(request_ids)

        running_requests_to_remove = set()      #正在运行中的请求(需要从self.running移除)
        waiting_requests_to_remove = []         #等待中的请求(需要从self.waiting中移除)
        valid_requests = []                     #最终需要处理的合法请求

        # ==================== 第一遍遍历：收集需要移除的请求 ====================
        # 目的：先把所有要处理的请求找出来,并按所在队列分类
        # 这样后面可以一次性批量从队列中移除,提高效率(避免边遍历边删除导致的问题)
        # First pass: collect requests to remove from queues
        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # Invalid request ID.
                continue

            valid_requests.append(request)
            # 根据当前状态决定应该从哪个队列移除
            if request.status == RequestStatus.RUNNING:
                running_requests_to_remove.add(request)
            else:
                waiting_requests_to_remove.append(request)

        # Remove all requests from queues at once for better efficiency
        # ==================== 批量从队列中移除请求(性能关键) ====================
        # 一次性移除比逐个 remove 效率高很多,尤其在同时取消大量请求时
        if running_requests_to_remove:
            self.running = remove_all(self.running, running_requests_to_remove)
        if waiting_requests_to_remove:
            self.waiting.remove_requests(waiting_requests_to_remove)

        # Second pass: set status and free requests
        # ==================== 第二遍遍历：更新状态并释放资源 ====================
        for request in valid_requests:
            request.status = finished_status
            self._free_request(request)

    def _free_request(self, request: Request) -> dict[str, Any] | None:
        """
        释放一个已完成请求所占用的所有资源
        任何请求结束(正常停止、用户 abort、达到 max_tokens、发生错误等)后,
        都会调用此函数来释放资源,避免内存泄漏,尤其是宝贵的 KV Cache.
        
        返回值：如果启用了 KV Connector(分布式 / disaggregated serving),
               可能会返回需要进行 KV 传输的参数,否则返回 None.
        """
        
        assert request.is_finished()

        # ==================== 第一步：处理 KV Connector(跨节点/设备 KV 传输) ====================
        # _connector_finished 会判断是否需要延迟释放 KV blocks(比如要把 KV Cache 传输到其他节点)
        # 返回两个值：
        #   - delay_free_blocks: 是否需要延迟释放 KV blocks(True 表示暂时不释放,等传输完成)
        #   - kv_xfer_params:    KV 传输需要的参数(如果不需要传输则为 None)
        delay_free_blocks, kv_xfer_params = self._connector_finished(request)
        
        # ==================== 释放 Encoder Cache(针对多模态模型) ====================
        # 如果是 Vision-Language Model(比如 Qwen3-VL、Qwen2.5-VL),会释放图像/视频编码后的 cache
        self.encoder_cache_manager.free(request)
        
        # ==================== 记录该请求已完成 ====================
        request_id = request.request_id
        # 把 request_id 加入全局已完成集合(用于后续通知上层)
        self.finished_req_ids.add(request_id)
        
        # 按 client_index 分组记录已完成请求(支持多客户端场景)
        # 例如：不同用户、不同 API 调用方使用同一个 Engine 时,需要分开统计
        if self.finished_req_ids_dict is not None:
            self.finished_req_ids_dict[request.client_index].add(request_id)

        # ==================== 释放 KV Cache Blocks(最重要的一步) ====================
        # 如果不需要延迟释放(正常情况),立即释放该请求占用的 KV Cache 块
        # 这会把 PagedAttention 中的物理 block 标记为可用,供后续请求复用
        # 这是防止 GPU 显存泄漏的核心操作
        if not delay_free_blocks:
            self._free_blocks(request)
        # 返回 KV 传输参数(如果有的话)
        # 上层(如 update_from_output)会根据这个返回值决定是否需要进行跨节点 KV 传输
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

        If reset_running_requests is True, all the running requests will be          如果 reset_running_requests 为 True,则所有正在运行的请求都会被抢占(preempt),并被移动到等待队列中.
        preempted and moved to the waiting queue.
        Otherwise, this method will only reset the KV prefix cache when there        否则,该方法只会在“当前没有任何正在运行的请求占用 KV cache”的情况下,才会重置 KV prefix cache.
        is no running requests taking KV cache.
        
        Args:
            reset_running_requests: 如果为 True,将强制抢占(Preempt)所有正在运行的请求.
            reset_connector: 是否同时重置外部连接器(Connector)的缓存.
        """
        if reset_running_requests:
            # For logging.
            timestamp = time.monotonic()
            # Invalidate all the current running requests KV's by pushing them to     为了能彻底清理缓存,必须将所有running队列中的请求失效
            # the waiting queue. In this case, we can reduce the ref count of all     通过将它们退会waiting队列,所有被占用的block引用计数会降为0
            # the kv blocks to 0 and thus we can make sure the reset is successful.   这样前缀缓存哈希表才能被安全清空
            # Preempt in reverse order so the requests will be added back to the      使用reverse顺序pop是为了保持先进先出的逻辑
            # running queue in FIFO order.                                            队列尾部的先出,重新入队能维持原有的优先级顺序
            while self.running:
                request = self.running.pop()
                self._preempt_request(request, timestamp)                             ## 执行抢占操作：释放该请求占用的物理块,并将其状态改回 WAITING
                # NOTE(zhuohan): For async scheduling, we need to discard the latest   特殊处理异步调度,丢弃当前正在生成的token占位符,避免回复运行后出现的token输出
                # output token on the fly to avoid a redundant repetitive output token.
                request.num_output_placeholders = 0
                request.discard_latest_async_tokens = True

            # Clear scheduled request ids cache. Since we are forcing preemption       清楚上一步已调度请求ID的缓存  因为我们强行终止了当前步骤的所有请求
            # + resumption in the same step, we must act as if these requests were     必须让ModelRunner知道这些请求,需要从持久化批次(persistent batch)中刷新掉
            # not scheduled in the prior step. They will be flushed from the
            # persistent batch in the model runner.
            self.prev_step_scheduled_req_ids.clear()

        reset_successful = self.kv_cache_manager.reset_prefix_cache()                  #执行底层的kv cache重置,调用kv cache manager清空哈希映射表(映射Token序列到Block的关系)
        if reset_running_requests and not reset_successful:                            #异常检查：如果我们已经清理了所有运行请求,重置依然失败,通常是因为有“影子引用”存在(例如正在进行的 P/D 远程 KV 传输).
            raise RuntimeError(
                "Failed to reset KV cache even when all the running requests are "
                "preempted and moved to the waiting queue. This is likely due to "
                "the presence of running requests waiting for remote KV transfer, "
                "which is not supported yet."
            )

        if reset_connector:                                                            #处理外部连接器缓存
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
        """
        汇总并生成调度器的运行统计数据(SchedulerStats) 该方法汇集了队列长度、kv cache状态,投机性能等关键监控指标
        """
        
        if not self.log_stats:                                                        #如果未开启统计日志功能,则直接返回None以减少开销
            return None
        prefix_cache_stats = self.kv_cache_manager.make_prefix_cache_stats()          #收集kv cache相关统计,从本地kv cache管理器获取前缀缓存的命中信息
        assert prefix_cache_stats is not None
        connector_prefix_cache_stats = self._make_connector_prefix_cache_stats()      #获取外部connector相关的前缀缓存统计(用于分布式或外部kv 存储)
        eviction_events = (                                                           #从指标收集器中排空drain驱逐事件
            self.kv_metrics_collector.drain_events()                                  #drain 意味着获取自上次调用以来的所有事件并清空记录器,防止数据重复
            if self.kv_metrics_collector is not None
            else []
        )
        spec_stats = spec_decoding_stats                                              #准备组件穿透数据
        connector_stats_payload = (
            kv_connector_stats.data if kv_connector_stats else None                   ## 如果存在外部 KV 连接器统计数据,则提取其内部的 payload 数据
        )
        return SchedulerStats(
            num_running_reqs=len(self.running),
            num_waiting_reqs=len(self.waiting),
            kv_cache_usage=self.kv_cache_manager.usage,                               #kv cache物理占用率
            prefix_cache_stats=prefix_cache_stats,                                    #缓存命中与事件详情
            connector_prefix_cache_stats=connector_prefix_cache_stats,
            kv_cache_eviction_events=eviction_events,
            spec_decoding_stats=spec_stats,                                           #专门组件的统计数据(由外部传入)
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
        """ 
        优雅关闭所有和kv cache相关的外部组件
        """
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
        Invoke the KV connector request_finished() method if applicable.                    如果存在kv connecor(外部kv 系统),调用它的request_finished()

        Returns optional kv transfer parameters to be included with the                     返回一些kv传输相关参数,这些信息可能会被附加到请求输出中
        request outputs.
        """
        if self.connector is None:
            return False, None

        # Free any out-of-window prefix blocks before we hand the block table to            先清理滑动窗口之外的block  ,先把kv交给外部系统之前,先把已经不会再参与attention的旧block清理掉   
        # the connector.                                                                    否则 
        self.kv_cache_manager.remove_skipped_blocks(
            request_id=request.request_id,
            total_computed_tokens=request.num_tokens,
        )

        block_ids = self.kv_cache_manager.get_block_ids(request.request_id)                 #拿到该 request 当前所有 block 的 ID.block_ids的结构一般是tuple[list[int], ...]  (按 kv cache group 分组)

        if not isinstance(self.connector, SupportsHMA):
            # NOTE(Kuntai): We should deprecate this code path after we enforce             老版本 connector 不支持HMA
            # all connectors to support HMA.                                                这里是旧接口路径 未来可能被移除  因为旧接口只支持单组
            # Hybrid memory allocator should be already turned off for this
            # code path, but let's double-check here.
            assert len(self.kv_cache_config.kv_cache_groups) == 1
            return self.connector.request_finished(request, block_ids[0])

        return self.connector.request_finished_all_groups(request, block_ids)

    def _update_waiting_for_remote_kv(self, request: Request) -> bool:
        """
        KV Connector: check if the request_id is finished_recving.                          kv connector相关逻辑,检查这个request的远程kv 是否已经接收完成

        The finished_recving_kv_req_ids list is populated                                   finished_recving_kv_req_ids这个列表是在前一次step()调用中的update_from_output阶段被填充的            
        on the previous steps()'s update_from_output based                                  其内容是根据worker侧的connector更新得到的,表示哪些request的KV已经传输完成
        on the worker side connector.
                                                                                            本函数作用：判断当前request的kv是否已经ready
        When the kv transfer is ready, we cache the blocks                                  如果ready：将这些block正式cache(纳入Prefix cache体系)
        and the request state will be moved back to WAITING from                            request 状态从WAITING_FOR_REMOTE_KV -> WAITING(可持续调度)
        WAITING_FOR_REMOTE_KV.
        
        返回：True:kv已经就绪,可以参与调度; False:kv未就绪,继续等待
        """
        assert self.connector is not None
        if request.request_id not in self.finished_recving_kv_req_ids:                      #检查kv 是否已经接收完成
            return False                                                                    

        
        #🚩 kv已经传输完成,但是需要区分成功和失败
        if request.request_id in self.failed_recving_kv_req_ids:                            #若远程kv加载失败(可能是部分失败),这个失败列表哪来的？scheduler 自己“并不知道失败”,它只是“被告知失败,是 worker / KV connector 在传输过程中检测失败后,上报给 scheduler 的
            # Request had KV load failures; num_computed_tokens was already                 # request.num_computed_tokens 已经在其他地方(_update_requests_with_invalid_blocks)被更新为有效的kv token数.不理解该句,chatgpt回答:失败发生后,系统会重新计算：哪些 KV 是“还有效的”,只把这些有效的计入 num_computed_tokens
            # updated in _update_requests_with_invalid_blocks
            if request.num_computed_tokens:                                                 #仍有一部分是有效的(可能部分加载成功) 把这些有效kv 加入cache
                # Cache any valid computed tokens.
                self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)
            else:                                                                           
                # No valid computed tokens, release allocated blocks.                       #完全没有有效kv->释放之前为这个reqeust分配的block,相当于远程加载彻底失败,注意后续重试仍可能命中本地cache
                # There may be a local cache hit on retry.
                self.kv_cache_manager.free(request)

            self.failed_recving_kv_req_ids.remove(request.request_id)                       #从“失败集合”中移除
        else:                                                                               #🚩远程kv 加载成功 
            # Now that the blocks are ready, actually cache them.
            (block_ids,) = self.kv_cache_manager.get_block_ids(request.request_id)          #拿到当前request的block id(已经分配好的)
            num_computed_tokens = len(block_ids) * self.block_size                          #计算这些block对应多少token
            # Handle the case where num request tokens less than one block.
            num_computed_tokens = min(num_computed_tokens, request.num_tokens)              #防止超过实际token数(例如最后一个token没填满).request.num_tokens为已经确定存在的token数
            if num_computed_tokens == request.num_tokens:                                   #特殊处理：如果刚好等于request.num_tokens , 减1 避免全部命中导致某些调度/缓存问题
                num_computed_tokens -= 1
            # This will cache the blocks iff caching is enabled.                            #把这些 block 加入 prefix cache(如果开启)
            self.kv_cache_manager.cache_blocks(request, num_computed_tokens)            

            # Update the request state for scheduling.                                      #更新 request 状态：表示这些 token 已经“计算完成”(其实是加载完成)
            request.num_computed_tokens = num_computed_tokens

        # Return that we are ready.
        self.finished_recving_kv_req_ids.remove(request.request_id)                         # 从“已接收完成集合”中移除(避免重复处理)
        return True

    def _update_from_kv_xfer_finished(self, kv_connector_output: KVConnectorOutput):
        """
                                                                                            把 worker 侧 KV 传输结果(发送/接收完成)同步到调度器状态里
        KV Connector: update the scheduler state based on the output.                       根据执行输出(output)来更新调度器(scheduler)的状态,

        The Worker side connectors add finished_recving and                                 在worker侧,connector会把2类请求信息写入output:
        finished_sending reqs to the output.                                                finished_sending:表示该请求的kv已经发送完成->可以释放对应的block
        * if finished_sending: free the blocks
        # if finished_recving: add to state so we can                                       finished_recving:表示该请求的kv已经接收完成->将其加入调度状态,使得该请求可以在下一步被重新调度执行
            schedule the request during the next step.
        """

        if self.connector is not None:                                                      #让connector自己更新内部状态,一些connector(如lmcache)可能需要根据output更新内部状态
            self.connector.update_connector_output(kv_connector_output)                     #比如pending请求,统计信息等

        # KV Connector:: update recv and send status from last step.                          处理接收完成的请求(finished_recving)
        for req_id in kv_connector_output.finished_recving or ():
            logger.debug("Finished recving KV transfer for request %s", req_id)             #标记这个request：它的KV已经加载到本地(gpu)了
            self.finished_recving_kv_req_ids.add(req_id)                                    #后续：_update_waiting_for_remote_kv()会用这个集合,把 request 从 WAITING_FOR_REMOTE_KV → WAITING
        for req_id in kv_connector_output.finished_sending or ():                           #处理发送完成的请求(finished_sending)
            logger.debug("Finished sending KV transfer for request %s", req_id)
            assert req_id in self.requests                                                  #确保这个request还在scheduler管理中
            self._free_blocks(self.requests[req_id])                                        #kv已经成功写入外部(比如lmcache)本地GPU上这份KV可以释放,注意这里释放的是本地副本,数据已经安全存在外部缓存中

    def _update_requests_with_invalid_blocks(
        self,
        requests: Iterable[Request],
        invalid_block_ids: set[int],
        evict_blocks: bool = True,
    ) -> tuple[set[str], int, set[int]]:
        """
        Identify and update requests affected by invalid KV cache blocks.                   是被并更新受到那些 无效kv cache block影响到的请求

        This method scans the given requests, detects those with invalid blocks             这个方法会遍历输入的一批请求：
        and adjusts their `num_computed_tokens` to the longest valid prefix.                - 找出哪些请求是用了无效的block
        For observability, it also accumulates the total number of tokens that              - 并把这些请求的num_computed_tokens阶段到 仍然有效的最长前缀
        will need to be recomputed across all affected requests.                            同时,为了方便观测(比如统计/日志),它还会累计所有受影响请求中,需要重新计算的 token 总数.

        Args:
            requests: The set of requests to scan for invalid blocks.                        需要扫描的一组请求(检查是否包含无效 block)
            invalid_block_ids: IDs of invalid blocks.                                        被判定为无效的block的ID集合(比如驱逐、损坏、远程加载失败等)
            evict_blocks: Whether to collect blocks for eviction (False for                  是否需要手机需要被驱逐的block (对于异步请求通常为 False,因为这些 block 还没真正进入 cache)
                async requests which aren't cached yet).

        Returns:
            tuple:
                - affected_req_ids (set[str]): IDs of requests impacted by                  受到无效 block 影响的 request_id 集合
                invalid blocks.
                - total_affected_tokens (int): Total number of tokens that must             所有受影响请求中,需要重新计算的 token 总数
                be recomputed across all affected requests.
                - blocks_to_evict (set[int]): Block IDs to evict from cache,                需要从 cache 中移除的 block ID 集合 包括无效的Block,以及以来这些block的下游block(因为前缀断了)
                including invalid blocks and downstream dependent blocks.
        example
            例如一个 request 的 block 序列为 [A B] [C D] [E F] [G H](block_size=2),如果中间的 [E F] 这个 block 变成 invalid,
            那么系统会把有效前缀截断为 [A B] [C D],并将 num_computed_tokens 回退到对应的位置;由于 KV cache 是严格的前缀依赖结构,
            [E F] 之后的 [G H] 也会一并失效,必须重新计算,同时这些无效 block(以及依赖它们的后续 block)也会被标记为需要从 cache 中清理.
        
        """
        affected_req_ids: set[str] = set()                                                   #被影响的request id
        total_affected_tokens = 0                                                            #总共需要重算的token数
        blocks_to_evict: set[int] = set()                                                    #需要从cache中清理的block
        # If a block is invalid and shared by multiple requests in the batch,                #用于处理多个request共享一个invalid block的情况
        # these requests must be rescheduled, but only the first will recompute              #确保这个block制备第一个request负责重算
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
            for idx, block_id in zip(range(req_num_computed_blocks), req_block_ids):            #逐block检查是否invalid
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
        Handle requests affected by invalid KV cache blocks.                        处理所有收到无效 kv block 影响的请求 

        Returns:
            Set of affected request IDs to skip in update_from_output main loop.    返回需要在主调度循环中跳过处理的request_id
        """
        should_fail = not self.recompute_kv_load_failures                           #是否采用失败策略,还是重算策略,true直接失败 false尝试回退并重新计算

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
