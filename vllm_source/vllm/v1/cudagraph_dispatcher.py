# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from itertools import product

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor
from vllm.logger import init_logger

logger = init_logger(__name__)


class CudagraphDispatcher:
    """
    Runtime cudagraph dispatcher to dispatch keys for multiple set of                               运行时cuda graph 分发器, 用于在多组CUDA Graph之间分发对应的key
    cudagraphs.

    The dispatcher stores two sets of dispatch keys, one for PIECEWISE and one                      dispatcher内部维护两套dispatch key: 一套用于PIECEWISE模式, 一套FULL模式
    for FULL cudagraph runtime mode. The keys are initialized depending on                          这些key会根据 attention backend是否支持 CompilationConfig中配置的cudagraph mode进行初始化
    attention support and what cudagraph mode is set in CompilationConfig. The                      dispatcher中保存的这些key,是运行哪些cudagraph可以合法调度的唯一可信来源
    keys stored in dispatcher are the only source of truth for valid
    cudagraphs that can be dispatched at runtime.

    At runtime, the dispatch method generates the runtime cudagraph mode (FULL,                     在运行时,dispatch方法会根据输入key:生成当前运行时的ucdagraph mode:FULL  PIECEWISE NONE
    PIECEWISE, or NONE for no cudagraph) and the valid key (batch descriptor)                       生成合法的dispatch key(batch descriptor)
    based on the input key. After dispatching (communicated via forward                             完成dispatch后(通过forward context传递):cudagraph wrapper会信任(dispatcher给出的)dispatch key
    context), the cudagraph wrappers will trust the dispatch key to either                          然后决定是执行capture还是replay 或直接fallback到普通wager runnable(当mode不匹配 或者mode==None时)
    capture or replay (if the mode matches), or pass through to the underlying
    runnable without cudagraph (if the mode does not match or mode is NONE).                        简单理解这个dispatcher本质上像CUDA Graph 的运行时路由器 + 调度中心
    """

    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.uniform_decode_query_len = (
            1
            if not self.vllm_config.speculative_config
            else 1 + self.vllm_config.speculative_config.num_speculative_tokens
        )

        # Dict to store valid cudagraph dispatching keys.
        self.cudagraph_keys: dict[CUDAGraphMode, set[BatchDescriptor]] = {
            CUDAGraphMode.PIECEWISE: set(),
            CUDAGraphMode.FULL: set(),
        }

        assert (
            not self.compilation_config.cudagraph_mode.requires_piecewise_compilation()              #不需要piecewise compilation(不一次把整个模型forward编译成一个大图,而是拆分成多个小块分别编译)
            or self.compilation_config.is_attention_compiled_piecewise()                             #或者attention已支持piece compile
        ), (
            "Compilation mode should be CompilationMode.VLLM_COMPILE when "
            "cudagraph_mode piecewise cudagraphs is used, "
            "and attention should be in splitting_ops or "
            "inductor splitting should be used. "
            f"cudagraph_mode={self.compilation_config.cudagraph_mode}, "
            f"compilation_mode={self.compilation_config.mode}, "
            f"splitting_ops={self.compilation_config.splitting_ops}"
        )

        self.keys_initialized = False

    def _create_padded_batch_descriptor(
        self, num_tokens: int, uniform_decode: bool, has_lora: bool
    ) -> BatchDescriptor:
        max_num_seqs = self.vllm_config.scheduler_config.max_num_seqs                                #scheduler允许的最大request数量
        uniform_decode_query_len = self.uniform_decode_query_len                                     #普通decode=1  spec decode=1+spec, 用于判断当前batch是否满足uniform decode 结构
        num_tokens_padded = self.vllm_config.pad_for_cudagraph(num_tokens)                           #为了适配CUDA Graph,把真实token数补齐后的token数 。对token数进行padding, 使其满足cudagraph的固定shape要求

        if uniform_decode and self.cudagraph_mode.has_mode(CUDAGraphMode.FULL):                      #uniform_decode的意思是所有request的query长度完全一致
            num_reqs = num_tokens_padded // uniform_decode_query_len
            assert num_tokens_padded % uniform_decode_query_len == 0
        else:                                                                                        #如果batch不uniform或者不允许FULL,
            uniform_decode = False
            num_reqs = min(num_tokens_padded, max_num_seqs)                                          #这里既然无法精确知道request数,就用token数给一个安全上界

        return BatchDescriptor(
            num_tokens=num_tokens_padded,
            num_reqs=num_reqs,
            uniform=uniform_decode,
            has_lora=has_lora,
        )

    def add_cudagraph_key(
        self, runtime_mode: CUDAGraphMode, batch_descriptor: BatchDescriptor
    ):
        assert runtime_mode in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL], (                      #None模式没有graph key的意义
            f"Invalid cudagraph runtime mode for keys: {runtime_mode}"
        )
        self.cudagraph_keys[runtime_mode].add(batch_descriptor)                                      #eg. {FULL:{BatchDescriptor(160, 32, True, False),BatchDescriptor(320, 64, True, False),}}

    def initialize_cudagraph_keys(
        self, cudagraph_mode: CUDAGraphMode, uniform_decode_query_len: int
    ):
        # This should be called only after attention backend is initialized. So we can               #注意必须在attention backend初始化后调用,因为不同attention backend对cuda graph的支持能力不同,因为有的backend支持FULL  有的只支持PIECEWISE
        # get the correct cudagraph mode after backend support is resolved.
        self.cudagraph_mode = cudagraph_mode

        # LoRA activation cases to specialize the cuda graphs on                                     #处理LoRA场景,CUDA Graph 对 shape / kernel 路径非常敏感,LoRA开启与否可能导致:graph不同  kernel不同  memory layout不同
        if self.vllm_config.lora_config:                                                             #因此有时需要LoRA单独录graph 
            if self.compilation_config.cudagraph_specialize_lora:
                lora_cases = [True, False]                                                           #分别录graph 
            else:
                lora_cases = [True]                                                                  #不区分lora on/off  统一都按LoRA graph处理,这样graph更少,capture更简单
        else:
            lora_cases = [False]                                                                     #没开lora时是False 

        # Note: we create all valid keys for cudagraph here but do not                              这里会提前创建所有合法的cudagraph key, 但并不保证这些key最终一定被真正使用
        # guarantee all keys would be used. For example, if we allow lazy                           例如如果未来支持lazy capturing,那么某些graph key可能永远不会被实际触发
        # capturing in future PR, some keys may never be triggered.                                 #如果mixed mode启用了cudagraph,则提前未mixed batch注册所有合法的graph key
        if cudagraph_mode.mixed_mode() != CUDAGraphMode.NONE:                                       #假设self.compilation_config.cudagraph_capture_sizes=[64,128],lora_cases=[True,False]
            for bs, has_lora in product(                                                            #product会生成(64,True)(64,False)(128,True)(128,False)
                self.compilation_config.cudagraph_capture_sizes, lora_cases
            ):
                self.add_cudagraph_key(
                    cudagraph_mode.mixed_mode(),
                    self._create_padded_batch_descriptor(
                        bs, False, has_lora
                    ).relax_for_mixed_batch_cudagraphs(),
                )

        # if decode cudagraph mode is FULL, and we don't already have mixed                          #如果decode阶段允许使用FULL cudagraph , decode graph与mixed graph是分离管理的
        # mode full cudagraphs then add them here.                                                   #则单独为full decode 注册cudagraph key
        if (
            cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
            and cudagraph_mode.separate_routine()
        ):
            max_num_tokens = (
                uniform_decode_query_len
                * self.vllm_config.scheduler_config.max_num_seqs
            )
            cudagraph_capture_sizes_for_decode = [
                x
                for x in self.compilation_config.cudagraph_capture_sizes
                if x <= max_num_tokens and x >= uniform_decode_query_len
            ]
            for bs, has_lora in product(cudagraph_capture_sizes_for_decode, lora_cases):
                self.add_cudagraph_key(
                    CUDAGraphMode.FULL,
                    self._create_padded_batch_descriptor(bs, True, has_lora),
                )

        self.keys_initialized = True

    def dispatch(
        self,
        num_tokens: int,
        uniform_decode: bool,
        has_lora: bool,
        disable_full: bool = False,
    ) -> tuple[CUDAGraphMode, BatchDescriptor]:
        """
        Given conditions(e.g.,batch descriptor and if using cascade attention),                         #根据当前batch条件(例如batch descriptor, 是否使用cascade attention等)
        dispatch to a cudagraph runtime mode and the valid batch descriptor.                            决定使用哪种 cudagraph runtime mode    使用哪个合法的batch descriptor
        A new batch descriptor is returned as we might dispatch a uniform batch                         注意返回的batch descriptor可能与原始descriptor不同
        to a graph that supports a more general batch (uniform to non-uniform).                         因为一个uniform batch有时会被降级到更通用的mixed/non-uniform cudagraph, 举例[32,5] uniform=True,但系统没有FULL graph,则可能fallback到PIECEWISE mixed graph
        """
        if (                                                                                            #以下情况直接禁用cudagraph
            not self.keys_initialized                                                                   #key还没初始化  当前模式不支持cudagraph token数超过cudagraph支持上限
            or self.cudagraph_mode == CUDAGraphMode.NONE
            or num_tokens > self.compilation_config.max_cudagraph_capture_size
        ):
            return CUDAGraphMode.NONE, BatchDescriptor(num_tokens)

        batch_desc = self._create_padded_batch_descriptor(                                              #创建当前batch的标准descriptor 举例num_tokens=160 uniform_decode=True has_lora=False-> BatchDescriptor(num_tokens=160,num_reqs=32,uniform=True,has_lora=False)
            num_tokens, uniform_decode, has_lora
        )
        relaxed_batch_desc = batch_desc.relax_for_mixed_batch_cudagraphs()

        if not disable_full:
            # check if key exists for full cudagraph
            if batch_desc in self.cudagraph_keys[CUDAGraphMode.FULL]:
                return CUDAGraphMode.FULL, batch_desc

            # otherwise, check if the relaxed key exists
            if relaxed_batch_desc in self.cudagraph_keys[CUDAGraphMode.FULL]:
                return CUDAGraphMode.FULL, relaxed_batch_desc

        # also check if the relaxed key exists for more "general"
        # piecewise cudagraph
        if relaxed_batch_desc in self.cudagraph_keys[CUDAGraphMode.PIECEWISE]:
            return CUDAGraphMode.PIECEWISE, relaxed_batch_desc

        # finally, just return no cudagraphs and a trivial batch descriptor
        return CUDAGraphMode.NONE, BatchDescriptor(num_tokens)
