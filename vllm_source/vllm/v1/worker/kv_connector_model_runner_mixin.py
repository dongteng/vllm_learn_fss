# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Define KV connector functionality mixin for model runners.
"""

import copy
from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import TYPE_CHECKING

import torch

from vllm.attention.backends.abstract import AttentionBackend
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.distributed.kv_transfer import (
    ensure_kv_transfer_shutdown,
    get_kv_transfer_group,
    has_kv_transfer_group,
)
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    KVConnectorOutput,
    ModelRunnerOutput,
)
from vllm.v1.worker.utils import AttentionGroup

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


# Defined as a kv connector functionality mixin for ModelRunner (GPU, TPU)
class KVConnectorModelRunnerMixin:
    """
    让modelrunner具备跨设备/跨进程kv cache传输能力
    """
    @staticmethod
    def maybe_setup_kv_connector(scheduler_output: "SchedulerOutput"):                                  #输入是调度器输出结果
        # Update KVConnector with the KVConnector metadata forward().                                   #在forward前,把kv相关的元信息传给kvconnector
        if has_kv_transfer_group():                                                                     #判断当前环境有没有启用kv传输组
            kv_connector = get_kv_transfer_group()                                                      #获取kv传输对象(类似一个全局的kv管理器)
            assert isinstance(kv_connector, KVConnectorBase)
            assert scheduler_output.kv_connector_metadata is not None                                   #确保调度器已经准备好了kv metadata
            kv_connector.bind_connector_metadata(scheduler_output.kv_connector_metadata)                #把 metadata “绑定”到 kvconnector 上

            # Background KV cache transfers happen here.                                                kv cache的后台传输
            # These transfers are designed to be async and the requests                                 异步传输、和当前正在执行的请求可能无关 eg.当前在生成A请求,同时后台在帮B请求加载KV Cache
            # involved may be disjoint from the running requests.
            # Do this here to save a collective_rpc.
            kv_connector.start_load_kv(get_forward_context())

    @staticmethod
    def ensure_kv_transfer_shutdown() -> None:
        # has_kv_transfer_group can be None during interpreter shutdown.                                确保kv传输模块被正确关闭 场景:程序退出 worker结束 防止后台线程/通信资源泄露
        if has_kv_transfer_group and has_kv_transfer_group():  # type: ignore[truthy-function]
            ensure_kv_transfer_shutdown()

    @staticmethod
    def maybe_wait_for_kv_save() -> None:                                                               #需要的时候,等待kv cache的保存操作完成
        if has_kv_transfer_group():                                                                     #是否开启了kv cache系统
            get_kv_transfer_group().wait_for_save()                                                     #等待所有正在进行的 KV cache “写出 / 保存”操作完成

    @staticmethod
    def get_finished_kv_transfers(
        scheduler_output: "SchedulerOutput",
    ) -> tuple[set[str] | None, set[str] | None]:                                                       #获取已完成kv cache传输的请求id集合
        if has_kv_transfer_group():
            return get_kv_transfer_group().get_finished(                                                #返回已经完成发送的请求ID集合, 已完成接收的请求ID集合
                scheduler_output.finished_req_ids
            )
        return None, None

    @staticmethod
    def kv_connector_no_forward(
        scheduler_output: "SchedulerOutput", vllm_config: VllmConfig
    ) -> ModelRunnerOutput:
        # KV send/recv even if no work to do.                                                           #即使当前没有真正的forward计算任务,也仍然需要推进kv cache的send/recv流程
        with (
            set_forward_context(None, vllm_config),                                                     #创建一个空forward context这里传入None表示,当前并不会真正执行model forward,但kv connecot的内部逻辑依然依赖环境,因此仍需要构造上下文
            KVConnectorModelRunnerMixin._get_kv_connector_output(                                       #获取kv connector输出对象,内部会推进kv send/recv,更新transfer状态, 收集connector output
                scheduler_output, wait_for_save=False                                                   #wait_for_save=False表示 不阻塞等待kv save完成,仅推进异步通信流程
            ) as kv_connector_output,
        ):
            pass

        if kv_connector_output.is_empty():                                                              #如果kv connector没有任何输出,说明没有kv send, 没有kv recv,没有需要同步的transfer
            return EMPTY_MODEL_RUNNER_OUTPUT

        output = copy.copy(EMPTY_MODEL_RUNNER_OUTPUT)                                                   #浅拷贝
        output.kv_connector_output = kv_connector_output
        return output

    @staticmethod
    def maybe_get_kv_connector_output(
        scheduler_output: "SchedulerOutput",
    ) -> AbstractContextManager[KVConnectorOutput | None]:                                              #返回一个上下文管理器,进入这个上下文之后,可能拿到KVConnectorOutput,也可能拿到None
        return (
            KVConnectorModelRunnerMixin._get_kv_connector_output(scheduler_output)
            if has_kv_transfer_group()
            else nullcontext()
        )

    # This context manager must be used within an active forward context.                               这个上下文管理器必须在一个已经激活的forward context中使用
    # It encapsulates the entire KV connector lifecycle within execute_model                            它负责在excute_model执行期间,封装整个kv connector的生命周期
    @staticmethod
    @contextmanager
    def _get_kv_connector_output(
        scheduler_output: "SchedulerOutput", wait_for_save: bool = True                                 #scheduler阶段输出的信息, 是否等待save操作完成
    ) -> Generator[KVConnectorOutput, None, None]:
        output = KVConnectorOutput()                                                                    #创建一个kvconnectorOuput对象 用于最终保存connector执行结果

        # Update KVConnector with the KVConnector metadata forward().                                   #获取全局kv transfer group 它是真正负责kv cache传输的connector
        kv_connector = get_kv_transfer_group()
        assert isinstance(kv_connector, KVConnectorBase)
        assert scheduler_output.kv_connector_metadata is not None
        kv_connector.bind_connector_metadata(scheduler_output.kv_connector_metadata)                    #将scheduler生成的metadata绑定到kv connector后续forward/load/save都依赖这些metadata

        # Background KV cache transfers happen here.                                                    这里开始后台 kv cache传输
        # These transfers are designed to be async and the requests                                     这些传输是异步的 execute_model不需要停下来等它们完成
        # involved may be disjoint from the running requests.                                           参与传输的request,可能和当前的running requests并不完全一致
        # Do this here to save a collective_rpc.                                                        在这里启动load 可以避免后面额外进行collective_rpc通信
        kv_connector.start_load_kv(get_forward_context())
        try:
                                                                                                        #外部代码会在这里真正执行模型forward
            yield output
        finally:
            if wait_for_save:
                kv_connector.wait_for_save()

            output.finished_sending, output.finished_recving = (                                        #获取已经完成发送/接收的request
                kv_connector.get_finished(scheduler_output.finished_req_ids)
            )
            output.invalid_block_ids = kv_connector.get_block_ids_with_load_errors()                    #获取load失败的block_id  比如网络错误 cache miss  block损坏

            output.kv_connector_stats = kv_connector.get_kv_connector_stats()                           #获取connector的统计信息
            output.kv_cache_events = kv_connector.get_kv_connector_kv_cache_events()

            kv_connector.clear_connector_metadata()

    @staticmethod
    def use_uniform_kv_cache(
        attn_groups: list[list[AttentionGroup]],
        cache_dtype: CacheDType,
    ) -> bool:
        """
        Determines whether a uniform KV layout should be used.                                           判断是否应该使用统一(uniform)的kv cache布局
        A uniform layout means all layers KV caches will share the same                                  所谓统一布局指的是 所有layer的kv cache共享同一个底层tensor
        underlying tensor, where for a given block number, the respective                                并且 对一个指定的Block编号,所有Layer对应的kv 数据会连续(contiguous)存储在一起
        KV data for all layers will be contiguous.
        This will allow efficient KV transfer of per-block KV data for all                               这样做可以实现:一次高效传输 某个block在所有layers上的kv 数据
        layers at once.
        Note this layout will only be applied given 3 conditions:                                        注意只有满足以下3个条件,才会启用这种布局
        1. The KV Cache config contains just a single group where all layers                             1.kv cache配置中只包含一个group,并且所有layer具有相同page size
            have the same page size.
        2. A KV connector is configured, and the KV connector instance prefers                           2.已配置kv connector并且该connector实例更倾向于使用这种布局
            to use this layout (prefer_cross_layer_blocks() returns True)                                  即prefer_cross_layer_blocks返回True
        2. The flash attention backend supports this layout                                              3. flash attention backend支持这种布局
            (get_kv_cache_stride_order(True) includes a placement for a                                    即 get_kv_cache_stride_order(True)返回的 stride/order 中包含 num_layers 维度的位置定义。
            num_layers dimension)

        Note that the actual placement of the num_layers dimensions                                      需要注意:num_layers维度在统一tensor中的实际排列方式,最终由attention backend决定
        in the unified layers tensors will be determined by the attention               
        backend.    
        Thus, the layers KV data may still not be contiguous per block                                   因此如果attention backend不支持,即便启用了统一布局,每个block的layer kv数据也仍然可能不是连续存储的
        if the attention backend does not support it.

        Args:
            attn_groups: The list of attention groups for this model                                     当前模型的attention group列表
            cache_dtype: The KV cache dtype                                                              kv cache使用的数据类型
        Returns:
            True if we should use a uniform KV cache layout.                                             如果应该使用统一 KV cache 布局则返回 True
        """

        if not has_kv_transfer_group():                                                                  #如果没有kv transfer就没必要用uniform kv layout
            return False
        if not get_kv_transfer_group().prefer_cross_layer_blocks:                                        #如果connector不需要uniform layout那么就没必要
            return False

        if len(attn_groups) != 1 or len(attn_groups[0]) != 1:                                            #uniform kv layout的前提之一 必须只有一个attention group , 一个kv cache group
            return False                                                                                 #因为uniform layout需要所有layers的kv shape完全一致

        attn_group = attn_groups[0][0]                                                                   #取出唯一的attention group
        kv_cache_spec = attn_group.kv_cache_spec                                                         #获取这个attention group对应的kv cache配置
        if not isinstance(kv_cache_spec, AttentionSpec):                                                 #确保kv cache配置是AttentionSpec类型,因为Uniform kv layout目前只支持attention类型kv cache, 如果是mamba  state-space model , 特殊cache backend则不支持
            return False

        attn_backend = attn_group.backend                                                                #取出当前attention backend如flashattention  flashinfer tritonbackent
        kv_cache_shape = attn_backend.get_kv_cache_shape(                                                #获取kv cache tensor的shape
            1234,
            kv_cache_spec.block_size,
            kv_cache_spec.num_kv_heads,
            kv_cache_spec.head_size,
            cache_dtype_str=cache_dtype,
        )

        try:
            kv_cache_stride_order = attn_backend.get_kv_cache_stride_order(                              #获取kv cache tensor的stride/layout顺序
                include_num_layers_dimension=True                                                        #这里本质是在问backend 你支不支持统一layers tensor
            )
        except (AttributeError, NotImplementedError):
            return False

        # check that attention backend include a layers dimension                                        #为什么加一 原本 KV cache shape: [num_blocks, block_size, num_heads, head_dim]
        return len(kv_cache_stride_order) == len(kv_cache_shape) + 1                                     #加入uniform layout后

    @staticmethod
    def allocate_uniform_kv_caches(
        kv_cache_config: KVCacheConfig,
        attn_groups: list[list[AttentionGroup]],
        cache_dtype: CacheDType,
        device: torch.device,
        kernel_block_sizes: list[int],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, type[AttentionBackend]]:
        """
        Initializes and reshapes KV caches for the simple case where all                                 为所有layers使用相同kv layout的简单场景初始化并重新组织(reshape)kv cache
        layers have the same layout.                                                                     这里相同layout指的是所有attention layers的kv cache具有一致的shape,block_size,memory layout
                                                                                                         因此可以把所有layers的kv cache合并到同一个统一的大tensor中       
        This function assumes use_uniform_kv_cache() returned True.                                      该函数默认use_uniform_kv_cache()已经返回True

        Args:
            kv_cache_config: The KV cache config
            attn_groups: The list of attention groups for this model
            cache_dtype: The KV cache dtype
            device: The torch device to allocate on.
            kernel_block_sizes: The kernel block sizes for each KV cache group.                           每个kv cache group对应的kernel block size
        Returns:
            A tuple (kv_caches, cross_layers_kv_cache, attn_backend) where:                               kv_caches: 根据layer名字找到对应layer的kv cache buffer
                kv_caches is a dict mapping between layer names to their
                    corresponding memory buffer for KV cache.
                cross_layers_kv_cache is the cross layers kv cache tensor                                 cross_layers_kv_cache:所有layers共享的统一kv cache tensor
                attn_backend is the attention backend matching this tensor                                attn_backend:与该kv cache tensor匹配的attention backend
        """
        attn_group = attn_groups[0][0]                                                                    #取出唯一的attention group
        kv_cache_spec = attn_group.kv_cache_spec                                                          #获取kv cache的规格配置
        assert isinstance(kv_cache_spec, AttentionSpec)

        tensor_sizes = set(
            kv_cache_tensor.size for kv_cache_tensor in kv_cache_config.kv_cache_tensors                  #收集所有layer kv tensor的size
        )
        assert len(tensor_sizes) == 1
        tensor_size = tensor_sizes.pop()

        page_size = kv_cache_spec.page_size_bytes
        assert tensor_size % page_size == 0
        num_blocks = tensor_size // page_size
        num_layers = len(kv_cache_config.kv_cache_tensors)
        total_size = tensor_size * num_layers

        assert len(kernel_block_sizes) == 1
        kernel_block_size = kernel_block_sizes[0]
        num_blocks_per_kv_block = kv_cache_spec.block_size // kernel_block_size
        kernel_num_blocks = num_blocks * num_blocks_per_kv_block

        attn_backend = attn_group.backend
        kv_cache_shape = attn_backend.get_kv_cache_shape(
            kernel_num_blocks,
            kernel_block_size,
            kv_cache_spec.num_kv_heads,
            kv_cache_spec.head_size,
            cache_dtype_str=cache_dtype,
        )

        # prepend a num_layers dimension into the shape
        kv_cache_shape = (num_layers,) + kv_cache_shape

        try:
            kv_cache_stride_order = attn_backend.get_kv_cache_stride_order(
                include_num_layers_dimension=True
            )
            assert len(kv_cache_stride_order) == len(kv_cache_shape)
        except (AttributeError, NotImplementedError):
            kv_cache_stride_order = tuple(range(len(kv_cache_shape)))

        kv_cache_shape = tuple(kv_cache_shape[i] for i in kv_cache_stride_order)

        logger.info("Allocating a cross layer KV cache of shape %s", kv_cache_shape)

        # allocate one contiguous buffer for all layers
        cross_layers_kv_cache = (
            torch.zeros(total_size, dtype=torch.int8, device=device)
            .view(kv_cache_spec.dtype)
            .view(kv_cache_shape)
        )

        # Maintain original KV shape view.
        inv_order = [
            kv_cache_stride_order.index(i) for i in range(len(kv_cache_stride_order))
        ]
        permuted_kv_cache = cross_layers_kv_cache.permute(*inv_order)

        kv_caches = {}
        for i, kv_cache_tensor in enumerate(kv_cache_config.kv_cache_tensors):
            tensor = permuted_kv_cache[i]
            for layer_name in kv_cache_tensor.shared_by:
                kv_caches[layer_name] = tensor

        return kv_caches, cross_layers_kv_cache, attn_backend
