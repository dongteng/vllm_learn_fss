# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Define EC connector functionality mixin for model runners.
"""

from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import TYPE_CHECKING

import torch

from vllm.distributed.ec_transfer import get_ec_transfer, has_ec_transfer
from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorBase
from vllm.logger import init_logger
from vllm.v1.outputs import ECConnectorOutput

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


# Defined as a EC connector functionality mixin for ModelRunner (GPU, TPU)  这段代码本质是在给ModelRunner增加一个外部缓存(EC,External Connector)通信能力,而且是用mixin的方式做到可插拔
class ECConnectorModelRunnerMixin:
    @staticmethod
    def maybe_save_ec_to_connector(
        encoder_cache: dict[str, torch.Tensor],                                                       #当前GPU上已有的kv cache
        mm_hash: str,                                                                                 #多模态或内容的唯一标识
    ):
        if not has_ec_transfer():                                                                     #检查是否启用了EC传输
            logger.debug("Not have ec transfer please check")
            return
        connector = get_ec_transfer()                                                                 #获取具体的缓存传输实现(可能是lmcache/RPC/RDMA)
        connector.save_caches(encoder_cache=encoder_cache, mm_hash=mm_hash)                           #把当前GPU的kv cache存出去(比如GPU到外部缓存系统)

    @staticmethod
    def get_finished_ec_transfers(
        scheduler_output: "SchedulerOutput",                                                          #调度器:知道哪些request已经结束
    ) -> tuple[set[str] | None, set[str] | None]:                                                      
        if has_ec_transfer():                                                                         #如果开启了ec
            return get_ec_transfer().get_finished(scheduler_output.finished_req_ids)                  #查询哪些request的cache:已经发送完成(sending)  已经接收完成(recving)
        return None, None                                                                             #如果没启用

    @staticmethod
    def maybe_get_ec_connector_output(
        scheduler_output: "SchedulerOutput",
        encoder_cache: dict[str, torch.Tensor],
        **kwargs,
    ) -> AbstractContextManager[ECConnectorOutput | None]:                                             #获取context manager(关键设计) 返回一个上下文管理器(with语法用)
        return (
            ECConnectorModelRunnerMixin._get_ec_connector_output(
                scheduler_output, encoder_cache, **kwargs
            )
            if has_ec_transfer()                                                                        #如果没启用:返回一个空context(这样外边)可以统一写with不用if
            else nullcontext()
        )

    # This context manager must be used within an active forward context.                               #必须在forward过程中使用
    # It encapsulates the entire EC connector lifecycle within execute_model                            它负责把缓存传输嵌入到模型执行生命周期中
    @staticmethod
    @contextmanager
    def _get_ec_connector_output(
        scheduler_output: "SchedulerOutput",
        encoder_cache: dict[str, torch.Tensor],
        **kwargs,
    ) -> Generator[ECConnectorOutput, None, None]:
        output = ECConnectorOutput()                                                                     #用来记录最终结果(哪些请求传输完成)

        ec_connector = get_ec_transfer()                                                                 #获取底层实现(如lmcache connector)
        assert isinstance(ec_connector, ECConnectorBase)
        assert scheduler_output.ec_connector_metadata is not None                                        #调取其提前准保好的metadata(比如request->cache映射)
        ec_connector.bind_connector_metadata(scheduler_output.ec_connector_metadata)                     #把这些metadata绑定到connector  相当于告诉lmcache:这批请求是谁,它们的cache key是什么

        if not ec_connector.is_producer:                                                                 #如果当前节点不是生产者  说明它是消费者 需要用别人的cache
            ec_connector.start_load_caches(encoder_cache, **kwargs)                                      #开始加载cache(通常是异步) 实例:gpu1在decode前,从lmcache拉gpu0的kv cache

        try:
            yield output                                                                                 #把控制权交给外部
        finally:
            output.finished_sending, output.finished_recving = (                                         #forward结束后  查询哪些请求已经发送完成,已经接收完成
                ec_connector.get_finished(scheduler_output.finished_req_ids)
            )

            ec_connector.clear_connector_metadata()                                                      #清理metadata
