# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import signal
import socket
from http import HTTPStatus
from typing import Any

import uvicorn
from fastapi import FastAPI, Request, Response

from vllm import envs
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.constants import (
    H11_MAX_HEADER_COUNT_DEFAULT,
    H11_MAX_INCOMPLETE_EVENT_SIZE_DEFAULT,
)
from vllm.entrypoints.ssl import SSLCertRefresher
from vllm.logger import init_logger
from vllm.utils.network_utils import find_process_using_port
from vllm.v1.engine.exceptions import EngineDeadError, EngineGenerateError

logger = init_logger(__name__)

#那里是Future对象？
#在asyncio中：写成本身不是Future也不是task,只是一个可await的对象。
#Task是事件循环中真正会调度执行的对象，它本质上是Future的子类
#await 一个 coroutine 会隐式把它包装成 Task，然后等待它完成。
async def serve_http(
    app: FastAPI,
    sock: socket.socket | None,
    enable_ssl_refresh: bool = False,
    **uvicorn_kwargs: Any,
):
    """
    Start a FastAPI app using Uvicorn, with support for custom Uvicorn config
    options.  Supports http header limits via h11_max_incomplete_event_size and
    h11_max_header_count.
    """
    #启动前把当前FastAPI注册的路由打印出来（方便debug和确认是否正确加载）
    logger.info("--------Available routes are:------------")
    for route in app.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)

        if methods is None or path is None:
            continue

        logger.info("Route: %s, Methods: %s", path, ", ".join(methods))

    # Extract header limit options if present
    h11_max_incomplete_event_size = uvicorn_kwargs.pop(
        "h11_max_incomplete_event_size", None
    )
    h11_max_header_count = uvicorn_kwargs.pop("h11_max_header_count", None)

    # Set safe defaults if not provided
    if h11_max_incomplete_event_size is None:
        h11_max_incomplete_event_size = H11_MAX_INCOMPLETE_EVENT_SIZE_DEFAULT
    if h11_max_header_count is None:
        h11_max_header_count = H11_MAX_HEADER_COUNT_DEFAULT

    config = uvicorn.Config(app, **uvicorn_kwargs) #用 FastAPI app + 剩下的所有 uvicorn 参数 创建 uvicorn 的配置对象。
    # Set header limits
    config.h11_max_incomplete_event_size = h11_max_incomplete_event_size
    config.h11_max_header_count = h11_max_header_count
    config.load()
    server = uvicorn.Server(config) #创建真正的unicorn server实例（这才是会真正listen的东西）
    _add_shutdown_handlers(app, server) #给app和server注册一些优雅关闭相关的回调（通常是把一些清理逻辑挂上去）

    loop = asyncio.get_running_loop() #拿到正在运行的事件循环 event loop，不会创建新 loop，只是“借用”当前这个（可能是 uvloop，也可能是标准 asyncio）

    watchdog_task = loop.create_task(watchdog_loop(server, app.state.engine_client))#loop.create_task(...) 把这个协程包装成一个 Task，并立刻调度它到事件循环里去运行。
    server_task = loop.create_task(server.serve(sockets=[sock] if sock else None))

    ssl_cert_refresher = (
        None
        if not enable_ssl_refresh
        else SSLCertRefresher(
            ssl_context=config.ssl,
            key_path=config.ssl_keyfile,
            cert_path=config.ssl_certfile,
            ca_path=config.ssl_ca_certs,
        )
    )

    def signal_handler() -> None:
        # prevents the uvicorn signal handler to exit early
        server_task.cancel()
        watchdog_task.cancel()
        if ssl_cert_refresher:
            ssl_cert_refresher.stop()

    async def dummy_shutdown() -> None:
        pass

    loop.add_signal_handler(signal.SIGINT, signal_handler)
    loop.add_signal_handler(signal.SIGTERM, signal_handler)

    try:
        await server_task
        return dummy_shutdown() #返回一个协程对象
    except asyncio.CancelledError:
        port = uvicorn_kwargs["port"]
        process = find_process_using_port(port)
        if process is not None:
            logger.warning(
                "port %s is used by process %s launched with command:\n%s",
                port,
                process,
                " ".join(process.cmdline()),
            )
        logger.info("Shutting down FastAPI HTTP server.")
        return server.shutdown()
    finally:
        watchdog_task.cancel() #执行try里的代码，遇到return 先不返回 先进入finally块，执行finally里的代码完毕后，才把return的值真正返回给调用者。


async def watchdog_loop(server: uvicorn.Server, engine: EngineClient):
    """
    # Watchdog task that runs in the background, checking
    # for error state in the engine. Needed to trigger shutdown
    # if an exception arises is StreamingResponse() generator.
    """
    VLLM_WATCHDOG_TIME_S = 500.0  #原来是5.0
    while True:
        await asyncio.sleep(VLLM_WATCHDOG_TIME_S) #每隔一段时间醒一次，看看推理引擎（engine）有没有出大事
        terminate_if_errored(server, engine)


def terminate_if_errored(server: uvicorn.Server, engine: EngineClient):
    """
    See discussions here on shutting down a uvicorn server
    https://github.com/encode/uvicorn/discussions/1103
    In this case we cannot await the server shutdown here
    because handler must first return to close the connection
    for this request.
    """
    engine_errored = engine.errored and not engine.is_running #引擎是否发生了未处理的错误 是否已经停止运行
    if not envs.VLLM_KEEP_ALIVE_ON_ENGINE_DEATH and engine_errored:#
        server.should_exit = True


def _add_shutdown_handlers(app: FastAPI, server: uvicorn.Server) -> None:
    """
    VLLM V1 AsyncLLM catches exceptions and returns
    only two types: EngineGenerateError and EngineDeadError.

    EngineGenerateError is raised by the per request generate()
    method. This error could be request specific (and therefore
    recoverable - e.g. if there is an error in input processing).

    EngineDeadError is raised by the background output_handler
    method. This error is global and therefore not recoverable.

    We register these @app.exception_handlers to return nice
    responses to the end user if they occur and shut down if needed.
    See https://fastapi.tiangolo.com/tutorial/handling-errors/
    for more details on how exception handlers work.

    If an exception is encountered in a StreamingResponse
    generator, the exception is not raised, since we already sent
    a 200 status. Rather, we send an error message as the next chunk.
    Since the exception is not raised, this means that the server
    will not automatically shut down. Instead, we use the watchdog
    background task for check for errored state.
    """

    @app.exception_handler(RuntimeError)
    @app.exception_handler(EngineDeadError)
    @app.exception_handler(EngineGenerateError)
    async def runtime_exception_handler(request: Request, __):
        terminate_if_errored(
            server=server,
            engine=request.app.state.engine_client,
        )

        return Response(status_code=HTTPStatus.INTERNAL_SERVER_ERROR)
