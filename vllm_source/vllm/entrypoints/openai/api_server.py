# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import hashlib
import importlib
import inspect
import json
import multiprocessing
import multiprocessing.forkserver as forkserver
import os
import secrets
import signal
import socket
import tempfile
import uuid
from argparse import Namespace
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Annotated, Any

import model_hosting_container_standards.sagemaker as sagemaker_standards
import pydantic
import uvloop
from fastapi import APIRouter, Depends, FastAPI, Form, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import iterate_in_threadpool
from starlette.datastructures import URL, Headers, MutableHeaders, State
from starlette.types import ASGIApp, Message, Receive, Scope, Send

import vllm.envs as envs
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.anthropic.protocol import (
    AnthropicError,
    AnthropicErrorResponse,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
)
from vllm.entrypoints.anthropic.serving_messages import AnthropicServingMessages
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.openai.orca_metrics import metrics_header
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    ErrorInfo,
    ErrorResponse,
    ResponsesRequest,
    ResponsesResponse,
    StreamingResponsesResponse,
    TranscriptionRequest,
    TranscriptionResponseVariant,
    TranslationRequest,
    TranslationResponseVariant,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_engine import OpenAIServing
from vllm.entrypoints.openai.serving_models import (
    BaseModelPath,
    OpenAIServingModels,
)
from vllm.entrypoints.openai.serving_responses import OpenAIServingResponses
from vllm.entrypoints.openai.serving_transcription import (
    OpenAIServingTranscription,
    OpenAIServingTranslation,
)
from vllm.entrypoints.openai.utils import validate_json_request
from vllm.entrypoints.pooling.classify.serving import ServingClassification
from vllm.entrypoints.pooling.embed.serving import OpenAIServingEmbedding
from vllm.entrypoints.pooling.pooling.serving import OpenAIServingPooling
from vllm.entrypoints.pooling.score.serving import ServingScores
from vllm.entrypoints.serve.disagg.serving import ServingTokens
from vllm.entrypoints.serve.elastic_ep.middleware import (
    ScalingMiddleware,
)
from vllm.entrypoints.serve.tokenize.serving import OpenAIServingTokenization
from vllm.entrypoints.tool_server import DemoToolServer, MCPToolServer, ToolServer
from vllm.entrypoints.utils import (
    cli_env_setup,
    load_aware_call,
    log_non_default_args,
    process_chat_template,
    process_lora_modules,
    with_cancellation,
)
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParserManager
from vllm.tasks import POOLING_TASKS
from vllm.tool_parsers import ToolParserManager
from vllm.usage.usage_lib import UsageContext
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.gc_utils import freeze_gc_heap
from vllm.utils.network_utils import is_valid_ipv6_address
from vllm.utils.system_utils import decorate_logs, set_ulimit
from vllm.version import __version__ as VLLM_VERSION

prometheus_multiproc_dir: tempfile.TemporaryDirectory

# Cannot use __name__ (https://github.com/vllm-project/vllm/pull/4765)
logger = init_logger("vllm.entrypoints.openai.api_server") #某些场景下会出现问题，所以硬编码了这个字符

ENDPOINT_LOAD_METRICS_FORMAT_HEADER_LABEL = "endpoint-load-metrics-format"

_running_tasks: set[asyncio.Task] = set() #全局维护一个集合，用来记录所有我们自己创建的、还在后台跑的 asyncio 任务。目的是：服务要关闭时能把这些任务都取消掉，避免“任务泄漏”。


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        if app.state.log_stats:
            engine_client: EngineClient = app.state.engine_client

            async def _force_log():
                while True:
                    await asyncio.sleep(envs.VLLM_LOG_STATS_INTERVAL)
                    await engine_client.do_log_stats()

            task = asyncio.create_task(_force_log()) #创建按异步任务 把协程丢进 event loop 并发执行
            _running_tasks.add(task)
            task.add_done_callback(_running_tasks.remove) #给任务绑定一个回调：任务结束/取消/异常时自动从集合里把自己删掉（非常干净的写法
        else:
            task = None

        # Mark the startup heap as static so that it's ignored by GC.
        # Reduces pause times of oldest generation collections.
        freeze_gc_heap()
        try:
            yield  #yield 控制权交给fastapi,让它真正接收请求
        finally:
            if task is not None:
                task.cancel()
    finally:
        # Ensure app state including engine ref is gc'd
        del app.state


@asynccontextmanager #装饰器使用者可以用 async with build_async_engine_client(...) as engine: 的写法，自动保证“用完就清理”。
async def build_async_engine_client(
    args: Namespace, #Namespace 指的就是 Python 标准库 argparse 里的 argparse.Namespace 类型。
    *,
    usage_context: UsageContext = UsageContext.OPENAI_API_SERVER,
    disable_frontend_multiprocessing: bool | None = None,
    client_config: dict[str, Any] | None = None,
) -> AsyncIterator[EngineClient]:
    """
    上下文管理器：构建并返回一个异步引擎客户端（EngineClient）
    它负责管理引擎客户端的整个生命周期：
    - 在进入上下文时创建并初始化引擎
    - 在退出上下文时（正常结束或异常）自动关闭和清理引擎
    """


    #这一块是性能+稳定性优化，针对Linux多进程启动方式的特殊处理
    #问题：如果直接用fork或spawn，每个 worker 进程启动时都要重新导入 vLLM 的所有模块（torch、transformers、vllm 本身等），非常慢 + 浪费内存。
    #解决方案：用forkserver" 模式 + set_forkserver_preload
    #先创建一个“fork server”守护进程
    #在这个守护进程里提前导入所有重量级模块（这里预加载了 vllm.v1.engine.async_llm）
    #以后创建 worker 时，直接从 forkserver fork 子进程 → 子进程继承已经导入好的模块 → 启动极快、内存开销小
    if os.getenv("VLLM_WORKER_MULTIPROC_METHOD") == "forkserver":
        # The executor is expected to be mp.
        # Pre-import heavy modules in the forkserver process
        #当执行器预期使用multiprocesing时
        #在forkserver进程中预先导入重量级模块
        logger.debug("Setup forkserver with pre-imports")
        multiprocessing.set_start_method("forkserver")
        multiprocessing.set_forkserver_preload(["vllm.v1.engine.async_llm"])
        forkserver.ensure_running()
        logger.debug("Forkserver setup complete!")

    # Context manager to handle engine_client lifecycle
    # Ensures everything is shutdown and cleaned up on error/exit
    #上下文管理器，用于处理engine_client的生命周期
    #确保出错或退出时所有资源都被正确关闭和清理
    engine_args = AsyncEngineArgs.from_cli_args(args)  #把命令行参数 args 转成 AsyncEngineArgs 对象（这是引擎的核心配置类）
    if client_config:
        engine_args._api_process_count = client_config.get("client_count", 1)
        engine_args._api_process_rank = client_config.get("client_index", 0)

    if disable_frontend_multiprocessing is None:
        disable_frontend_multiprocessing = bool(args.disable_frontend_multiprocessing)

    async with build_async_engine_client_from_engine_args(engine_args,usage_context=usage_context,disable_frontend_multiprocessing=disable_frontend_multiprocessing,client_config=client_config,) as engine:
        yield engine


@asynccontextmanager
async def build_async_engine_client_from_engine_args(
    engine_args: AsyncEngineArgs,
    *,#仅限关键字参数分隔符，前边的可以用位置传参 也可以关键字传参
    usage_context: UsageContext = UsageContext.OPENAI_API_SERVER,
    disable_frontend_multiprocessing: bool = False,
    client_config: dict[str, Any] | None = None,
) -> AsyncIterator[EngineClient]:
    """
    Create EngineClient, either:
        - in-process using the AsyncLLMEngine Directly
        - multiprocess using AsyncLLMEngine RPC

    Returns the Client or None if the creation failed.
    """

    # Create the EngineConfig (determines if we can use V1).
    vllm_config = engine_args.create_engine_config(usage_context=usage_context)

    if disable_frontend_multiprocessing:
        logger.warning("V1 is enabled, but got --disable-frontend-multiprocessing.")

    from vllm.v1.engine.async_llm import AsyncLLM

    async_llm: AsyncLLM | None = None #声明变量并初始化为 None

    # Don't mutate the input client_config
    client_config = dict(client_config) if client_config else {}
    client_count = client_config.pop("client_count", 1)
    client_index = client_config.pop("client_index", 0)

    try:
        async_llm = AsyncLLM.from_vllm_config(
            vllm_config=vllm_config,
            usage_context=usage_context,
            enable_log_requests=engine_args.enable_log_requests,
            aggregate_engine_logging=engine_args.aggregate_engine_logging,
            disable_log_stats=engine_args.disable_log_stats,
            client_addresses=client_config,
            client_count=client_count,
            client_index=client_index,
        )

        # Don't keep the dummy data in memory
        assert async_llm is not None
        await async_llm.reset_mm_cache()

        yield async_llm #关键一步：把资源交给使用者 ，yield 本身的值（yield 右边的表达式）会被赋值给 as 后面的变量。
    finally:
        if async_llm:
            async_llm.shutdown()


router = APIRouter()


def base(request: Request) -> OpenAIServing:
    # Reuse the existing instance
    return tokenization(request)


def models(request: Request) -> OpenAIServingModels:
    return request.app.state.openai_serving_models


def responses(request: Request) -> OpenAIServingResponses | None:
    return request.app.state.openai_serving_responses


def messages(request: Request) -> AnthropicServingMessages:
    return request.app.state.anthropic_serving_messages


def chat(request: Request) -> OpenAIServingChat | None:
    return request.app.state.openai_serving_chat


def completion(request: Request) -> OpenAIServingCompletion | None:
    return request.app.state.openai_serving_completion


def tokenization(request: Request) -> OpenAIServingTokenization:
    return request.app.state.openai_serving_tokenization


def transcription(request: Request) -> OpenAIServingTranscription:
    return request.app.state.openai_serving_transcription


def translation(request: Request) -> OpenAIServingTranslation:
    return request.app.state.openai_serving_translation


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


def generate_tokens(request: Request) -> ServingTokens | None:
    return request.app.state.serving_tokens


@router.get("/load")
async def get_server_load_metrics(request: Request):
    # This endpoint returns the current server load metrics.
    # It tracks requests utilizing the GPU from the following routes:
    # - /v1/chat/completions
    # - /v1/completions
    # - /v1/audio/transcriptions
    # - /v1/audio/translations
    # - /v1/embeddings
    # - /pooling
    # - /classify
    # - /score
    # - /v1/score
    # - /rerank
    # - /v1/rerank
    # - /v2/rerank
    return JSONResponse(content={"server_load": request.app.state.server_load_metrics})


@router.get("/v1/models")
async def show_available_models(raw_request: Request):
    handler = models(raw_request)

    models_ = await handler.show_available_models()
    return JSONResponse(content=models_.model_dump())


@router.get("/version")
async def show_version():
    ver = {"version": VLLM_VERSION}
    return JSONResponse(content=ver)


async def _convert_stream_to_sse_events(
    generator: AsyncGenerator[StreamingResponsesResponse, None],
) -> AsyncGenerator[str, None]:
    """Convert the generator to a stream of events in SSE format"""
    async for event in generator:
        event_type = getattr(event, "type", "unknown")
        # https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#event_stream_format
        event_data = (
            f"event: {event_type}\ndata: {event.model_dump_json(indent=None)}\n\n"
        )
        yield event_data


@router.post(
    "/v1/responses",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"content": {"text/event-stream": {}}},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
async def create_responses(request: ResponsesRequest, raw_request: Request):
    handler = responses(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(
            message="The model does not support Responses API"
        )
    try:
        generator = await handler.create_responses(request, raw_request)
    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=str(e)
        ) from e

    if isinstance(generator, ErrorResponse):
        return JSONResponse(
            content=generator.model_dump(), status_code=generator.error.code
        )
    elif isinstance(generator, ResponsesResponse):
        return JSONResponse(content=generator.model_dump())

    return StreamingResponse(
        content=_convert_stream_to_sse_events(generator), media_type="text/event-stream"
    )


@router.get("/v1/responses/{response_id}")
async def retrieve_responses(
    response_id: str,
    raw_request: Request,
    starting_after: int | None = None,
    stream: bool | None = False,
):
    handler = responses(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(
            message="The model does not support Responses API"
        )

    try:
        response = await handler.retrieve_responses(
            response_id,
            starting_after=starting_after,
            stream=stream,
        )
    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=str(e)
        ) from e

    if isinstance(response, ErrorResponse):
        return JSONResponse(
            content=response.model_dump(), status_code=response.error.code
        )
    elif isinstance(response, ResponsesResponse):
        return JSONResponse(content=response.model_dump())
    return StreamingResponse(
        content=_convert_stream_to_sse_events(response), media_type="text/event-stream"
    )


@router.post("/v1/responses/{response_id}/cancel")
async def cancel_responses(response_id: str, raw_request: Request):
    handler = responses(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(
            message="The model does not support Responses API"
        )

    try:
        response = await handler.cancel_responses(response_id)
    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=str(e)
        ) from e

    if isinstance(response, ErrorResponse):
        return JSONResponse(
            content=response.model_dump(), status_code=response.error.code
        )
    return JSONResponse(content=response.model_dump())


@router.post(
    "/v1/messages",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"content": {"text/event-stream": {}}},
        HTTPStatus.BAD_REQUEST.value: {"model": AnthropicErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": AnthropicErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": AnthropicErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_messages(request: AnthropicMessagesRequest, raw_request: Request):
    def translate_error_response(response: ErrorResponse) -> JSONResponse:
        anthropic_error = AnthropicErrorResponse(
            error=AnthropicError(
                type=response.error.type,
                message=response.error.message,
            )
        )
        return JSONResponse(
            status_code=response.error.code, content=anthropic_error.model_dump()
        )

    handler = messages(raw_request)
    if handler is None:
        error = base(raw_request).create_error_response(
            message="The model does not support Messages API"
        )
        return translate_error_response(error)

    try:
        generator = await handler.create_messages(request, raw_request)
    except Exception as e:
        logger.exception("Error in create_messages: %s", e)
        return JSONResponse(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            content=AnthropicErrorResponse(
                error=AnthropicError(
                    type="internal_error",
                    message=str(e),
                )
            ).model_dump(),
        )

    if isinstance(generator, ErrorResponse):
        return translate_error_response(generator)

    elif isinstance(generator, AnthropicMessagesResponse):
        resp = generator.model_dump(exclude_none=True)
        logger.debug("Anthropic Messages Response: %s", resp)
        return JSONResponse(content=resp)

    return StreamingResponse(content=generator, media_type="text/event-stream")


@router.post(
    "/v1/chat/completions",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"content": {"text/event-stream": {}}},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    metrics_header_format = raw_request.headers.get(
        ENDPOINT_LOAD_METRICS_FORMAT_HEADER_LABEL, ""
    )
    handler = chat(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(
            message="The model does not support Chat Completions API"
        )
    try:
        generator = await handler.create_chat_completion(request, raw_request)
    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=str(e)
        ) from e
    if isinstance(generator, ErrorResponse):
        return JSONResponse(
            content=generator.model_dump(), status_code=generator.error.code
        )

    elif isinstance(generator, ChatCompletionResponse):
        return JSONResponse(
            content=generator.model_dump(),
            headers=metrics_header(metrics_header_format),
        )

    return StreamingResponse(content=generator, media_type="text/event-stream")


@router.post(
    "/v1/completions",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.OK.value: {"content": {"text/event-stream": {}}},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.NOT_FOUND.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_completion(request: CompletionRequest, raw_request: Request):
    metrics_header_format = raw_request.headers.get(
        ENDPOINT_LOAD_METRICS_FORMAT_HEADER_LABEL, ""
    )
    handler = completion(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(
            message="The model does not support Completions API"
        )

    try:
        generator = await handler.create_completion(request, raw_request)
    except OverflowError as e:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value, detail=str(e)
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=str(e)
        ) from e

    if isinstance(generator, ErrorResponse):
        return JSONResponse(
            content=generator.model_dump(), status_code=generator.error.code
        )
    elif isinstance(generator, CompletionResponse):
        return JSONResponse(
            content=generator.model_dump(),
            headers=metrics_header(metrics_header_format),
        )

    return StreamingResponse(content=generator, media_type="text/event-stream")


@router.post(
    "/v1/audio/transcriptions",
    responses={
        HTTPStatus.OK.value: {"content": {"text/event-stream": {}}},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.UNPROCESSABLE_ENTITY.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_transcriptions(
    raw_request: Request, request: Annotated[TranscriptionRequest, Form()]
):
    handler = transcription(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(
            message="The model does not support Transcriptions API"
        )

    audio_data = await request.file.read()
    try:
        generator = await handler.create_transcription(audio_data, request, raw_request)
    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=str(e)
        ) from e

    if isinstance(generator, ErrorResponse):
        return JSONResponse(
            content=generator.model_dump(), status_code=generator.error.code
        )

    elif isinstance(generator, TranscriptionResponseVariant):
        return JSONResponse(content=generator.model_dump())

    return StreamingResponse(content=generator, media_type="text/event-stream")


@router.post(
    "/v1/audio/translations",
    responses={
        HTTPStatus.OK.value: {"content": {"text/event-stream": {}}},
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.UNPROCESSABLE_ENTITY.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def create_translations(
    request: Annotated[TranslationRequest, Form()], raw_request: Request
):
    handler = translation(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(
            message="The model does not support Translations API"
        )

    audio_data = await request.file.read()
    try:
        generator = await handler.create_translation(audio_data, request, raw_request)
    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=str(e)
        ) from e

    if isinstance(generator, ErrorResponse):
        return JSONResponse(
            content=generator.model_dump(), status_code=generator.error.code
        )

    elif isinstance(generator, TranslationResponseVariant):
        return JSONResponse(content=generator.model_dump())

    return StreamingResponse(content=generator, media_type="text/event-stream")


def load_log_config(log_config_file: str | None) -> dict | None:
    if not log_config_file:
        return None
    try:
        with open(log_config_file) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(
            "Failed to load log config from file %s: error %s", log_config_file, e
        )
        return None


class AuthenticationMiddleware:
    """
    Pure ASGI middleware that authenticates each request by checking
    if the Authorization Bearer token exists and equals anyof "{api_key}".

    Notes
    -----
    There are two cases in which authentication is skipped:
        1. The HTTP method is OPTIONS.
        2. The request path doesn't start with /v1 (e.g. /health).
    """

    def __init__(self, app: ASGIApp, tokens: list[str]) -> None:
        self.app = app
        self.api_tokens = [hashlib.sha256(t.encode("utf-8")).digest() for t in tokens]

    def verify_token(self, headers: Headers) -> bool:
        authorization_header_value = headers.get("Authorization")
        if not authorization_header_value:
            return False

        scheme, _, param = authorization_header_value.partition(" ")
        if scheme.lower() != "bearer":
            return False

        param_hash = hashlib.sha256(param.encode("utf-8")).digest()

        token_match = False
        for token_hash in self.api_tokens:
            token_match |= secrets.compare_digest(param_hash, token_hash)

        return token_match

    def __call__(self, scope: Scope, receive: Receive, send: Send) -> Awaitable[None]:
        if scope["type"] not in ("http", "websocket") or scope["method"] == "OPTIONS":
            # scope["type"] can be "lifespan" or "startup" for example,
            # in which case we don't need to do anything
            return self.app(scope, receive, send)
        root_path = scope.get("root_path", "")
        url_path = URL(scope=scope).path.removeprefix(root_path)
        headers = Headers(scope=scope)
        # Type narrow to satisfy mypy.
        if url_path.startswith("/v1") and not self.verify_token(headers):
            response = JSONResponse(content={"error": "Unauthorized"}, status_code=401)
            return response(scope, receive, send)
        return self.app(scope, receive, send)


class XRequestIdMiddleware:
    """
    Middleware the set's the X-Request-Id header for each response
    to a random uuid4 (hex) value if the header isn't already
    present in the request, otherwise use the provided request id.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    def __call__(self, scope: Scope, receive: Receive, send: Send) -> Awaitable[None]:
        if scope["type"] not in ("http", "websocket"):
            return self.app(scope, receive, send)

        # Extract the request headers.
        request_headers = Headers(scope=scope)

        async def send_with_request_id(message: Message) -> None:
            """
            Custom send function to mutate the response headers
            and append X-Request-Id to it.
            """
            if message["type"] == "http.response.start":
                response_headers = MutableHeaders(raw=message["headers"])
                request_id = request_headers.get("X-Request-Id", uuid.uuid4().hex)
                response_headers.append("X-Request-Id", request_id)
            await send(message)

        return self.app(scope, receive, send_with_request_id)


def _extract_content_from_chunk(chunk_data: dict) -> str:
    """Extract content from a streaming response chunk."""
    try:
        from vllm.entrypoints.openai.protocol import (
            ChatCompletionStreamResponse,
            CompletionStreamResponse,
        )

        # Try using Completion types for type-safe parsing
        if chunk_data.get("object") == "chat.completion.chunk":
            chat_response = ChatCompletionStreamResponse.model_validate(chunk_data)
            if chat_response.choices and chat_response.choices[0].delta.content:
                return chat_response.choices[0].delta.content
        elif chunk_data.get("object") == "text_completion":
            completion_response = CompletionStreamResponse.model_validate(chunk_data)
            if completion_response.choices and completion_response.choices[0].text:
                return completion_response.choices[0].text
    except pydantic.ValidationError:
        # Fallback to manual parsing
        if "choices" in chunk_data and chunk_data["choices"]:
            choice = chunk_data["choices"][0]
            if "delta" in choice and choice["delta"].get("content"):
                return choice["delta"]["content"]
            elif choice.get("text"):
                return choice["text"]
    return ""


class SSEDecoder:
    """Robust Server-Sent Events decoder for streaming responses."""

    def __init__(self):
        self.buffer = ""
        self.content_buffer = []

    def decode_chunk(self, chunk: bytes) -> list[dict]:
        """Decode a chunk of SSE data and return parsed events."""
        import json

        try:
            chunk_str = chunk.decode("utf-8")
        except UnicodeDecodeError:
            # Skip malformed chunks
            return []

        self.buffer += chunk_str
        events = []

        # Process complete lines
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            line = line.rstrip("\r")  # Handle CRLF

            if line.startswith("data: "):
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    events.append({"type": "done"})
                elif data_str:
                    try:
                        event_data = json.loads(data_str)
                        events.append({"type": "data", "data": event_data})
                    except json.JSONDecodeError:
                        # Skip malformed JSON
                        continue

        return events

    def extract_content(self, event_data: dict) -> str:
        """Extract content from event data."""
        return _extract_content_from_chunk(event_data)

    def add_content(self, content: str) -> None:
        """Add content to the buffer."""
        if content:
            self.content_buffer.append(content)

    def get_complete_content(self) -> str:
        """Get the complete buffered content."""
        return "".join(self.content_buffer)


def _log_streaming_response(response, response_body: list) -> None:
    """Log streaming response with robust SSE parsing."""
    from starlette.concurrency import iterate_in_threadpool

    sse_decoder = SSEDecoder()
    chunk_count = 0

    def buffered_iterator():
        nonlocal chunk_count

        for chunk in response_body:
            chunk_count += 1
            yield chunk

            # Parse SSE events from chunk
            events = sse_decoder.decode_chunk(chunk)

            for event in events:
                if event["type"] == "data":
                    content = sse_decoder.extract_content(event["data"])
                    sse_decoder.add_content(content)
                elif event["type"] == "done":
                    # Log complete content when done
                    full_content = sse_decoder.get_complete_content()
                    if full_content:
                        # Truncate if too long
                        if len(full_content) > 2048:
                            full_content = full_content[:2048] + ""
                            "...[truncated]"
                        logger.info(
                            "response_body={streaming_complete: content=%r, chunks=%d}",
                            full_content,
                            chunk_count,
                        )
                    else:
                        logger.info(
                            "response_body={streaming_complete: no_content, chunks=%d}",
                            chunk_count,
                        )
                    return

    response.body_iterator = iterate_in_threadpool(buffered_iterator())
    logger.info("response_body={streaming_started: chunks=%d}", len(response_body))


def _log_non_streaming_response(response_body: list) -> None:
    """Log non-streaming response."""
    try:
        decoded_body = response_body[0].decode()
        logger.info("response_body={%s}", decoded_body)
    except UnicodeDecodeError:
        logger.info("response_body={<binary_data>}")


def build_app(args: Namespace) -> FastAPI:
    if args.disable_fastapi_docs:
        app = FastAPI(
            openapi_url=None, docs_url=None, redoc_url=None, lifespan=lifespan
        )
    else:
        app = FastAPI(lifespan=lifespan)
    app.state.args = args #把启动参数存入app全局状态，后续所有请求都可以访问

    #注册vLLM主要API路由，把核心API路由注册到FastAPI
    from vllm.entrypoints.serve import register_vllm_serve_api_routers
    register_vllm_serve_api_routers(app)

    #注册SageMaker兼容API
    from vllm.entrypoints.sagemaker.routes import register_sagemaker_routes
    register_sagemaker_routes(router)
    app.include_router(router)

    app.root_path = args.root_path


    #注册pooling api 向量池/embedding polling api
    from vllm.entrypoints.pooling import register_pooling_api_routers
    register_pooling_api_routers(app)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials, #是否允许携带凭证（cookies、HTTP 认证、客户端证书等）。
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(_: Request, exc: HTTPException):
        """
        当任何路由抛出fastapi.HTTPException时，统一把错误转换成 OpenAI 风格的 JSON 错误响应格式，而不是 FastAPI 默认的错误格式。
        """
        err = ErrorResponse(
            error=ErrorInfo(
                message=exc.detail,
                type=HTTPStatus(exc.status_code).phrase,
                code=exc.status_code,
            )
        )
        return JSONResponse(err.model_dump(), status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_: Request, exc: RequestValidationError):
        from vllm.entrypoints.openai.protocol import VLLMValidationError

        param = None
        for error in exc.errors():
            if "ctx" in error and "error" in error["ctx"]:
                ctx_error = error["ctx"]["error"]
                if isinstance(ctx_error, VLLMValidationError):
                    param = ctx_error.parameter
                    break

        exc_str = str(exc)
        errors_str = str(exc.errors())

        if exc.errors() and errors_str and errors_str != exc_str:
            message = f"{exc_str} {errors_str}"
        else:
            message = exc_str

        err = ErrorResponse(
            error=ErrorInfo(
                message=message,
                type=HTTPStatus.BAD_REQUEST.phrase,
                code=HTTPStatus.BAD_REQUEST,
                param=param,
            )
        )
        return JSONResponse(err.model_dump(), status_code=HTTPStatus.BAD_REQUEST)

    # Ensure --api-key option from CLI takes precedence over VLLM_API_KEY
    #明确说明意图：命令行参数 --api-key 的优先级要高于环境变量 VLLM_API_KEY。
    if tokens := [key for key in (args.api_key or [envs.VLLM_API_KEY]) if key]:
        """
        这是一个带 walrus operator（:= 海象运算符） 的 if 语句，同时完成赋值和判断。
        args.api_key：命令行传入的 --api-key 参数，通常是一个列表（支持多个 key，比如 --api-key sk-xxx --api-key sk-yyy）
        如果用户没传 --api-key，args.api_key 是 None 或空列表
        (args.api_key or [envs.VLLM_API_KEY])：
        如果 args.api_key 有值（非空列表），就用它
        如果没有，就 fallback 到 [envs.VLLM_API_KEY]（环境变量的值，可能是单个字符串或 None）
        
        [key for key in ... if key]：过滤掉空字符串、None 等无效值，只保留真正有内容的 key
        tokens := ...：把最终过滤后的有效 key 列表赋值给 tokens，同时作为 if 的条件
        优先级总结：
        用户传了 --api-key sk-aaa sk-bbb → tokens = ["sk-aaa", "sk-bbb"]
        没传 --api-key，但设了环境变量 VLLM_API_KEY=sk-ccc → tokens = ["sk-ccc"]
        两者都没设 → tokens = [] → if 不成立，不加认证
        """
        app.add_middleware(AuthenticationMiddleware, tokens=tokens)

    if args.enable_request_id_headers:
        """
        这个中间件通常会做三件事：
        1）如果请求头中已经有 X-Request-ID，就直接沿用。
        2）如果没有，就生成一个新的唯一 ID（比如 uuid）。
        3）把这个 ID：
        注入到 request 对象中，方便业务代码读取
        写入 response header，返回给客户端
        写入日志上下文，方便统一追踪
        """
        app.add_middleware(XRequestIdMiddleware)

    # Add scaling middleware to check for scaling state
    """
    待看
    给应用加一层“扩缩容状态检测中间件”，在处理请求前判断当前服务是否处于 scaling（扩容 / 缩容）状态，如果是，就采取特殊处理（通常是拒绝请求或返回友好提示），避免流量打到不稳定节点上。
    这个中间件一般会做什么？典型逻辑是：
    1）检查当前进程 / 容器是否处于 scaling 状态，比如：   
    正在启动（warming up）  
    正在下线（draining） 
    正在扩容初始化
    2）如果是，直接返回：
    HTTP 503 Service Unavailable
    或 429 Too Many Requests
    并带上 Retry-After 头，提示客户端稍后重试
    3）如果不是，放行请求，继续进入后续中间件和业务逻辑
    """
    app.add_middleware(ScalingMiddleware)

    if envs.VLLM_DEBUG_LOG_API_SERVER_RESPONSE:
        """
        在调试模式下，给 API Server 加一个中间件，把每个 HTTP 响应的内容完整打进日志，用于排查问题，但因为可能泄露敏感数据，所以强烈不建议在生产环境打开。
        """
        logger.warning(
            "CAUTION: Enabling log response in the API Server. "
            "This can include sensitive information and should be "
            "avoided in production."
        )

        @app.middleware("http")
        async def log_response(request: Request, call_next):
            response = await call_next(request)
            response_body = [section async for section in response.body_iterator]
            response.body_iterator = iterate_in_threadpool(iter(response_body))
            # Check if this is a streaming response by looking at content-type
            content_type = response.headers.get("content-type", "")
            is_streaming = content_type == "text/event-stream; charset=utf-8"

            # Log response body based on type
            if not response_body:
                logger.info("response_body={<empty>}")
            elif is_streaming:
                _log_streaming_response(response, response_body)
            else:
                _log_non_streaming_response(response_body)
            return response

    for middleware in args.middleware:
        module_path, object_name = middleware.rsplit(".", 1)
        imported = getattr(importlib.import_module(module_path), object_name)
        if inspect.isclass(imported):
            app.add_middleware(imported)  # type: ignore[arg-type]
        elif inspect.iscoroutinefunction(imported):
            app.middleware("http")(imported)
        else:
            raise ValueError(
                f"Invalid middleware {middleware}. Must be a function or a class."
            )

    app = sagemaker_standards.bootstrap(app) #把你的 FastAPI / ASGI 应用 app 挂上 SageMaker 官方推荐的一些标准化中间件和配置，使它符合 SageMaker 推理服务的最佳实践

    return app


async def init_app_state(
    engine_client: EngineClient,
    state: State,
    args: Namespace,
) -> None:
    vllm_config = engine_client.vllm_config

    if args.served_model_name is not None:
        served_model_names = args.served_model_name
    else:
        served_model_names = [args.model]

    if args.enable_log_requests:
        request_logger = RequestLogger(max_log_len=args.max_log_len)
    else:
        request_logger = None

    base_model_paths = [
        BaseModelPath(name=name, model_path=args.model) for name in served_model_names
    ]

    state.engine_client = engine_client
    state.log_stats = not args.disable_log_stats
    state.vllm_config = vllm_config
    state.args = args
    supported_tasks = await engine_client.get_supported_tasks()
    logger.info("Supported tasks: %s", supported_tasks)

    resolved_chat_template = await process_chat_template(
        args.chat_template, engine_client, vllm_config.model_config
    )

    #根据启动参数args.tool_server，动态决定是否启用 “工具调用系统（tool calling）”，以及使用哪种工具服务器实现
    if args.tool_server == "demo":
        tool_server: ToolServer | None = DemoToolServer()
        assert isinstance(tool_server, DemoToolServer)
        await tool_server.init_and_validate()
    elif args.tool_server:
        tool_server = MCPToolServer()
        await tool_server.add_tool_server(args.tool_server)
    else:
        tool_server = None

    # Merge default_mm_loras into the static lora_modules

    default_mm_loras = (
        vllm_config.lora_config.default_mm_loras
        if vllm_config.lora_config is not None
        else {}
    )
    lora_modules = process_lora_modules(args.lora_modules, default_mm_loras)

    state.openai_serving_models = OpenAIServingModels(
        engine_client=engine_client,
        base_model_paths=base_model_paths,
        lora_modules=lora_modules,
    )
    await state.openai_serving_models.init_static_loras()
    state.openai_serving_responses = (
        OpenAIServingResponses(
            engine_client,
            state.openai_serving_models,
            request_logger=request_logger,
            chat_template=resolved_chat_template,
            chat_template_content_format=args.chat_template_content_format,
            return_tokens_as_token_ids=args.return_tokens_as_token_ids,
            enable_auto_tools=args.enable_auto_tool_choice,
            tool_parser=args.tool_call_parser,
            tool_server=tool_server,
            reasoning_parser=args.structured_outputs_config.reasoning_parser,
            enable_prompt_tokens_details=args.enable_prompt_tokens_details,
            enable_force_include_usage=args.enable_force_include_usage,
            enable_log_outputs=args.enable_log_outputs,
            log_error_stack=args.log_error_stack,
        )
        if "generate" in supported_tasks
        else None
    )

    #chatgpt说会在@app.post("/v1/chat/completions")路由处会有：return await state.openai_serving_responses.create_chat_completion(...)
    #这是响应的全部业务逻辑
    #为啥用括号包裹啊？
    state.openai_serving_chat = (
        # 如果支持 "generate" 任务，才会创建这个服务对象
        OpenAIServingChat(          # ← 创建 OpenAI 兼容的聊天服务实例
            engine_client,          # 底层推理引擎客户端
            state.openai_serving_models,  # 已加载的模型信息
            args.response_role,         # 回答时默认使用的 role（通常是 "assistant"）
            request_logger=request_logger,  # 请求日志记录器
            chat_template=resolved_chat_template,  # 已解析好的聊天模板（jinja2 或类似）
            chat_template_content_format=args.chat_template_content_format,
            trust_request_chat_template=args.trust_request_chat_template,  # 是否信任客户端传来的模板（安全相关）
            return_tokens_as_token_ids=args.return_tokens_as_token_ids,  # 是否把 token 以 id 形式返回（调试用）
            enable_auto_tools=args.enable_auto_tool_choice,  # 是否自动开启 tool calling
            exclude_tools_when_tool_choice_none=...,  # 当 tool_choice="none" 时是否隐藏 tools
            tool_parser=args.tool_call_parser,  # 工具调用结果的解析器
            reasoning_parser=...,  # 结构化输出/推理标签解析器
            enable_prompt_tokens_details=...,  # 是否返回详细的 prompt token 信息
            enable_force_include_usage=...,  # 强制包含 usage 字段
            enable_log_outputs=...,  # 是否记录模型输出
            log_error_stack=...,  # 出错时是否记录完整堆栈
        )
        # 关键条件判断：只有当支持 "generate" 任务时才创建，否则赋值为 None
        if "generate" in supported_tasks
        else None
    )
    # Warm up chat template processing to avoid first-request latency
    if state.openai_serving_chat is not None:
        await state.openai_serving_chat.warmup()
    state.openai_serving_completion = (
        OpenAIServingCompletion(
            engine_client,
            state.openai_serving_models,
            request_logger=request_logger,
            return_tokens_as_token_ids=args.return_tokens_as_token_ids,
            enable_prompt_tokens_details=args.enable_prompt_tokens_details,
            enable_force_include_usage=args.enable_force_include_usage,
            log_error_stack=args.log_error_stack,
        )
        if "generate" in supported_tasks
        else None
    )
    state.openai_serving_pooling = (
        (
            OpenAIServingPooling(
                engine_client,
                state.openai_serving_models,
                supported_tasks=supported_tasks,
                request_logger=request_logger,
                chat_template=resolved_chat_template,
                chat_template_content_format=args.chat_template_content_format,
                trust_request_chat_template=args.trust_request_chat_template,
                log_error_stack=args.log_error_stack,
            )
        )
        if any(task in POOLING_TASKS for task in supported_tasks)
        else None
    )
    state.openai_serving_embedding = (
        OpenAIServingEmbedding(
            engine_client,
            state.openai_serving_models,
            request_logger=request_logger,
            chat_template=resolved_chat_template,
            chat_template_content_format=args.chat_template_content_format,
            trust_request_chat_template=args.trust_request_chat_template,
            log_error_stack=args.log_error_stack,
        )
        if "embed" in supported_tasks
        else None
    )
    state.openai_serving_classification = (
        ServingClassification(
            engine_client,
            state.openai_serving_models,
            request_logger=request_logger,
            chat_template=resolved_chat_template,
            chat_template_content_format=args.chat_template_content_format,
            trust_request_chat_template=args.trust_request_chat_template,
            log_error_stack=args.log_error_stack,
        )
        if "classify" in supported_tasks
        else None
    )
    state.openai_serving_scores = (
        ServingScores(
            engine_client,
            state.openai_serving_models,
            request_logger=request_logger,
            score_template=resolved_chat_template,
            log_error_stack=args.log_error_stack,
        )
        if ("embed" in supported_tasks or "score" in supported_tasks)
        else None
    )
    state.openai_serving_tokenization = OpenAIServingTokenization(
        engine_client,
        state.openai_serving_models,
        request_logger=request_logger,
        chat_template=resolved_chat_template,
        chat_template_content_format=args.chat_template_content_format,
        trust_request_chat_template=args.trust_request_chat_template,
        log_error_stack=args.log_error_stack,
    )
    state.openai_serving_transcription = (
        OpenAIServingTranscription(
            engine_client,
            state.openai_serving_models,
            request_logger=request_logger,
            log_error_stack=args.log_error_stack,
            enable_force_include_usage=args.enable_force_include_usage,
        )
        if "transcription" in supported_tasks
        else None
    )
    state.openai_serving_translation = (
        OpenAIServingTranslation(
            engine_client,
            state.openai_serving_models,
            request_logger=request_logger,
            log_error_stack=args.log_error_stack,
            enable_force_include_usage=args.enable_force_include_usage,
        )
        if "transcription" in supported_tasks
        else None
    )
    state.anthropic_serving_messages = (
        AnthropicServingMessages(
            engine_client,
            state.openai_serving_models,
            args.response_role,
            request_logger=request_logger,
            chat_template=resolved_chat_template,
            chat_template_content_format=args.chat_template_content_format,
            return_tokens_as_token_ids=args.return_tokens_as_token_ids,
            enable_auto_tools=args.enable_auto_tool_choice,
            tool_parser=args.tool_call_parser,
            reasoning_parser=args.structured_outputs_config.reasoning_parser,
            enable_prompt_tokens_details=args.enable_prompt_tokens_details,
            enable_force_include_usage=args.enable_force_include_usage,
        )
        if "generate" in supported_tasks
        else None
    )
    state.serving_tokens = (
        ServingTokens(
            engine_client,
            state.openai_serving_models,
            request_logger=request_logger,
            return_tokens_as_token_ids=args.return_tokens_as_token_ids,
            log_error_stack=args.log_error_stack,
            enable_prompt_tokens_details=args.enable_prompt_tokens_details,
            enable_log_outputs=args.enable_log_outputs,
            force_no_detokenize=args.tokens_only,
        )
        if "generate" in supported_tasks
        else None
    )

    state.enable_server_load_tracking = args.enable_server_load_tracking
    state.server_load_metrics = 0


def create_server_socket(addr: tuple[str, int]) -> socket.socket: #输出是一个socket对象
    family = socket.AF_INET  #默认是IPV4协议
    if is_valid_ipv6_address(addr[0]): #如果IP是IPV6 那么就切换成IPv6协议
        family = socket.AF_INET6

    sock = socket.socket(family=family, type=socket.SOCK_STREAM) #创建一个真正的socket对象，type=socket.SOCK_STREAM->TCP链接
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) #防止端口占用导致重启失败，简单说如果立即关掉服务器重启，address already in use就是因为端口还没被释放，这行代码告诉系统，我这个端口可以马上复用，不需要等
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1) #允许多个进程、线程绑定同一个端口
    sock.bind(addr)

    return sock


def create_server_unix_socket(path: str) -> socket.socket:
    sock = socket.socket(family=socket.AF_UNIX, type=socket.SOCK_STREAM)
    sock.bind(path)
    return sock


def validate_api_server_args(args):
    """
    参数校验函数
    检查用户传入的 两个和“工具调用/结构化输出”相关的参数是否合法，如果不合法就立刻抛异常，让程序停止启动，避免后面运行时才崩溃。
    当你启动 vLLM 的 OpenAI 兼容 API 服务器时（比如 vllm serve 或 python -m vllm.entrypoints.openai.api_server），会传入一大堆参数（--enable-auto-tool-choice、--tool-call-parser 等）。
    其中有两个参数和“模型如何处理工具调用（tool calling）/结构化输出（structured outputs）”有关：

    --enable-auto-tool-choice + --tool-call-parser
    --reasoning-parser（通常藏在 structured_outputs_config 里）

    这段代码的任务就是：确认这两个配置选的值是 vLLM 目前真的支持的，否则直接报错。
    """
    valid_tool_parses = ToolParserManager.list_registered()
    if args.enable_auto_tool_choice and args.tool_call_parser not in valid_tool_parses:
        raise KeyError(
            f"invalid tool call parser: {args.tool_call_parser} "
            f"(chose from {{ {','.join(valid_tool_parses)} }})"
        )

    valid_reasoning_parsers = ReasoningParserManager.list_registered()
    if (
        reasoning_parser := args.structured_outputs_config.reasoning_parser
    ) and reasoning_parser not in valid_reasoning_parsers:
        raise KeyError(
            f"invalid reasoning parser: {reasoning_parser} "
            f"(chose from {{ {','.join(valid_reasoning_parsers)} }})"
        )


def setup_server(args):
    """Validate API server args, set up signal handler, create socket
    ready to serve."""

    logger.info("vLLM API server version %s", VLLM_VERSION)
    log_non_default_args(args)

    if args.tool_parser_plugin and len(args.tool_parser_plugin) > 3:
        ToolParserManager.import_tool_parser(args.tool_parser_plugin)

    if args.reasoning_parser_plugin and len(args.reasoning_parser_plugin) > 3:
        ReasoningParserManager.import_reasoning_parser(args.reasoning_parser_plugin)

    validate_api_server_args(args)

    # workaround to make sure that we bind the port before the engine is set up.
    # This avoids race conditions with ray.
    # see https://github.com/vllm-project/vllm/issues/8204
    if args.uds:
        sock = create_server_unix_socket(args.uds)
    else:
        sock_addr = (args.host or "", args.port)
        sock = create_server_socket(sock_addr) #这句话的意思是：我要创建一个服务器用的网络插座，并把它交给sock这个变量。

    # workaround to avoid footguns where uvicorn drops requests with too
    # many concurrent requests active
    set_ulimit()#把系统允许"同时打开的文件数量上限"调大

    def signal_handler(*_) -> None:
        """
        信号处理器函数：参数 *_ 表示忽略信号编号和栈帧（我们不关心具体是哪个信号或从哪来的）
        什么都不做，直接raise
        KeyboardInterrupt 是 Python 收到 Ctrl+C 时默认抛的异常
        """
        # Interrupt server on sigterm while initializing
        raise KeyboardInterrupt("terminated")
    #SIGTERM 是 Linux/Unix 系统中的一种进程信号（signal），全称是 SIGnal TERMinate，信号编号为 15（常用 kill -15 或直接 kill 命令默认发送的就是它）。
    #把 SIGTERM 信号绑定到上面这个 handler SIGTERM 是“温柔终止”信号（kill -15），生产环境最常用（Kubernetes pod 终止、systemd stop、重启等都会发这个）
    #从这一行开始，只要进程收到 SIGTERM，就会执行 raise KeyboardInterrupt("terminated")
    signal.signal(signal.SIGTERM, signal_handler) #在服务器初始化阶段（模型还没加载完、引擎还没准备好）如果收到 SIGTERM 信号，就立刻抛出 KeyboardInterrupt 异常，让整个进程快速退出，而不是继续卡住或半死不活。

    if args.uds:
        listen_address = f"unix:{args.uds}"
    else:
        addr, port = sock_addr
        is_ssl = args.ssl_keyfile and args.ssl_certfile
        host_part = f"[{addr}]" if is_valid_ipv6_address(addr) else addr or "0.0.0.0"
        listen_address = f"http{'s' if is_ssl else ''}://{host_part}:{port}"
    return listen_address, sock


async def run_server(args, **uvicorn_kwargs) -> None:
    """Run a single-worker API server."""

    # Add process-specific prefix to stdout and stderr.
    decorate_logs("APIServer") #这是一个 vLLM 内部工具函数

    listen_address, sock = setup_server(args)
    await run_server_worker(listen_address, sock, args, **uvicorn_kwargs)


async def run_server_worker(
    listen_address, sock, args, client_config=None, **uvicorn_kwargs
) -> None:
    """Run a single API server worker."""

    if args.tool_parser_plugin and len(args.tool_parser_plugin) > 3:
        ToolParserManager.import_tool_parser(args.tool_parser_plugin)

    if args.reasoning_parser_plugin and len(args.reasoning_parser_plugin) > 3:
        ReasoningParserManager.import_reasoning_parser(args.reasoning_parser_plugin)

    # Load logging config for uvicorn if specified
    log_config = load_log_config(args.log_config_file) #uvicorn日志配置
    if log_config is not None:
        uvicorn_kwargs["log_config"] = log_config

    async with build_async_engine_client(args,client_config=client_config,) as engine_client:
        #整个服务的核心引擎初始化，可以理解为：启动 LLM 引擎进程 + 建立 RPC + 建立 GPU 通信 + 初始化调度器
        app = build_app(args)

        await init_app_state(engine_client, app.state, args) #初始化app运行状态
        #%d → 占位符，表示整数  %s → 占位符，表示字符串
        logger.info(
            "Starting vLLM API server %d on %s",
            engine_client.vllm_config.parallel_config._api_process_rank,
            listen_address,
        )
        #真正启动HTTP服务（服务开始对外提供能力），为什么返回shutdown_task，因为serve_http不是阻塞运行，而是启动后台任务
        #shutdown_task表示等待服务器关闭的future对象
        shutdown_task = await serve_http(
            app,
            sock=sock,
            enable_ssl_refresh=args.enable_ssl_refresh,
            host=args.host,
            port=args.port,
            log_level=args.uvicorn_log_level,
            # NOTE: When the 'disable_uvicorn_access_log' value is True,
            # no access log will be output.
            access_log=not args.disable_uvicorn_access_log,
            timeout_keep_alive=envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            ssl_ca_certs=args.ssl_ca_certs,
            ssl_cert_reqs=args.ssl_cert_reqs,
            h11_max_incomplete_event_size=args.h11_max_incomplete_event_size,
            h11_max_header_count=args.h11_max_header_count,
            **uvicorn_kwargs,
        )#serve_http 内部的 server.serve() 是一个会一直运行到服务器关闭才返回的协程，如果你直接 await 它，整个函数就会卡在这里，永远到不了后面的代码。

    # NB: Await server shutdown only after the backend context is exited
    try:
        await shutdown_task
    finally:
        sock.close()


if __name__ == "__main__":
    # NOTE(simon):
    # This section should be in sync with vllm/entrypoints/cli/main.py for CLI
    # entrypoints.
    cli_env_setup() #设置多进程启动方式为spawn
    parser = FlexibleArgumentParser(description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args() #里才是参数真正被解析并更新进 args 的时刻 ,parse_args()（无参数调用）会自动读取 sys.argv[1:]（即命令行里 --model ... 后面的所有部分）。
    validate_parsed_serve_args(args)
    #
    uvloop.run(run_server(args))
