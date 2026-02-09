# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This file contains the command line arguments for the vLLM's
OpenAI-compatible server. It is kept in a separate file for documentation
purposes.
"""

import argparse
import json
import ssl
from collections.abc import Sequence
from dataclasses import field
from typing import Literal

from pydantic.dataclasses import dataclass

import vllm.envs as envs
from vllm.config import config
from vllm.engine.arg_utils import AsyncEngineArgs, optional_type
from vllm.entrypoints.chat_utils import (
    ChatTemplateContentFormatOption,
    validate_chat_template,
)
from vllm.entrypoints.constants import (
    H11_MAX_HEADER_COUNT_DEFAULT,
    H11_MAX_INCOMPLETE_EVENT_SIZE_DEFAULT,
)
from vllm.entrypoints.openai.serving_models import LoRAModulePath
from vllm.logger import init_logger
from vllm.tool_parsers import ToolParserManager
from vllm.utils.argparse_utils import FlexibleArgumentParser

logger = init_logger(__name__)


class LoRAParserAction(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[str] | None,
        option_string: str | None = None,
    ):
        if values is None:
            values = []
        if isinstance(values, str):
            raise TypeError("Expected values to be a list")

        lora_list: list[LoRAModulePath] = []
        for item in values:
            if item in [None, ""]:  # Skip if item is None or empty string
                continue
            if "=" in item and "," not in item:  # Old format: name=path
                name, path = item.split("=")
                lora_list.append(LoRAModulePath(name, path))
            else:  # Assume JSON format
                try:
                    lora_dict = json.loads(item)
                    lora = LoRAModulePath(**lora_dict)
                    lora_list.append(lora)
                except json.JSONDecodeError:
                    parser.error(f"Invalid JSON format for --lora-modules: {item}")
                except TypeError as e:
                    parser.error(
                        f"Invalid fields for --lora-modules: {item} - {str(e)}"
                    )
        setattr(namespace, self.dest, lora_list)


@config
@dataclass
class FrontendArgs:
    """Arguments for the OpenAI-compatible frontend server."""

    host: str | None = None
    """Host name."""
    port: int = 8000
    """Port number."""
    uds: str | None = None
    """Unix domain socket path. If set, host and port arguments are ignored.不监听tcp端口 而监听本地unix socket文件"""
    uvicorn_log_level: Literal[
        "debug", "info", "warning", "error", "critical", "trace"
    ] = "info"
    """Log level for uvicorn."""
    disable_uvicorn_access_log: bool = False
    """Disable uvicorn access log."""
    allow_credentials: bool = False
    """Allow credentials."""
    allowed_origins: list[str] = field(default_factory=lambda: ["*"])
    """Allowed origins."""
    allowed_methods: list[str] = field(default_factory=lambda: ["*"])
    """Allowed methods."""
    allowed_headers: list[str] = field(default_factory=lambda: ["*"])
    """Allowed headers."""
    api_key: list[str] | None = None
    """If provided, the server will require one of these keys to be presented in
    the header."""
    lora_modules: list[LoRAModulePath] | None = None
    """LoRA modules configurations in either 'name=path' format or JSON format
    or JSON list format. Example (old format): `'name=path'` Example (new
    format): `{\"name\": \"name\", \"path\": \"lora_path\",
    \"base_model_name\": \"id\"}`""" #指定lora模型的，不用合并权重了啊
    chat_template: str | None = None
    """The file path to the chat template, or the template in single-line form
    for the specified model."""
    chat_template_content_format: ChatTemplateContentFormatOption = "auto"
    """The format to render message content within a chat template.

    * "string" will render the content as a string. Example: `"Hello World"`
    * "openai" will render the content as a list of dictionaries, similar to
      OpenAI schema. Example: `[{"type": "text", "text": "Hello world!"}]`"""
    trust_request_chat_template: bool = False
    """Whether to trust the chat template provided in the request. If False,
    the server will always use the chat template specified by `--chat-template`
    or the ones from tokenizer."""
    response_role: str = "assistant"
    """The role name to return if `request.add_generation_prompt=true`."""
    ssl_keyfile: str | None = None
    """The file path to the SSL key file."""
    ssl_certfile: str | None = None
    """The file path to the SSL cert file."""
    ssl_ca_certs: str | None = None
    """The CA certificates file."""
    enable_ssl_refresh: bool = False
    """Refresh SSL Context when SSL certificate files change"""
    ssl_cert_reqs: int = int(ssl.CERT_NONE)
    """Whether client certificate is required (see stdlib ssl module's)."""
    root_path: str | None = None
    """FastAPI root_path when app is behind a path based routing proxy."""
    middleware: list[str] = field(default_factory=lambda: [])
    """Additional ASGI middleware to apply to the app. We accept multiple
    --middleware arguments. The value should be an import path. If a function
    is provided, vLLM will add it to the server using
    `@app.middleware('http')`. If a class is provided, vLLM will
    add it to the server using `app.add_middleware()`."""
    return_tokens_as_token_ids: bool = False
    """When `--max-logprobs` is specified, represents single tokens as
    strings of the form 'token_id:{token_id}' so that tokens that are not
    JSON-encodable can be identified."""
    disable_frontend_multiprocessing: bool = False #强制把 OpenAI 前端服务器（FastAPI 部分）和模型推理引擎（LLMEngine）跑在同一个进程里，而不是默认的多进程分离。
    """If specified, will run the OpenAI frontend server in the same process as #
    the model serving engine.""" #调试更容易（单进程，栈追踪清晰，不用跨进程追 bug）些环境（如某些容器、调试工具、或特殊硬件）多进程有兼容问题
    enable_request_id_headers: bool = False
    """If specified, API server will add X-Request-Id header to responses."""
    enable_auto_tool_choice: bool = False #自动工具选择（Auto Tool Choice）功能，让模型在 tool_choice="auto" 时自主决定是否调用工具，
    """Enable auto tool choice for supported models. Use `--tool-call-parser`
    to specify which parser to use.""" #vLLM 支持 tool_choice 的值：auto（模型自己决定）、none（强制不调用）、required（强制必须调用）、或指定某个工具名。必须搭配--tool-call-parser 使用，否则启动会报错。
    exclude_tools_when_tool_choice_none: bool = False
    """If specified, exclude tool definitions in prompts when
    tool_choice='none'."""
    tool_call_parser: str | None = None #指定哪个解析器来把模型生成的原始工具调用输出（通常是 JSON 或 XML 格式的文本）转换成 OpenAI 兼容的 tool_calls 结构（list of dicts）。
    """Select the tool call parser depending on the model that you're using.
    This is used to parse the model-generated tool call into OpenAI API format.
    Required for `--enable-auto-tool-choice`. You can choose any option from
    the built-in parsers or register a plugin via `--tool-parser-plugin`."""
    tool_parser_plugin: str = "" #定义工具解析器插件的路径或名称，让你能注册自己的 parser。
    """Special the tool parser plugin write to parse the model-generated tool
    into OpenAI API format, the name register in this plugin can be used in
    `--tool-call-parser`."""
    tool_server: str | None = None #外部工具服务器的地址列表（逗号分隔的 host:port），vLLM 会把工具调用转发到这些服务器执行
    """Comma-separated list of host:port pairs (IPv4, IPv6, or hostname).
    Examples: 127.0.0.1:8000, [::1]:8000, localhost:1234. Or `demo` for demo
    purpose."""
    log_config_file: str | None = envs.VLLM_LOGGING_CONFIG_PATH
    """Path to logging config JSON file for both vllm and uvicorn"""
    max_log_len: int | None = None  #限制日志长度 避免日志文件爆炸
    """Max number of prompt characters or prompt ID numbers being printed in
    log. The default of None means unlimited."""
    disable_fastapi_docs: bool = False #禁用 FastAPI 自带的 OpenAPI 文档
    """Disable FastAPI's OpenAPI schema, Swagger UI, and ReDoc endpoint."""
    enable_prompt_tokens_details: bool = False #在响应的usage字段中，额外返回prompt_tokens_details  适用场景：精细计费、分析 prompt 结构。
    """If set to True, enable prompt_tokens_details in usage."""
    enable_server_load_tracking: bool = False #app.state 里启用 server_load_metrics 跟踪（负载指标，如队列长度、等待时间等）。适用场景：内部监控、Prometheus 采集、动态扩缩容。
    """If set to True, enable tracking server_load_metrics in the app state."""
    enable_force_include_usage: bool = False #强制每个响应都带上 usage 字段（token 消耗统计），即使客户端没请求。适用场景：统一计费系统、强制记录所有请求消耗。
    """If set to True, including usage on every request."""
    enable_tokenizer_info_endpoint: bool = False #开启 /tokenizer_info 接口，返回 tokenizer 配置（包括 chat template、特殊 token 等）。
    """Enable the `/tokenizer_info` endpoint. May expose chat
    templates and other tokenizer configuration."""
    enable_log_outputs: bool = False #是否在日志里打印模型生成的输出（generation）。必须同时开启 --enable-log-requests 才生效。
    """If True, log model outputs (generations).
    Requires --enable-log-requests."""
    h11_max_incomplete_event_size: int = H11_MAX_INCOMPLETE_EVENT_SIZE_DEFAULT
    """Maximum size (bytes) of an incomplete HTTP event (header or body) for
    h11 parser. Helps mitigate header abuse. Default: 4194304 (4 MB)."""
    h11_max_header_count: int = H11_MAX_HEADER_COUNT_DEFAULT #单个请求允许的最大 HTTP header 数量
    """Maximum number of HTTP headers allowed in a request for h11 parser.
    Helps mitigate header abuse. Default: 256."""
    log_error_stack: bool = envs.VLLM_SERVER_DEV_MODE
    """If set to True, log the stack trace of error responses"""
    tokens_only: bool = False #只启用 Tokens In/Out 端点（极简模式），关闭其他所有 OpenAI 兼容接口。
    """
    If set to True, only enable the Tokens In<>Out endpoint. 
    This is intended for use in a Disaggregated Everything setup.
    """

    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
        from vllm.engine.arg_utils import get_kwargs

        frontend_kwargs = get_kwargs(FrontendArgs)

        # Special case: allowed_origins, allowed_methods, allowed_headers all
        # need json.loads type
        # Should also remove nargs
        frontend_kwargs["allowed_origins"]["type"] = json.loads
        frontend_kwargs["allowed_methods"]["type"] = json.loads
        frontend_kwargs["allowed_headers"]["type"] = json.loads
        del frontend_kwargs["allowed_origins"]["nargs"]
        del frontend_kwargs["allowed_methods"]["nargs"]
        del frontend_kwargs["allowed_headers"]["nargs"]

        # Special case: LoRA modules need custom parser action and
        # optional_type(str)
        frontend_kwargs["lora_modules"]["type"] = optional_type(str)
        frontend_kwargs["lora_modules"]["action"] = LoRAParserAction

        # Special case: Middleware needs to append action
        frontend_kwargs["middleware"]["action"] = "append"
        frontend_kwargs["middleware"]["type"] = str
        if "nargs" in frontend_kwargs["middleware"]:
            del frontend_kwargs["middleware"]["nargs"]
        frontend_kwargs["middleware"]["default"] = []

        # Special case: Tool call parser shows built-in options.
        valid_tool_parsers = list(ToolParserManager.list_registered())
        parsers_str = ",".join(valid_tool_parsers)
        frontend_kwargs["tool_call_parser"]["metavar"] = (
            f"{{{parsers_str}}} or name registered in --tool-parser-plugin"
        )

        frontend_group = parser.add_argument_group(
            title="Frontend",
            description=FrontendArgs.__doc__,
        )

        for key, value in frontend_kwargs.items():
            frontend_group.add_argument(f"--{key.replace('_', '-')}", **value)

        return parser


def make_arg_parser(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
    """Create the CLI argument parser used by the OpenAI API server.

    We rely on the helper methods of `FrontendArgs` and `AsyncEngineArgs` to
    register all arguments instead of manually enumerating them here. This
    avoids code duplication and keeps the argument definitions in one place.
    """
    parser.add_argument(
        "model_tag",
        type=str,
        nargs="?",
        help="The model tag to serve (optional if specified in config)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run in headless mode. See multi-node data parallel "
        "documentation for more details.",
    )
    parser.add_argument(
        "--api-server-count",
        "-asc",
        type=int,
        default=1,
        help="How many API server processes to run.",
    )
    parser.add_argument(
        "--config",
        help="Read CLI options from a config file. "
        "Must be a YAML with the following options: "
        "https://docs.vllm.ai/en/latest/configuration/serve_args.html",
    )
    parser = FrontendArgs.add_cli_args(parser)
    parser = AsyncEngineArgs.add_cli_args(parser)

    return parser


def validate_parsed_serve_args(args: argparse.Namespace):
    """Quick checks for model serve args that raise prior to loading."""
    if hasattr(args, "subparser") and args.subparser != "serve":
        return

    # Ensure that the chat template is valid; raises if it likely isn't
    validate_chat_template(args.chat_template)

    # Enable auto tool needs a tool call parser to be valid
    if args.enable_auto_tool_choice and not args.tool_call_parser: #如果你开了自动工具调用（--enable-auto-tool-choice），就必须同时指定工具调用解析器（--tool-call-parser）。
        raise TypeError("Error: --enable-auto-tool-choice requires --tool-call-parser")
    if args.enable_log_outputs and not args.enable_log_requests:#如果你想记录模型的输出内容（--enable-log-outputs），就必须先开启请求日志（--enable-log-requests）。
        raise TypeError("Error: --enable-log-outputs requires --enable-log-requests")


def create_parser_for_docs() -> FlexibleArgumentParser:
    parser_for_docs = FlexibleArgumentParser(
        prog="-m vllm.entrypoints.openai.api_server"
    )
    return make_arg_parser(parser_for_docs)
