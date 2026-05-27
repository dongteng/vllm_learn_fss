# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import os
from abc import abstractmethod
from collections.abc import Callable, Sequence
from functools import cached_property
from typing import TYPE_CHECKING, Any

from vllm.entrypoints.tool_server import ToolServer
from vllm.logger import init_logger
from vllm.utils.collection_utils import is_list_of
from vllm.utils.import_utils import import_from_path

if TYPE_CHECKING:
    from vllm.entrypoints.openai.protocol import (
        ChatCompletionRequest,
        DeltaMessage,
        ResponsesRequest,
    )
    from vllm.tokenizers import TokenizerLike
else:
    ChatCompletionRequest = Any
    DeltaMessage = Any
    ResponsesRequest = Any
    TokenizerLike = Any

logger = init_logger(__name__)


class ReasoningParser:
    """
    Abstract reasoning parser class that should not be used directly.                                               抽象的推理解析器类,不应直接用
    Provided and methods should be used in derived classes.                                                         应该在其派生类中实现并使用相关方法

    It is used to extract reasoning content from the model output.                                                  用于从模型输出中提取推理内容
    """

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        self.model_tokenizer = tokenizer

    @cached_property
    def vocab(self) -> dict[str, int]:
        # NOTE: Only PreTrainedTokenizerFast is guaranteed to have .vocab
        # whereas all tokenizers have .get_vocab()
        return self.model_tokenizer.get_vocab()

    @abstractmethod
    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        """
        Check if the reasoning content ends in the input_ids.                                                       判断input_ids中推理内容是否已经结束

        It is used in structured engines like `xgrammar` to check if the                                            该方法用于结构化引擎(如xgrammar)中,用于检测模型输出中的推理部分是否结束
        reasoning content ends in the model output.

        Parameters:
        input_ids: list[int]
            The input_ids of the model output.

        Returns:
        bool
            True if the reasoning content ends in the input_ids.
        """

    def is_reasoning_end_streaming(
        self, input_ids: list[int], delta_ids: list[int]
    ) -> bool:
        """
        Check if the reasoning content ends in the input_ids on a
        decode step.

        It is used in structured engines like `xgrammar` to check if the
        reasoning content ends in the model output during a decode step.
        `input_ids` the entire model output and `delta_ids` are the last few
        computed tokens of the model output (like during a decode step).

        Parameters:
        input_ids: list[int]
            The entire model output.
        delta_ids: list[int]
            The last few computed tokens of the model output at the current decode step.

        Returns:
        bool
            True if the reasoning content ends in the `delta_ids` on a
            decode step.
        """
        return self.is_reasoning_end(input_ids)

    @abstractmethod
    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """
        Extract content token ids from the input_ids.                                                                  从input_ids中提取内容token的id
        Parameters:
        input_ids: list[int]
            The input_ids of the model output.
        Returns:
        list[int]
            The extracted content from the input_ids.
        """

    @abstractmethod
    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning content from a complete model-generated string.                                               从完整的模型生成字符串中提取推理内容(reasoning)

        Used for non-streaming responses where we have the entire model response                                        该方法用于非流失响应场景,即再将结果发送给客户端之前,已经拿到了完整的模型输出
        available before sending to the client.

        Parameters:
        model_output: str
            The model-generated string to extract reasoning content from.

        request: ChatCompletionRequest
            The request object that was used to generate the model_output.

        Returns:
        tuple[Optional[str], Optional[str]]
            A tuple containing the reasoning content and the content.
        """

    @abstractmethod
    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        """
        Instance method that should be implemented for extracting reasoning                                             从不完整响应中提取reasoning 该方法用于处理reasoning调用以及流失输出场景
        from an incomplete response; for use when handling reasoning calls and                                          之所以必须设计为实例方法,是因为它需要维护状态  不仅需要当前的token/diff信息
        streaming. Has to be an instance method because  it requires state -                                            还需要知道此前已经解析和提取过哪些内容(见构造函数)
        the current tokens/diffs, but also the information about what has
        previously been parsed and extracted (see constructor)
        """

    def prepare_structured_tag(
        self,
        original_tag: str | None,
        tool_server: ToolServer | None,
    ) -> str | None:
        """
        Instance method that is implemented for preparing the structured tag                                            用于准备结构化标签(structred tag)的实例方法
        Otherwise, None is returned
        """
        return None


class ReasoningParserManager:
    """
    Central registry for ReasoningParser implementations.

    Supports two registration modes:
      - Eager registration via `register_module`
      - Lazy registration via `register_lazy_module`

    Each reasoning parser must inherit from `ReasoningParser`.
    """

    reasoning_parsers: dict[str, type[ReasoningParser]] = {}
    lazy_parsers: dict[str, tuple[str, str]] = {}  # name -> (module_path, class_name)

    @classmethod
    def get_reasoning_parser(cls, name: str) -> type[ReasoningParser]:
        """
        Retrieve a registered or lazily registered ReasoningParser class.

        If the parser is lazily registered, it will be imported and cached
        on first access.

        Raises:
            KeyError: if no parser is found under the given name.
        """
        if name in cls.reasoning_parsers:
            return cls.reasoning_parsers[name]

        if name in cls.lazy_parsers:
            return cls._load_lazy_parser(name)

        registered = ", ".join(cls.list_registered())
        raise KeyError(
            f"Reasoning parser '{name}' not found. Available parsers: {registered}"
        )

    @classmethod
    def list_registered(cls) -> list[str]:
        """Return names of all eagerly and lazily registered reasoning parsers."""
        return sorted(set(cls.reasoning_parsers.keys()) | set(cls.lazy_parsers.keys()))

    @classmethod
    def _load_lazy_parser(cls, name: str) -> type[ReasoningParser]:
        """Import and register a lazily loaded reasoning parser."""
        module_path, class_name = cls.lazy_parsers[name]
        try:
            mod = importlib.import_module(module_path)
            parser_cls = getattr(mod, class_name)
            if not issubclass(parser_cls, ReasoningParser):
                raise TypeError(
                    f"{class_name} in {module_path} is not a ReasoningParser subclass."
                )

            cls.reasoning_parsers[name] = parser_cls  # cache
            return parser_cls
        except Exception as e:
            logger.exception(
                "Failed to import lazy reasoning parser '%s' from %s: %s",
                name,
                module_path,
                e,
            )
            raise

    @classmethod
    def _register_module(
        cls,
        module: type[ReasoningParser],
        module_name: str | list[str] | None = None,
        force: bool = True,
    ) -> None:
        """Register a ReasoningParser class immediately."""
        if not issubclass(module, ReasoningParser):
            raise TypeError(
                f"module must be subclass of ReasoningParser, but got {type(module)}"
            )

        if module_name is None:
            module_names = [module.__name__]
        elif isinstance(module_name, str):
            module_names = [module_name]
        elif is_list_of(module_name, str):
            module_names = module_name
        else:
            raise TypeError("module_name must be str, list[str], or None.")

        for name in module_names:
            if not force and name in cls.reasoning_parsers:
                existed = cls.reasoning_parsers[name]
                raise KeyError(f"{name} is already registered at {existed.__module__}")
            cls.reasoning_parsers[name] = module

    @classmethod
    def register_lazy_module(cls, name: str, module_path: str, class_name: str) -> None:
        """
        Register a lazy module mapping for delayed import.

        Example:
            ReasoningParserManager.register_lazy_module(
                name="qwen3",
                module_path="vllm.reasoning.parsers.qwen3_reasoning_parser",
                class_name="Qwen3ReasoningParser",
            )
        """
        cls.lazy_parsers[name] = (module_path, class_name)

    @classmethod
    def register_module(
        cls,
        name: str | list[str] | None = None,
        force: bool = True,
        module: type[ReasoningParser] | None = None,
    ) -> (
        type[ReasoningParser] | Callable[[type[ReasoningParser]], type[ReasoningParser]]
    ):
        """
        Register module with the given name or name list. it can be used as a
        decoder(with module as None) or normal function(with module as not
        None).
        """
        if not isinstance(force, bool):
            raise TypeError(f"force must be a boolean, but got {type(force)}")

        # Immediate registration (explicit call)
        if module is not None:
            cls._register_module(module=module, module_name=name, force=force)
            return module

        # Decorator usage
        def _decorator(obj: type[ReasoningParser]) -> type[ReasoningParser]:
            module_path = obj.__module__
            class_name = obj.__name__

            if isinstance(name, str):
                names = [name]
            elif is_list_of(name, str):
                names = name
            else:
                names = [class_name]

            for n in names:
                cls.lazy_parsers[n] = (module_path, class_name)

            return obj

        return _decorator

    @classmethod
    def import_reasoning_parser(cls, plugin_path: str) -> None:
        """
        Import a user-defined reasoning parser by the path
        of the reasoning parser define file.
        """
        module_name = os.path.splitext(os.path.basename(plugin_path))[0]

        try:
            import_from_path(module_name, plugin_path)
        except Exception:
            logger.exception(
                "Failed to load module '%s' from %s.", module_name, plugin_path
            )
            return
