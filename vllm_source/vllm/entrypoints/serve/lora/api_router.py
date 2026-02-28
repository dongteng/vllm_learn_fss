# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import model_hosting_container_standards.sagemaker as sagemaker_standards
from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse, Response

from vllm import envs
from vllm.entrypoints.openai.api_server import models, validate_json_request
from vllm.entrypoints.openai.protocol import (
    ErrorResponse,
    LoadLoRAAdapterRequest,
    UnloadLoRAAdapterRequest,
)
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.logger import init_logger

logger = init_logger(__name__)
router = APIRouter()


def attach_router(app: FastAPI):
    """
    在 vLLM 的 OpenAI 兼容 API 服务器中，可选地注册两个动态 LoRA 适配器管理端点（/v1/load_lora_adapter 和 /v1/unload_lora_adapter），允许运行时（runtime）加载/卸载 LoRA 适配器。但这个功能只在开发/测试环境下推荐使用，生产环境默认关闭（因为有安全、稳定性、性能风险）。
    """
    if not envs.VLLM_ALLOW_RUNTIME_LORA_UPDATING:
        """If LoRA dynamic loading & unloading is not enabled, do nothing."""
        return
    logger.warning(
        "LoRA dynamic loading & unloading is enabled in the API server. "
        "This should ONLY be used for local development!"
    )

    @sagemaker_standards.register_load_adapter_handler(
        request_shape={
            "lora_name": "body.name",
            "lora_path": "body.src",
        },
    )
    @router.post("/v1/load_lora_adapter", dependencies=[Depends(validate_json_request)])
    async def load_lora_adapter(request: LoadLoRAAdapterRequest, raw_request: Request):
        handler: OpenAIServingModels = models(raw_request)
        response = await handler.load_lora_adapter(request)
        if isinstance(response, ErrorResponse):
            return JSONResponse(
                content=response.model_dump(), status_code=response.error.code
            )

        return Response(status_code=200, content=response)

    @sagemaker_standards.register_unload_adapter_handler(
        request_shape={
            "lora_name": "path_params.adapter_name",
        }
    )
    @router.post(
        "/v1/unload_lora_adapter", dependencies=[Depends(validate_json_request)]
    )
    async def unload_lora_adapter(
        request: UnloadLoRAAdapterRequest, raw_request: Request
    ):
        handler: OpenAIServingModels = models(raw_request)
        response = await handler.unload_lora_adapter(request)
        if isinstance(response, ErrorResponse):
            return JSONResponse(
                content=response.model_dump(), status_code=response.error.code
            )

        return Response(status_code=200, content=response)

    # register the router
    app.include_router(router)
