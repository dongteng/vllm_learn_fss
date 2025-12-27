#!/usr/bin/env bash

nsys profile \
  -o report.nsys-rep \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --delay 30 \
  --duration 150 \
  vllm serve Qwen/Qwen3-1.7B \
  --host 0.0.0.0 \
  --port 13333          


# 客户端
no_proxy="*" HTTP_PROXY="" HTTPS_PROXY="" http_proxy="" https_proxy="" \
vllm bench serve \
--backend vllm \
--model Qwen/Qwen3-1.7B \
--num-prompts 1 \
--dataset-name random \
--random-input 1280 \
--random-output 30000 \
--port 13333