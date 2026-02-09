http_proxy= https_proxy= no_proxy=* python3 -m vllm.entrypoints.openai.api_server \
  --model "/data/nas/zhjm/model_cache/docker-shared/Qwen3-1.7B" \
  --dtype float16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.95 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 256 \
  --port "13311"    \
  --enforce-eager