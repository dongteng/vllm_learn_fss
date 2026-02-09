import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["VLLM_USE_V1"] = "1"  # 必须在 import vllm 之前！
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from vllm import LLM, SamplingParams

#Gauge瞬时值 显存
#Counter累积量(如token)总数
from vllm.v1.metrics.reader import Counter, Gauge, Histogram, Vector

# Sample prompts.
prompts = ["The future of AI is"] * 128
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


def main():
    # Create an LLM.
    llm = LLM(
        model="/data/nas/zhjm/model_cache/docker-shared/Qwen3-1.7B",
        max_model_len=4096,
        max_num_seqs=128,#最多同时调度128条序列
        gpu_memory_utilization=0.4,
        disable_log_stats=False,
    )

    # Generate texts from the prompts.
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    t1 = time.perf_counter()
    new_tokens = sum(len(out.outputs[0].token_ids) for out in outputs)
    throughput = new_tokens / (t1 - t0) #总生成 token 数 ÷ 总耗时 = tok/s
    print(f"Throughput = {throughput:.2f} tok/s")

    # Print the outputs.
    print("-" * 50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        # print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        # print("-" * 50)

    # Dump all metrics
    for metric in llm.get_metrics():
        if isinstance(metric, Gauge): #显存使用量
            print(f"==={metric.name}当前占用的显存量 (gauge) = {metric.value}")
        elif isinstance(metric, Counter):
            print(f"==={metric.name} (counter) = {metric.value}")
        elif isinstance(metric, Vector):
            print(f"==={metric.name} (vector) = {metric.values}")
        elif isinstance(metric, Histogram):
            print(f"==={metric.name} (histogram)")
            print(f"===    sum = {metric.sum}")
            print(f"===    count = {metric.count}")
            for bucket_le, value in metric.buckets.items():
                print(f"    {bucket_le} = {value}")


if __name__ == "__main__":
    main()
