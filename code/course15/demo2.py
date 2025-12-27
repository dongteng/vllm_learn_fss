from vllm import LLM
from vllm.config import CompilationConfig, CUDAGraphMode, PassConfig
from vllm.sampling_params import SamplingParams


def main():
    # 创建采样参数，控制文本生成的行为
    sampling_params = SamplingParams(
        temperature=0.8,   # 温度参数：控制输出的随机性 (0.0-1.0+)，值越高输出越随机
        top_p=0.95         # 核采样参数：仅从概率累积和达到95%的token中采样
    )

    # 创建编译配置 - 针对内存优化进行调整
    compilation_config = CompilationConfig(
        mode=3,  # 编译模式级别 3
        backend="inductor",  # 使用 PyTorch Inductor 作为编译后端
        cudagraph_mode=CUDAGraphMode.PIECEWISE,  # 使用分块 CUDA 图模式，比完整图模式内存更友好
        cudagraph_capture_sizes=[1, 4, 8],  # 减少捕获的图大小范围，降低调优负载
        compile_sizes=[1, 4],  # 编译特定的输入大小，减少编译工作量
        use_inductor_graph_partition=False,  # 禁用Inductor图分区，简化编译过程
        cache_dir="./vllm_compile_cache",  # 指定编译缓存目录，避免重复编译
    )

    # 创建 LLM 实例并启用编译优化
    llm = LLM(
        model="Qwen/Qwen3-1.7B",  # 模型路径：使用1.7B参数的小型模型
        compilation_config=compilation_config,  # 应用上述编译配置
        max_model_len=4096,  # 最大模型长度：限制序列长度以减少内存使用
        gpu_memory_utilization=0.9,  # GPU内存利用率：90%的使用率，保留一些余量
    )

    # 执行推理（首次调用会触发编译，后续调用使用缓存）
    prompts = [
        "Hello, how are you?",  # 简单的问候提示
        "The future of AI is",  # 开放式生成提示
    ]

    # 生成文本 - 第一次运行会较慢（编译阶段），后续运行会更快
    outputs = llm.generate(prompts, sampling_params)

    # 输出生成结果
    for output in outputs:
        prompt = output.prompt  # 原始提示文本
        generated_text = output.outputs[0].text  # 模型生成的文本
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == "__main__":
    # 程序入口点
    main()
