#pragma once

#include "scaled_mm_c2x.cuh"
#include "cutlass/float8.h"

/**
 * This file defines Gemm kernel configurations for SM89 (FP8) based on the Gemm
 * shape.
 */

namespace vllm {

template <typename InType, typename OutType,
          template <typename, typename> typename Epilogue>
struct sm89_fp8_fallback_gemm {
  // Shared Memory required by this Gemm - 61440 bytes
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using TileShape = typename cutlass::gemm::GemmShape<64, 128, 64>;
  using WarpShape = typename cutlass::gemm::GemmShape<32, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;
  using FP8MathOperator = typename cutlass::arch::OpMultiplyAdd;
  using Cutlass2xGemm =
      cutlass_2x_gemm<cutlass::arch::Sm89, enable_sm89_to_sm90, InType, OutType,
                      Epilogue, TileShape, WarpShape, InstructionShape, 5,
                      FP8MathOperator>;
};

struct sm89_fp8_config_default {
  // M in (256, inf)
  using WarpShape = typename cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;
  using FP8MathOperator = typename cutlass::arch::OpMultiplyAddFastAccum;

  template <typename InType, typename OutType,
            template <typename, typename> typename Epilogue,
            typename... EpilogueArgs>
  static void dispatch(torch::Tensor& out, torch::Tensor const& a,
                       torch::Tensor const& b, EpilogueArgs&&... args) {
    static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
    TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn);

    using FallbackGemm =
        typename sm89_fp8_fallback_gemm<InType, OutType,
                                        Epilogue>::Cutlass2xGemm;

    uint32_t const n = out.size(1);
    uint32_t const np2 = next_pow_2(n);

    if (np2 <= 4096) {
      using TileShape = typename cutlass::gemm::GemmShape<128, 128, 64>;

      return vllm::fallback_cutlass_gemm_caller<
          vllm::cutlass_2x_gemm<cutlass::arch::Sm89, vllm::enable_sm89_to_sm90,
                                InType, OutType, Epilogue, TileShape, WarpShape,
                                InstructionShape, 5, FP8MathOperator>,
          FallbackGemm>(out, a, b, std::forward<EpilogueArgs>(args)...);
    } else if (np2 <= 8192) {
      using TileShape = typename cutlass::gemm::GemmShape<256, 128, 64>;

      return vllm::fallback_cutlass_gemm_caller<
          vllm::cutlass_2x_gemm<cutlass::arch::Sm89, vllm::enable_sm89_to_sm90,
                                InType, OutType, Epilogue, TileShape, WarpShape,
                                InstructionShape, 3, FP8MathOperator>,
          FallbackGemm>(out, a, b, std::forward<EpilogueArgs>(args)...);

    } else {
      using TileShape = typename cutlass::gemm::GemmShape<128, 128, 64>;

      return vllm::fallback_cutlass_gemm_caller<
          vllm::cutlass_2x_gemm<cutlass::arch::Sm89, vllm::enable_sm89_to_sm90,
                                InType, OutType, Epilogue, TileShape, WarpShape,
                                InstructionShape, 5, FP8MathOperator>,
          FallbackGemm>(out, a, b, std::forward<EpilogueArgs>(args)...);
    }
  }
};

struct sm89_fp8_config_M256 {
  // M in (128, 256]
  using WarpShape = typename cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;
  using FP8MathOperator = typename cutlass::arch::OpMultiplyAddFastAccum;

  template <typename InType, typename OutType,
            template <typename, typename> typename Epilogue,
            typename... EpilogueArgs>
  static void dispatch(torch::Tensor& out, torch::Tensor const& a,
                       torch::Tensor const& b, EpilogueArgs&&... args) {
    static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
    TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn);

    using FallbackGemm =
        typename sm89_fp8_fallback_gemm<InType, OutType,
                                        Epilogue>::Cutlass2xGemm;

    uint32_t const n = out.size(1);
    uint32_t const np2 = next_pow_2(n);

    if (np2 <= 4096) {
      using TileShape = typename cutlass::gemm::GemmShape<64, 128, 128>;

      return vllm::fallback_cutlass_gemm_caller<
          vllm::cutlass_2x_gemm<cutlass::arch::Sm89, vllm::enable_sm89_to_sm90,
                                InType, OutType, Epilogue, TileShape, WarpShape,
                                InstructionShape, 3, FP8MathOperator>,
          FallbackGemm>(out, a, b, std::forward<EpilogueArgs>(args)...);
    } else {
      using TileShape = typename cutlass::gemm::GemmShape<128, 128, 64>;

      return vllm::fallback_cutlass_gemm_caller<
          vllm::cutlass_2x_gemm<cutlass::arch::Sm89, vllm::enable_sm89_to_sm90,
                                InType, OutType, Epilogue, TileShape, WarpShape,
                                InstructionShape, 5, FP8MathOperator>,
          FallbackGemm>(out, a, b, std::forward<EpilogueArgs>(args)...);
    }
  }
};

struct sm89_fp8_config_M128 {
  // M in (64, 128]
  using WarpShape = typename cutlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;
  using FP8MathOperator = typename cutlass::arch::OpMultiplyAddFastAccum;

  template <typename InType, typename OutType,
            template <typename, typename> typename Epilogue,
            typename... EpilogueArgs>
  static void dispatch(torch::Tensor& out, torch::Tensor const& a,
                       torch::Tensor const& b, EpilogueArgs&&... args) {
    static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
    TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn);

    using FallbackGemm =
        typename sm89_fp8_fallback_gemm<InType, OutType,
                                        Epilogue>::Cutlass2xGemm;

    uint32_t const n = out.size(1);
    uint32_t const np2 = next_pow_2(n);

    if (np2 <= 8192) {
      using TileShape = typename cutlass::gemm::GemmShape<64, 128, 128>;

      return vllm::fallback_cutlass_gemm_caller<
          vllm::cutlass_2x_gemm<cutlass::arch::Sm89, vllm::enable_sm89_to_sm90,
                                InType, OutType, Epilogue, TileShape, WarpShape,
                                InstructionShape, 3, FP8MathOperator>,
          FallbackGemm>(out, a, b, std::forward<EpilogueArgs>(args)...);

    } else if (np2 <= 16384) {
      using TileShape = typename cutlass::gemm::GemmShape<128, 128, 64>;

      return vllm::fallback_cutlass_gemm_caller<
          vllm::cutlass_2x_gemm<cutlass::arch::Sm89, vllm::enable_sm89_to_sm90,
                                InType, OutType, Epilogue, TileShape, WarpShape,
                                InstructionShape, 5, FP8MathOperator>,
          FallbackGemm>(out, a, b, std::forward<EpilogueArgs>(args)...);
    } else {
      using TileShape = typename cutlass::gemm::GemmShape<128, 64, 128>;

      return vllm::fallback_cutlass_gemm_caller<
          vllm::cutlass_2x_gemm<cutlass::arch::Sm89, vllm::enable_sm89_to_sm90,
                                InType, OutType, Epilogue, TileShape, WarpShape,
                                InstructionShape, 3, FP8MathOperator>,
          FallbackGemm>(out, a, b, std::forward<EpilogueArgs>(args)...);
    }
  }
};

struct sm89_fp8_config_M64 {
  // M in (32, 64]
  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;

  template <typename InType, typename OutType,
            template <typename, typename> typename Epilogue,
            typename... EpilogueArgs>
  static void dispatch(torch::Tensor& out, torch::Tensor const& a,
                       torch::Tensor const& b, EpilogueArgs&&... args) {
    static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
    TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn);

    using FallbackGemm =
        typename sm89_fp8_fallback_gemm<InType, OutType,
                                        Epilogue>::Cutlass2xGemm;

    uint32_t const n = out.size(1);
    uint32_t const np2 = next_pow_2(n);

    if (np2 <= 8196) {
      using TileShape = typename cutlass::gemm::GemmShape<64, 64, 128>;
      using WarpShape = typename cutlass::gemm::GemmShape<32, 64, 64>;
      using FP8MathOperator = typename cutlass::arch::OpMultiplyAdd;

      return vllm::fallback_cutlass_gemm_caller<
          vllm::cutlass_2x_gemm<cutlass::arch::Sm89, vllm::enable_sm89_to_sm90,
                                InType, OutType, Epilogue, TileShape, WarpShape,
                                InstructionShape, 5, FP8MathOperator>,
          FallbackGemm>(out, a, b, std::forward<EpilogueArgs>(args)...);
    } else if (np2 <= 16384) {
      using TileShape = typename cutlass::gemm::GemmShape<64, 128, 128>;
      using WarpShape = typename cutlass::gemm::GemmShape<64, 64, 64>;
      using FP8MathOperator = typename cutlass::arch::OpMultiplyAddFastAccum;

      return vllm::fallback_cutlass_gemm_caller<
          vllm::cutlass_2x_gemm<cutlass::arch::Sm89, vllm::enable_sm89_to_sm90,
                                InType, OutType, Epilogue, TileShape, WarpShape,
                                InstructionShape, 3, FP8MathOperator>,
          FallbackGemm>(out, a, b, std::forward<EpilogueArgs>(args)...);
    } else {
      using TileShape = typename cutlass::gemm::GemmShape<64, 64, 128>;
      using WarpShape = typename cutlass::gemm::GemmShape<32, 64, 64>;
      using FP8MathOperator = typename cutlass::arch::OpMultiplyAdd;

      return vllm::fallback_cutlass_gemm_caller<
          vllm::cutlass_2x_gemm<cutlass::arch::Sm89, vllm::enable_sm89_to_sm90,
                                InType, OutType, Epilogue, TileShape, WarpShape,
                                InstructionShape, 5, FP8MathOperator>,
          FallbackGemm>(out, a, b, std::forward<EpilogueArgs>(args)...);
    }
  }
};

struct sm89_fp8_config_M32 {
  // M in (16, 32]
  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;
  using FP8MathOperator = typename cutlass::arch::OpMultiplyAddFastAccum;

  template <typename InType, typename OutType,
            template <typename, typename> typename Epilogue,
            typename... EpilogueArgs>
  static void dispatch(torch::Tensor& out, torch::Tensor const& a,
                       torch::Tensor const& b, EpilogueArgs&&... args) {
    static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
    TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn);

    using FallbackGemm =
        typename sm89_fp8_fallback_gemm<InType, OutType,
                                        Epilogue>::Cutlass2xGemm;

    uint32_t const n = out.size(1);
    uint32_t const np2 = next_pow_2(n);

    if (np2 <= 8192) {
      using TileShape = typename cutlass::gemm::GemmShape<32, 64, 128>;
      using WarpShape = typename cutlass::gemm::GemmShape<16, 64, 64>;

      return vllm::fallback_cutlass_gemm_caller<
          vllm::cutlass_2x_gemm<cutlass::arch::Sm89, vllm::enable_sm89_to_sm90,
                                InType, OutType, Epilogue, TileShape, WarpShape,
                                InstructionShape, 5, FP8MathOperator>,
          FallbackGemm>(out, a, b, std::forward<EpilogueArgs>(args)...);
    } else if (np2 <= 16384) {
      using TileShape = typename cutlass::gemm::GemmShape<32, 128, 128>;
      using WarpShape = typename cutlass::gemm::GemmShape<32, 64, 64>;

      return vllm::fallback_cutlass_gemm_caller<
          vllm::cutlass_2x_gemm<cutlass::arch::Sm89, vllm::enable_sm89_to_sm90,
                                InType, OutType, Epilogue, TileShape, WarpShape,
                                InstructionShape, 4, FP8MathOperator>,
          FallbackGemm>(out, a, b, std::forward<EpilogueArgs>(args)...);
    } else {
      using TileShape = typename cutlass::gemm::GemmShape<32, 64, 128>;
      using WarpShape = typename cutlass::gemm::GemmShape<16, 64, 64>;

      return vllm::fallback_cutlass_gemm_caller<
          vllm::cutlass_2x_gemm<cutlass::arch::Sm89, vllm::enable_sm89_to_sm90,
                                InType, OutType, Epilogue, TileShape, WarpShape,
                                InstructionShape, 5, FP8MathOperator>,
          FallbackGemm>(out, a, b, std::forward<EpilogueArgs>(args)...);
    }
  }
};

struct sm89_fp8_config_M16 {
  // M in [1, 16]
  using WarpShape = typename cutlass::gemm::GemmShape<16, 64, 64>;                          //编译器看到 cutlass::gemm::GemmShape<16, 64, 64> 时，它不确定这个东西到底是 类型还是静态成员变量或者其他  typename明确告诉这是一个类型
  using InstructionShape = typename cutlass::gemm::GemmShape<16, 8, 32>;                    //cuda 层级的title大小  用于tensor core的单次乘加操作
  using FP8MathOperator = typename cutlass::arch::OpMultiplyAddFastAccum;                   //使用FP8的乘加操作
  static const int32_t MainLoopStages = 5;                                                  //主循环pipeline阶段数 用于隐藏访问延迟 数值越大 越能隐藏访问延迟？为什么？即在每个threadblock内部同事加载多少步tile的数据

  template <typename InType, typename OutType,
            template <typename, typename> typename Epilogue,
            typename... EpilogueArgs>
  static void dispatch(torch::Tensor& out, torch::Tensor const& a,                         //dispatch函数 真正调度CUTLASS GEMM kernel
                       torch::Tensor const& b, EpilogueArgs&&... args) {
    static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
    TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn);

    using FallbackGemm =                                                                   //当我们自定义kernel不支持某些形状,会自动退回到这个更通用的GEMM
        typename sm89_fp8_fallback_gemm<InType, OutType,
                                        Epilogue>::Cutlass2xGemm;

    uint32_t const n = out.size(1);                                                        //输出矩阵的列数N
    uint32_t const np2 = next_pow_2(n);                                                    //计算N的下一个2的幂

    if (np2 <= 8192) {                                                                     //根据N的大小选择不同的TileShape 
      using TileShape = typename cutlass::gemm::GemmShape<16, 64, 128>;                    //TileShape是threadblock(一个CUDA block)负责计算的整体大小

      return vllm::fallback_cutlass_gemm_caller<                                           //调用封装好的CUTAKASS 2.X kernel启动器
          vllm::cutlass_2x_gemm<cutlass::arch::Sm89, vllm::enable_sm89_to_sm90,
                                InType, OutType, Epilogue, TileShape, WarpShape,
                                InstructionShape, MainLoopStages,
                                FP8MathOperator>,
          FallbackGemm>(out, a, b, std::forward<EpilogueArgs>(args)...);
    } else if (np2 <= 24576) {                                                              //对于超大列数 使用更大的tile M=32 , N=64,K=128
      using TileShape = typename cutlass::gemm::GemmShape<16, 128, 64>;

      return vllm::fallback_cutlass_gemm_caller<
          vllm::cutlass_2x_gemm<cutlass::arch::Sm89, vllm::enable_sm89_to_sm90,
                                InType, OutType, Epilogue, TileShape, WarpShape,
                                InstructionShape, MainLoopStages,
                                FP8MathOperator>,
          FallbackGemm>(out, a, b, std::forward<EpilogueArgs>(args)...);
    } else {
      using TileShape = typename cutlass::gemm::GemmShape<32, 64, 128>;

      return vllm::fallback_cutlass_gemm_caller<
          vllm::cutlass_2x_gemm<cutlass::arch::Sm89, vllm::enable_sm89_to_sm90,
                                InType, OutType, Epilogue, TileShape, WarpShape,
                                InstructionShape, MainLoopStages,
                                FP8MathOperator>,
          FallbackGemm>(out, a, b, std::forward<EpilogueArgs>(args)...);
    }
  }
};

template <typename InType, typename OutType,
          template <typename, typename> typename Epilogue,
          typename... EpilogueArgs>
inline void cutlass_gemm_sm89_fp8_dispatch(torch::Tensor& out,                        //inline直接把代码复制粘贴到调用它的地方
                                           torch::Tensor const& a,
                                           torch::Tensor const& b,
                                           EpilogueArgs&&... args) {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(b.dtype() == torch::kFloat8_e4m3fn);

  uint32_t const m = a.size(0);
  uint32_t const mp2 =
      std::max(static_cast<uint32_t>(16), next_pow_2(m));  // next power of 2           计算不小于M的2的整数次幂

  if (mp2 <= 16) {
    // M in [1, 16]
    return sm89_fp8_config_M16::dispatch<InType, OutType, Epilogue>(
        out, a, b, std::forward<EpilogueArgs>(args)...);
  } else if (mp2 <= 32) {
    // M in (16, 32]
    return sm89_fp8_config_M32::dispatch<InType, OutType, Epilogue>(
        out, a, b, std::forward<EpilogueArgs>(args)...);
  } else if (mp2 <= 64) {
    // M in (32, 64]
    return sm89_fp8_config_M64::dispatch<InType, OutType, Epilogue>(
        out, a, b, std::forward<EpilogueArgs>(args)...);
  } else if (mp2 <= 128) {
    // M in (64, 128]
    return sm89_fp8_config_M128::dispatch<InType, OutType, Epilogue>(
        out, a, b, std::forward<EpilogueArgs>(args)...);
  } else if (mp2 <= 256) {
    // M in (128, 256]
    return sm89_fp8_config_M256::dispatch<InType, OutType, Epilogue>(
        out, a, b, std::forward<EpilogueArgs>(args)...);
  } else {
    // M in (256, inf)
    return sm89_fp8_config_default::dispatch<InType, OutType, Epilogue>(
        out, a, b, std::forward<EpilogueArgs>(args)...);
  }
}

}  // namespace vllm
