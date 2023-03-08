// This file is auto-generated. See "generate_kernels.py"
#include "../../cutlass_dual_gemm.h"

template<>
__global__ void DualKernel<cutlass::gemm::kernel::DualGemm<cutlass::half_t, float, true, cutlass::epilogue::thread::Sigmoid, cutlass::arch::Sm75>>(typename cutlass::gemm::kernel::DualGemm<cutlass::half_t, float, true, cutlass::epilogue::thread::Sigmoid, cutlass::arch::Sm75>::Params params) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750 && __CUDA_ARCH__ < 800
  using Operator = cutlass::gemm::kernel::DualGemm<cutlass::half_t, float, true, cutlass::epilogue::thread::Sigmoid, cutlass::arch::Sm75>; 
  Operator op;
  op(params);
#endif
    printf(
        "FATAL: kernel `DualKernel` is for sm75-sm80, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}

template<>
__global__ void DualKernel<cutlass::gemm::kernel::DualGemm<cutlass::half_t, float, true, cutlass::epilogue::thread::SiLu, cutlass::arch::Sm75>>(typename cutlass::gemm::kernel::DualGemm<cutlass::half_t, float, true, cutlass::epilogue::thread::SiLu, cutlass::arch::Sm75>::Params params) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750 && __CUDA_ARCH__ < 800
  using Operator = cutlass::gemm::kernel::DualGemm<cutlass::half_t, float, true, cutlass::epilogue::thread::SiLu, cutlass::arch::Sm75>; 
  Operator op;
  op(params);
#endif
    printf(
        "FATAL: kernel `DualKernel` is for sm75-sm80, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}

template<>
__global__ void DualKernel<cutlass::gemm::kernel::DualGemm<cutlass::half_t, float, true, cutlass::epilogue::thread::GELU_taylor, cutlass::arch::Sm75>>(typename cutlass::gemm::kernel::DualGemm<cutlass::half_t, float, true, cutlass::epilogue::thread::GELU_taylor, cutlass::arch::Sm75>::Params params) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750 && __CUDA_ARCH__ < 800
  using Operator = cutlass::gemm::kernel::DualGemm<cutlass::half_t, float, true, cutlass::epilogue::thread::GELU_taylor, cutlass::arch::Sm75>; 
  Operator op;
  op(params);
#endif
    printf(
        "FATAL: kernel `DualKernel` is for sm75-sm80, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}

template<>
__global__ void DualKernel<cutlass::gemm::kernel::DualGemm<cutlass::half_t, float, false, cutlass::epilogue::thread::Sigmoid, cutlass::arch::Sm75>>(typename cutlass::gemm::kernel::DualGemm<cutlass::half_t, float, false, cutlass::epilogue::thread::Sigmoid, cutlass::arch::Sm75>::Params params) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750 && __CUDA_ARCH__ < 800
  using Operator = cutlass::gemm::kernel::DualGemm<cutlass::half_t, float, false, cutlass::epilogue::thread::Sigmoid, cutlass::arch::Sm75>; 
  Operator op;
  op(params);
#endif
    printf(
        "FATAL: kernel `DualKernel` is for sm75-sm80, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}

template<>
__global__ void DualKernel<cutlass::gemm::kernel::DualGemm<cutlass::half_t, float, false, cutlass::epilogue::thread::SiLu, cutlass::arch::Sm75>>(typename cutlass::gemm::kernel::DualGemm<cutlass::half_t, float, false, cutlass::epilogue::thread::SiLu, cutlass::arch::Sm75>::Params params) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750 && __CUDA_ARCH__ < 800
  using Operator = cutlass::gemm::kernel::DualGemm<cutlass::half_t, float, false, cutlass::epilogue::thread::SiLu, cutlass::arch::Sm75>; 
  Operator op;
  op(params);
#endif
    printf(
        "FATAL: kernel `DualKernel` is for sm75-sm80, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}

template<>
__global__ void DualKernel<cutlass::gemm::kernel::DualGemm<cutlass::half_t, float, false, cutlass::epilogue::thread::GELU_taylor, cutlass::arch::Sm75>>(typename cutlass::gemm::kernel::DualGemm<cutlass::half_t, float, false, cutlass::epilogue::thread::GELU_taylor, cutlass::arch::Sm75>::Params params) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750 && __CUDA_ARCH__ < 800
  using Operator = cutlass::gemm::kernel::DualGemm<cutlass::half_t, float, false, cutlass::epilogue::thread::GELU_taylor, cutlass::arch::Sm75>; 
  Operator op;
  op(params);
#endif
    printf(
        "FATAL: kernel `DualKernel` is for sm75-sm80, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}
