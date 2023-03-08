// This file is auto-generated. See "generate_kernels.py"
#include "../../cutlass_dual_gemm.h"

template<>
__global__ void DualKernel<cutlass::gemm::kernel::DualGemm<cutlass::half_t, cutlass::half_t, true, cutlass::epilogue::thread::Sigmoid, cutlass::arch::Sm80>>(typename cutlass::gemm::kernel::DualGemm<cutlass::half_t, cutlass::half_t, true, cutlass::epilogue::thread::Sigmoid, cutlass::arch::Sm80>::Params params) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800 && __CUDA_ARCH__ < 900
  using Operator = cutlass::gemm::kernel::DualGemm<cutlass::half_t, cutlass::half_t, true, cutlass::epilogue::thread::Sigmoid, cutlass::arch::Sm80>; 
  Operator op;
  op(params);
#endif
    printf(
        "FATAL: kernel `DualKernel` is for sm80-sm90, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}

template<>
__global__ void DualKernel<cutlass::gemm::kernel::DualGemm<cutlass::half_t, cutlass::half_t, true, cutlass::epilogue::thread::SiLu, cutlass::arch::Sm80>>(typename cutlass::gemm::kernel::DualGemm<cutlass::half_t, cutlass::half_t, true, cutlass::epilogue::thread::SiLu, cutlass::arch::Sm80>::Params params) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800 && __CUDA_ARCH__ < 900
  using Operator = cutlass::gemm::kernel::DualGemm<cutlass::half_t, cutlass::half_t, true, cutlass::epilogue::thread::SiLu, cutlass::arch::Sm80>; 
  Operator op;
  op(params);
#endif
    printf(
        "FATAL: kernel `DualKernel` is for sm80-sm90, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}

template<>
__global__ void DualKernel<cutlass::gemm::kernel::DualGemm<cutlass::half_t, cutlass::half_t, true, cutlass::epilogue::thread::GELU_taylor, cutlass::arch::Sm80>>(typename cutlass::gemm::kernel::DualGemm<cutlass::half_t, cutlass::half_t, true, cutlass::epilogue::thread::GELU_taylor, cutlass::arch::Sm80>::Params params) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800 && __CUDA_ARCH__ < 900
  using Operator = cutlass::gemm::kernel::DualGemm<cutlass::half_t, cutlass::half_t, true, cutlass::epilogue::thread::GELU_taylor, cutlass::arch::Sm80>; 
  Operator op;
  op(params);
#endif
    printf(
        "FATAL: kernel `DualKernel` is for sm80-sm90, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}

template<>
__global__ void DualKernel<cutlass::gemm::kernel::DualGemm<cutlass::half_t, cutlass::half_t, false, cutlass::epilogue::thread::Sigmoid, cutlass::arch::Sm80>>(typename cutlass::gemm::kernel::DualGemm<cutlass::half_t, cutlass::half_t, false, cutlass::epilogue::thread::Sigmoid, cutlass::arch::Sm80>::Params params) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800 && __CUDA_ARCH__ < 900
  using Operator = cutlass::gemm::kernel::DualGemm<cutlass::half_t, cutlass::half_t, false, cutlass::epilogue::thread::Sigmoid, cutlass::arch::Sm80>; 
  Operator op;
  op(params);
#endif
    printf(
        "FATAL: kernel `DualKernel` is for sm80-sm90, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}

template<>
__global__ void DualKernel<cutlass::gemm::kernel::DualGemm<cutlass::half_t, cutlass::half_t, false, cutlass::epilogue::thread::SiLu, cutlass::arch::Sm80>>(typename cutlass::gemm::kernel::DualGemm<cutlass::half_t, cutlass::half_t, false, cutlass::epilogue::thread::SiLu, cutlass::arch::Sm80>::Params params) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800 && __CUDA_ARCH__ < 900
  using Operator = cutlass::gemm::kernel::DualGemm<cutlass::half_t, cutlass::half_t, false, cutlass::epilogue::thread::SiLu, cutlass::arch::Sm80>; 
  Operator op;
  op(params);
#endif
    printf(
        "FATAL: kernel `DualKernel` is for sm80-sm90, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}

template<>
__global__ void DualKernel<cutlass::gemm::kernel::DualGemm<cutlass::half_t, cutlass::half_t, false, cutlass::epilogue::thread::GELU_taylor, cutlass::arch::Sm80>>(typename cutlass::gemm::kernel::DualGemm<cutlass::half_t, cutlass::half_t, false, cutlass::epilogue::thread::GELU_taylor, cutlass::arch::Sm80>::Params params) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800 && __CUDA_ARCH__ < 900
  using Operator = cutlass::gemm::kernel::DualGemm<cutlass::half_t, cutlass::half_t, false, cutlass::epilogue::thread::GELU_taylor, cutlass::arch::Sm80>; 
  Operator op;
  op(params);
#endif
    printf(
        "FATAL: kernel `DualKernel` is for sm80-sm90, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}
