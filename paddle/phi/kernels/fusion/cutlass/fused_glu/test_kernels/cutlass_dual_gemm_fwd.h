// This file is auto-generated. See "generate_kernels.py"
#pragma once
#include "../cutlass_dual_gemm.h"

// ======== f16 / sm70 ========

template<>
__global__ void DualKernel<cutlass::gemm::kernel::DualGemm<cutlass::half_t, float, true, cutlass::epilogue::thread::Sigmoid, cutlass::arch::Sm70>>(typename cutlass::gemm::kernel::DualGemm<cutlass::half_t, float, true, cutlass::epilogue::thread::Sigmoid, cutlass::arch::Sm70>::Params params);

template<>
__global__ void DualKernel<cutlass::gemm::kernel::DualGemm<cutlass::half_t, float, true, cutlass::epilogue::thread::SiLu, cutlass::arch::Sm70>>(typename cutlass::gemm::kernel::DualGemm<cutlass::half_t, float, true, cutlass::epilogue::thread::SiLu, cutlass::arch::Sm70>::Params params);

template<>
__global__ void DualKernel<cutlass::gemm::kernel::DualGemm<cutlass::half_t, float, true, cutlass::epilogue::thread::GELU_taylor, cutlass::arch::Sm70>>(typename cutlass::gemm::kernel::DualGemm<cutlass::half_t, float, true, cutlass::epilogue::thread::GELU_taylor, cutlass::arch::Sm70>::Params params);

template<>
__global__ void DualKernel<cutlass::gemm::kernel::DualGemm<cutlass::half_t, float, false, cutlass::epilogue::thread::Sigmoid, cutlass::arch::Sm70>>(typename cutlass::gemm::kernel::DualGemm<cutlass::half_t, float, false, cutlass::epilogue::thread::Sigmoid, cutlass::arch::Sm70>::Params params);

template<>
__global__ void DualKernel<cutlass::gemm::kernel::DualGemm<cutlass::half_t, float, false, cutlass::epilogue::thread::SiLu, cutlass::arch::Sm70>>(typename cutlass::gemm::kernel::DualGemm<cutlass::half_t, float, false, cutlass::epilogue::thread::SiLu, cutlass::arch::Sm70>::Params params);

template<>
__global__ void DualKernel<cutlass::gemm::kernel::DualGemm<cutlass::half_t, float, false, cutlass::epilogue::thread::GELU_taylor, cutlass::arch::Sm70>>(typename cutlass::gemm::kernel::DualGemm<cutlass::half_t, float, false, cutlass::epilogue::thread::GELU_taylor, cutlass::arch::Sm70>::Params params);

template<>
__global__ void DualKernel<cutlass::gemm::kernel::DualGemm<cutlass::half_t, cutlass::half_t, true, cutlass::epilogue::thread::Sigmoid, cutlass::arch::Sm70>>(typename cutlass::gemm::kernel::DualGemm<cutlass::half_t, cutlass::half_t, true, cutlass::epilogue::thread::Sigmoid, cutlass::arch::Sm70>::Params params);

template<>
__global__ void DualKernel<cutlass::gemm::kernel::DualGemm<cutlass::half_t, cutlass::half_t, true, cutlass::epilogue::thread::SiLu, cutlass::arch::Sm70>>(typename cutlass::gemm::kernel::DualGemm<cutlass::half_t, cutlass::half_t, true, cutlass::epilogue::thread::SiLu, cutlass::arch::Sm70>::Params params);

template<>
__global__ void DualKernel<cutlass::gemm::kernel::DualGemm<cutlass::half_t, cutlass::half_t, true, cutlass::epilogue::thread::GELU_taylor, cutlass::arch::Sm70>>(typename cutlass::gemm::kernel::DualGemm<cutlass::half_t, cutlass::half_t, true, cutlass::epilogue::thread::GELU_taylor, cutlass::arch::Sm70>::Params params);

template<>
__global__ void DualKernel<cutlass::gemm::kernel::DualGemm<cutlass::half_t, cutlass::half_t, false, cutlass::epilogue::thread::Sigmoid, cutlass::arch::Sm70>>(typename cutlass::gemm::kernel::DualGemm<cutlass::half_t, cutlass::half_t, false, cutlass::epilogue::thread::Sigmoid, cutlass::arch::Sm70>::Params params);

template<>
__global__ void DualKernel<cutlass::gemm::kernel::DualGemm<cutlass::half_t, cutlass::half_t, false, cutlass::epilogue::thread::SiLu, cutlass::arch::Sm70>>(typename cutlass::gemm::kernel::DualGemm<cutlass::half_t, cutlass::half_t, false, cutlass::epilogue::thread::SiLu, cutlass::arch::Sm70>::Params params);

template<>
__global__ void DualKernel<cutlass::gemm::kernel::DualGemm<cutlass::half_t, cutlass::half_t, false, cutlass::epilogue::thread::GELU_taylor, cutlass::arch::Sm70>>(typename cutlass::gemm::kernel::DualGemm<cutlass::half_t, cutlass::half_t, false, cutlass::epilogue::thread::GELU_taylor, cutlass::arch::Sm70>::Params params);


// ======== f16 / sm75 ========

template<>
__global__ void DualKernel<cutlass::gemm::kernel::DualGemm<cutlass::half_t, float, true, cutlass::epilogue::thread::Sigmoid, cutlass::arch::Sm75>>(typename cutlass::gemm::kernel::DualGemm<cutlass::half_t, float, true, cutlass::epilogue::thread::Sigmoid, cutlass::arch::Sm75>::Params params);

template<>
__global__ void DualKernel<cutlass::gemm::kernel::DualGemm<cutlass::half_t, float, true, cutlass::epilogue::thread::SiLu, cutlass::arch::Sm75>>(typename cutlass::gemm::kernel::DualGemm<cutlass::half_t, float, true, cutlass::epilogue::thread::SiLu, cutlass::arch::Sm75>::Params params);

template<>
__global__ void DualKernel<cutlass::gemm::kernel::DualGemm<cutlass::half_t, float, true, cutlass::epilogue::thread::GELU_taylor, cutlass::arch::Sm75>>(typename cutlass::gemm::kernel::DualGemm<cutlass::half_t, float, true, cutlass::epilogue::thread::GELU_taylor, cutlass::arch::Sm75>::Params params);

template<>
__global__ void DualKernel<cutlass::gemm::kernel::DualGemm<cutlass::half_t, float, false, cutlass::epilogue::thread::Sigmoid, cutlass::arch::Sm75>>(typename cutlass::gemm::kernel::DualGemm<cutlass::half_t, float, false, cutlass::epilogue::thread::Sigmoid, cutlass::arch::Sm75>::Params params);

template<>
__global__ void DualKernel<cutlass::gemm::kernel::DualGemm<cutlass::half_t, float, false, cutlass::epilogue::thread::SiLu, cutlass::arch::Sm75>>(typename cutlass::gemm::kernel::DualGemm<cutlass::half_t, float, false, cutlass::epilogue::thread::SiLu, cutlass::arch::Sm75>::Params params);

template<>
__global__ void DualKernel<cutlass::gemm::kernel::DualGemm<cutlass::half_t, float, false, cutlass::epilogue::thread::GELU_taylor, cutlass::arch::Sm75>>(typename cutlass::gemm::kernel::DualGemm<cutlass::half_t, float, false, cutlass::epilogue::thread::GELU_taylor, cutlass::arch::Sm75>::Params params);

template<>
__global__ void DualKernel<cutlass::gemm::kernel::DualGemm<cutlass::half_t, cutlass::half_t, true, cutlass::epilogue::thread::Sigmoid, cutlass::arch::Sm75>>(typename cutlass::gemm::kernel::DualGemm<cutlass::half_t, cutlass::half_t, true, cutlass::epilogue::thread::Sigmoid, cutlass::arch::Sm75>::Params params);

template<>
__global__ void DualKernel<cutlass::gemm::kernel::DualGemm<cutlass::half_t, cutlass::half_t, true, cutlass::epilogue::thread::SiLu, cutlass::arch::Sm75>>(typename cutlass::gemm::kernel::DualGemm<cutlass::half_t, cutlass::half_t, true, cutlass::epilogue::thread::SiLu, cutlass::arch::Sm75>::Params params);

template<>
__global__ void DualKernel<cutlass::gemm::kernel::DualGemm<cutlass::half_t, cutlass::half_t, true, cutlass::epilogue::thread::GELU_taylor, cutlass::arch::Sm75>>(typename cutlass::gemm::kernel::DualGemm<cutlass::half_t, cutlass::half_t, true, cutlass::epilogue::thread::GELU_taylor, cutlass::arch::Sm75>::Params params);

template<>
__global__ void DualKernel<cutlass::gemm::kernel::DualGemm<cutlass::half_t, cutlass::half_t, false, cutlass::epilogue::thread::Sigmoid, cutlass::arch::Sm75>>(typename cutlass::gemm::kernel::DualGemm<cutlass::half_t, cutlass::half_t, false, cutlass::epilogue::thread::Sigmoid, cutlass::arch::Sm75>::Params params);

template<>
__global__ void DualKernel<cutlass::gemm::kernel::DualGemm<cutlass::half_t, cutlass::half_t, false, cutlass::epilogue::thread::SiLu, cutlass::arch::Sm75>>(typename cutlass::gemm::kernel::DualGemm<cutlass::half_t, cutlass::half_t, false, cutlass::epilogue::thread::SiLu, cutlass::arch::Sm75>::Params params);

template<>
__global__ void DualKernel<cutlass::gemm::kernel::DualGemm<cutlass::half_t, cutlass::half_t, false, cutlass::epilogue::thread::GELU_taylor, cutlass::arch::Sm75>>(typename cutlass::gemm::kernel::DualGemm<cutlass::half_t, cutlass::half_t, false, cutlass::epilogue::thread::GELU_taylor, cutlass::arch::Sm75>::Params params);


// ======== f16 / sm80 ========

template<>
__global__ void DualKernel<cutlass::gemm::kernel::DualGemm<cutlass::half_t, float, true, cutlass::epilogue::thread::Sigmoid, cutlass::arch::Sm80>>(typename cutlass::gemm::kernel::DualGemm<cutlass::half_t, float, true, cutlass::epilogue::thread::Sigmoid, cutlass::arch::Sm80>::Params params);

template<>
__global__ void DualKernel<cutlass::gemm::kernel::DualGemm<cutlass::half_t, float, true, cutlass::epilogue::thread::SiLu, cutlass::arch::Sm80>>(typename cutlass::gemm::kernel::DualGemm<cutlass::half_t, float, true, cutlass::epilogue::thread::SiLu, cutlass::arch::Sm80>::Params params);

template<>
__global__ void DualKernel<cutlass::gemm::kernel::DualGemm<cutlass::half_t, float, true, cutlass::epilogue::thread::GELU_taylor, cutlass::arch::Sm80>>(typename cutlass::gemm::kernel::DualGemm<cutlass::half_t, float, true, cutlass::epilogue::thread::GELU_taylor, cutlass::arch::Sm80>::Params params);

template<>
__global__ void DualKernel<cutlass::gemm::kernel::DualGemm<cutlass::half_t, float, false, cutlass::epilogue::thread::Sigmoid, cutlass::arch::Sm80>>(typename cutlass::gemm::kernel::DualGemm<cutlass::half_t, float, false, cutlass::epilogue::thread::Sigmoid, cutlass::arch::Sm80>::Params params);

template<>
__global__ void DualKernel<cutlass::gemm::kernel::DualGemm<cutlass::half_t, float, false, cutlass::epilogue::thread::SiLu, cutlass::arch::Sm80>>(typename cutlass::gemm::kernel::DualGemm<cutlass::half_t, float, false, cutlass::epilogue::thread::SiLu, cutlass::arch::Sm80>::Params params);

template<>
__global__ void DualKernel<cutlass::gemm::kernel::DualGemm<cutlass::half_t, float, false, cutlass::epilogue::thread::GELU_taylor, cutlass::arch::Sm80>>(typename cutlass::gemm::kernel::DualGemm<cutlass::half_t, float, false, cutlass::epilogue::thread::GELU_taylor, cutlass::arch::Sm80>::Params params);

template<>
__global__ void DualKernel<cutlass::gemm::kernel::DualGemm<cutlass::half_t, cutlass::half_t, true, cutlass::epilogue::thread::Sigmoid, cutlass::arch::Sm80>>(typename cutlass::gemm::kernel::DualGemm<cutlass::half_t, cutlass::half_t, true, cutlass::epilogue::thread::Sigmoid, cutlass::arch::Sm80>::Params params);

template<>
__global__ void DualKernel<cutlass::gemm::kernel::DualGemm<cutlass::half_t, cutlass::half_t, true, cutlass::epilogue::thread::SiLu, cutlass::arch::Sm80>>(typename cutlass::gemm::kernel::DualGemm<cutlass::half_t, cutlass::half_t, true, cutlass::epilogue::thread::SiLu, cutlass::arch::Sm80>::Params params);

template<>
__global__ void DualKernel<cutlass::gemm::kernel::DualGemm<cutlass::half_t, cutlass::half_t, true, cutlass::epilogue::thread::GELU_taylor, cutlass::arch::Sm80>>(typename cutlass::gemm::kernel::DualGemm<cutlass::half_t, cutlass::half_t, true, cutlass::epilogue::thread::GELU_taylor, cutlass::arch::Sm80>::Params params);

template<>
__global__ void DualKernel<cutlass::gemm::kernel::DualGemm<cutlass::half_t, cutlass::half_t, false, cutlass::epilogue::thread::Sigmoid, cutlass::arch::Sm80>>(typename cutlass::gemm::kernel::DualGemm<cutlass::half_t, cutlass::half_t, false, cutlass::epilogue::thread::Sigmoid, cutlass::arch::Sm80>::Params params);

template<>
__global__ void DualKernel<cutlass::gemm::kernel::DualGemm<cutlass::half_t, cutlass::half_t, false, cutlass::epilogue::thread::SiLu, cutlass::arch::Sm80>>(typename cutlass::gemm::kernel::DualGemm<cutlass::half_t, cutlass::half_t, false, cutlass::epilogue::thread::SiLu, cutlass::arch::Sm80>::Params params);

template<>
__global__ void DualKernel<cutlass::gemm::kernel::DualGemm<cutlass::half_t, cutlass::half_t, false, cutlass::epilogue::thread::GELU_taylor, cutlass::arch::Sm80>>(typename cutlass::gemm::kernel::DualGemm<cutlass::half_t, cutlass::half_t, false, cutlass::epilogue::thread::GELU_taylor, cutlass::arch::Sm80>::Params params);

