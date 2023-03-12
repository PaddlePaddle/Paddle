// This file is auto-generated. See "generate_kernels.py"
#pragma once
#include "../kernel/dual_gemm.h"

// ======== f16 / sm70 ========

template<>
__global__ void DualKernel<cutlass::gemm::kernel::DualGemm<cutlass::half_t, 
                                                           float, 
                                                           false, 
                                                           cutlass::epilogue::thread::SiLu, 
                                                           cutlass::arch::Sm80>>(
    cutlass::gemm::kernel::DualGemm<cutlass::half_t, float, false, 
    cutlass::epilogue::thread::SiLu, cutlass::arch::Sm80>::Params params);
