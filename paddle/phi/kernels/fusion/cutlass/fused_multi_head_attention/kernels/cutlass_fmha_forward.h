// This file is auto-generated. See "generate_kernels.py"
#pragma once
#ifndef XFORMERS_MEM_EFF_ATTENTION_DISABLE_FORWARD
#include "../kernel_forward.h"

// ======== f16 / sm70 ========

template<>
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, true, false, false>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, true, false, false>::kMinBlocksPerSm)
attention_kernel_batched<AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, true, false, false>>(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, true, false, false>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, true, false, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, true, false, true>::kMinBlocksPerSm)
attention_kernel_batched<AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, true, false, true>>(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, true, false, true>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, true, true, false>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, true, true, false>::kMinBlocksPerSm)
attention_kernel_batched<AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, true, true, false>>(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, true, true, false>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, true, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, true, true, true>::kMinBlocksPerSm)
attention_kernel_batched<AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, true, true, true>>(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, true, true, true>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, true, false, false>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, true, false, false>::kMinBlocksPerSm)
attention_kernel_batched<AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, true, false, false>>(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, true, false, false>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, true, false, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, true, false, true>::kMinBlocksPerSm)
attention_kernel_batched<AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, true, false, true>>(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, true, false, true>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, true, true, false>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, true, true, false>::kMinBlocksPerSm)
attention_kernel_batched<AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, true, true, false>>(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, true, true, false>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, true, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, true, true, true>::kMinBlocksPerSm)
attention_kernel_batched<AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, true, true, true>>(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, true, true, true>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, false, false, false>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, false, false, false>::kMinBlocksPerSm)
attention_kernel_batched<AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, false, false, false>>(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, false, false, false>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, false, false, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, false, false, true>::kMinBlocksPerSm)
attention_kernel_batched<AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, false, false, true>>(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, false, false, true>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, false, true, false>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, false, true, false>::kMinBlocksPerSm)
attention_kernel_batched<AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, false, true, false>>(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, false, true, false>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, false, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, false, true, true>::kMinBlocksPerSm)
attention_kernel_batched<AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, false, true, true>>(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, false, true, true>::Params params);


// ======== f16 / sm75 ========

template<>
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, true, false, false>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, true, false, false>::kMinBlocksPerSm)
attention_kernel_batched<AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, true, false, false>>(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, true, false, false>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, true, false, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, true, false, true>::kMinBlocksPerSm)
attention_kernel_batched<AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, true, false, true>>(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, true, false, true>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, true, true, false>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, true, true, false>::kMinBlocksPerSm)
attention_kernel_batched<AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, true, true, false>>(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, true, true, false>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, true, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, true, true, true>::kMinBlocksPerSm)
attention_kernel_batched<AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, true, true, true>>(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, true, true, true>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, true, false, false>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, true, false, false>::kMinBlocksPerSm)
attention_kernel_batched<AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, true, false, false>>(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, true, false, false>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, true, false, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, true, false, true>::kMinBlocksPerSm)
attention_kernel_batched<AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, true, false, true>>(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, true, false, true>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, true, true, false>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, true, true, false>::kMinBlocksPerSm)
attention_kernel_batched<AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, true, true, false>>(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, true, true, false>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, true, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, true, true, true>::kMinBlocksPerSm)
attention_kernel_batched<AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, true, true, true>>(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, true, true, true>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, false, false, false>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, false, false, false>::kMinBlocksPerSm)
attention_kernel_batched<AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, false, false, false>>(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, false, false, false>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, false, false, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, false, false, true>::kMinBlocksPerSm)
attention_kernel_batched<AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, false, false, true>>(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, false, false, true>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, false, true, false>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, false, true, false>::kMinBlocksPerSm)
attention_kernel_batched<AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, false, true, false>>(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, false, true, false>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, false, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, false, true, true>::kMinBlocksPerSm)
attention_kernel_batched<AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, false, true, true>>(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, false, true, true>::Params params);


// ======== f16 / sm80 ========

template<>
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, true, false, false>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, true, false, false>::kMinBlocksPerSm)
attention_kernel_batched<AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, true, false, false>>(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, true, false, false>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, true, false, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, true, false, true>::kMinBlocksPerSm)
attention_kernel_batched<AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, true, false, true>>(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, true, false, true>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, true, true, false>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, true, true, false>::kMinBlocksPerSm)
attention_kernel_batched<AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, true, true, false>>(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, true, true, false>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, true, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, true, true, true>::kMinBlocksPerSm)
attention_kernel_batched<AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, true, true, true>>(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, true, true, true>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, true, false, false>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, true, false, false>::kMinBlocksPerSm)
attention_kernel_batched<AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, true, false, false>>(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, true, false, false>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, true, false, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, true, false, true>::kMinBlocksPerSm)
attention_kernel_batched<AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, true, false, true>>(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, true, false, true>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, true, true, false>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, true, true, false>::kMinBlocksPerSm)
attention_kernel_batched<AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, true, true, false>>(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, true, true, false>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, true, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, true, true, true>::kMinBlocksPerSm)
attention_kernel_batched<AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, true, true, true>>(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, true, true, true>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, false, false, false>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, false, false, false>::kMinBlocksPerSm)
attention_kernel_batched<AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, false, false, false>>(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, false, false, false>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, false, false, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, false, false, true>::kMinBlocksPerSm)
attention_kernel_batched<AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, false, false, true>>(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, false, false, true>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, false, true, false>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, false, true, false>::kMinBlocksPerSm)
attention_kernel_batched<AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, false, true, false>>(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, false, true, false>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, false, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, false, true, true>::kMinBlocksPerSm)
attention_kernel_batched<AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, false, true, true>>(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, false, true, true>::Params params);

#endif // XFORMERS_MEM_EFF_ATTENTION_DISABLE_FORWARD
