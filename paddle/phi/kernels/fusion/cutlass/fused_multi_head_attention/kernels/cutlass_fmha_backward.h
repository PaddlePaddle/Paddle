// This file is auto-generated. See "generate_kernels.py"
#pragma once
#ifndef XFORMERS_MEM_EFF_ATTENTION_DISABLE_BACKWARD
#include "../kernel_backward.h"

// ======== f16 / sm70 ========

template<>
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 128, 64, 1024>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 128, 64, 1024>::kMinBlocksPerSm)
attention_kernel_backward_batched<AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 128, 64, 1024>>(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, false, 128, 64, 1024>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, true, 128, 64, 1024>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, true, 128, 64, 1024>::kMinBlocksPerSm)
attention_kernel_backward_batched<AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, true, 128, 64, 1024>>(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, false, true, 128, 64, 1024>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 128, 64, 1024>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 128, 64, 1024>::kMinBlocksPerSm)
attention_kernel_backward_batched<AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 128, 64, 1024>>(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, false, 128, 64, 1024>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, true, 128, 64, 1024>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, true, 128, 64, 1024>::kMinBlocksPerSm)
attention_kernel_backward_batched<AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, true, 128, 64, 1024>>(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, true, true, true, 128, 64, 1024>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, false, 128, 64, 1024>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, false, 128, 64, 1024>::kMinBlocksPerSm)
attention_kernel_backward_batched<AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, false, 128, 64, 1024>>(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, false, 128, 64, 1024>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, true, 128, 64, 1024>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, true, 128, 64, 1024>::kMinBlocksPerSm)
attention_kernel_backward_batched<AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, true, 128, 64, 1024>>(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, false, true, 128, 64, 1024>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, false, 128, 64, 1024>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, false, 128, 64, 1024>::kMinBlocksPerSm)
attention_kernel_backward_batched<AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, false, 128, 64, 1024>>(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, false, 128, 64, 1024>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, true, 128, 64, 1024>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, true, 128, 64, 1024>::kMinBlocksPerSm)
attention_kernel_backward_batched<AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, true, 128, 64, 1024>>(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, true, 128, 64, 1024>::Params params);


// ======== f32 / sm70 ========

template<>
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, float, true, false, false, 128, 64, 1024>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, float, true, false, false, 128, 64, 1024>::kMinBlocksPerSm)
attention_kernel_backward_batched<AttentionBackwardKernel<cutlass::arch::Sm70, float, true, false, false, 128, 64, 1024>>(typename AttentionBackwardKernel<cutlass::arch::Sm70, float, true, false, false, 128, 64, 1024>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, float, true, false, true, 128, 64, 1024>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, float, true, false, true, 128, 64, 1024>::kMinBlocksPerSm)
attention_kernel_backward_batched<AttentionBackwardKernel<cutlass::arch::Sm70, float, true, false, true, 128, 64, 1024>>(typename AttentionBackwardKernel<cutlass::arch::Sm70, float, true, false, true, 128, 64, 1024>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, float, true, true, false, 128, 64, 1024>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, float, true, true, false, 128, 64, 1024>::kMinBlocksPerSm)
attention_kernel_backward_batched<AttentionBackwardKernel<cutlass::arch::Sm70, float, true, true, false, 128, 64, 1024>>(typename AttentionBackwardKernel<cutlass::arch::Sm70, float, true, true, false, 128, 64, 1024>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, float, true, true, true, 128, 64, 1024>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, float, true, true, true, 128, 64, 1024>::kMinBlocksPerSm)
attention_kernel_backward_batched<AttentionBackwardKernel<cutlass::arch::Sm70, float, true, true, true, 128, 64, 1024>>(typename AttentionBackwardKernel<cutlass::arch::Sm70, float, true, true, true, 128, 64, 1024>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, float, false, false, false, 128, 64, 1024>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, float, false, false, false, 128, 64, 1024>::kMinBlocksPerSm)
attention_kernel_backward_batched<AttentionBackwardKernel<cutlass::arch::Sm70, float, false, false, false, 128, 64, 1024>>(typename AttentionBackwardKernel<cutlass::arch::Sm70, float, false, false, false, 128, 64, 1024>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, float, false, false, true, 128, 64, 1024>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, float, false, false, true, 128, 64, 1024>::kMinBlocksPerSm)
attention_kernel_backward_batched<AttentionBackwardKernel<cutlass::arch::Sm70, float, false, false, true, 128, 64, 1024>>(typename AttentionBackwardKernel<cutlass::arch::Sm70, float, false, false, true, 128, 64, 1024>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, float, false, true, false, 128, 64, 1024>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, float, false, true, false, 128, 64, 1024>::kMinBlocksPerSm)
attention_kernel_backward_batched<AttentionBackwardKernel<cutlass::arch::Sm70, float, false, true, false, 128, 64, 1024>>(typename AttentionBackwardKernel<cutlass::arch::Sm70, float, false, true, false, 128, 64, 1024>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, float, false, true, true, 128, 64, 1024>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, float, false, true, true, 128, 64, 1024>::kMinBlocksPerSm)
attention_kernel_backward_batched<AttentionBackwardKernel<cutlass::arch::Sm70, float, false, true, true, 128, 64, 1024>>(typename AttentionBackwardKernel<cutlass::arch::Sm70, float, false, true, true, 128, 64, 1024>::Params params);


// ======== f16 / sm75 ========

template<>
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, false, false, 128, 64, 1024>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, false, false, 128, 64, 1024>::kMinBlocksPerSm)
attention_kernel_backward_batched<AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, false, false, 128, 64, 1024>>(typename AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, false, false, 128, 64, 1024>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, false, true, 128, 64, 1024>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, false, true, 128, 64, 1024>::kMinBlocksPerSm)
attention_kernel_backward_batched<AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, false, true, 128, 64, 1024>>(typename AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, false, true, 128, 64, 1024>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, true, false, 128, 64, 1024>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, true, false, 128, 64, 1024>::kMinBlocksPerSm)
attention_kernel_backward_batched<AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, true, false, 128, 64, 1024>>(typename AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, true, false, 128, 64, 1024>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, true, true, 128, 64, 1024>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, true, true, 128, 64, 1024>::kMinBlocksPerSm)
attention_kernel_backward_batched<AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, true, true, 128, 64, 1024>>(typename AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, true, true, true, 128, 64, 1024>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, false, false, 128, 64, 1024>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, false, false, 128, 64, 1024>::kMinBlocksPerSm)
attention_kernel_backward_batched<AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, false, false, 128, 64, 1024>>(typename AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, false, false, 128, 64, 1024>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, false, true, 128, 64, 1024>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, false, true, 128, 64, 1024>::kMinBlocksPerSm)
attention_kernel_backward_batched<AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, false, true, 128, 64, 1024>>(typename AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, false, true, 128, 64, 1024>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, true, false, 128, 64, 1024>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, true, false, 128, 64, 1024>::kMinBlocksPerSm)
attention_kernel_backward_batched<AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, true, false, 128, 64, 1024>>(typename AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, true, false, 128, 64, 1024>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, true, true, 128, 64, 1024>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, true, true, 128, 64, 1024>::kMinBlocksPerSm)
attention_kernel_backward_batched<AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, true, true, 128, 64, 1024>>(typename AttentionBackwardKernel<cutlass::arch::Sm75, cutlass::half_t, false, true, true, 128, 64, 1024>::Params params);


// ======== f32 / sm75 ========

template<>
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, float, true, false, false, 128, 64, 1024>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm75, float, true, false, false, 128, 64, 1024>::kMinBlocksPerSm)
attention_kernel_backward_batched<AttentionBackwardKernel<cutlass::arch::Sm75, float, true, false, false, 128, 64, 1024>>(typename AttentionBackwardKernel<cutlass::arch::Sm75, float, true, false, false, 128, 64, 1024>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, float, true, false, true, 128, 64, 1024>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm75, float, true, false, true, 128, 64, 1024>::kMinBlocksPerSm)
attention_kernel_backward_batched<AttentionBackwardKernel<cutlass::arch::Sm75, float, true, false, true, 128, 64, 1024>>(typename AttentionBackwardKernel<cutlass::arch::Sm75, float, true, false, true, 128, 64, 1024>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, float, true, true, false, 128, 64, 1024>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm75, float, true, true, false, 128, 64, 1024>::kMinBlocksPerSm)
attention_kernel_backward_batched<AttentionBackwardKernel<cutlass::arch::Sm75, float, true, true, false, 128, 64, 1024>>(typename AttentionBackwardKernel<cutlass::arch::Sm75, float, true, true, false, 128, 64, 1024>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, float, true, true, true, 128, 64, 1024>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm75, float, true, true, true, 128, 64, 1024>::kMinBlocksPerSm)
attention_kernel_backward_batched<AttentionBackwardKernel<cutlass::arch::Sm75, float, true, true, true, 128, 64, 1024>>(typename AttentionBackwardKernel<cutlass::arch::Sm75, float, true, true, true, 128, 64, 1024>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, float, false, false, false, 128, 64, 1024>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm75, float, false, false, false, 128, 64, 1024>::kMinBlocksPerSm)
attention_kernel_backward_batched<AttentionBackwardKernel<cutlass::arch::Sm75, float, false, false, false, 128, 64, 1024>>(typename AttentionBackwardKernel<cutlass::arch::Sm75, float, false, false, false, 128, 64, 1024>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, float, false, false, true, 128, 64, 1024>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm75, float, false, false, true, 128, 64, 1024>::kMinBlocksPerSm)
attention_kernel_backward_batched<AttentionBackwardKernel<cutlass::arch::Sm75, float, false, false, true, 128, 64, 1024>>(typename AttentionBackwardKernel<cutlass::arch::Sm75, float, false, false, true, 128, 64, 1024>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, float, false, true, false, 128, 64, 1024>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm75, float, false, true, false, 128, 64, 1024>::kMinBlocksPerSm)
attention_kernel_backward_batched<AttentionBackwardKernel<cutlass::arch::Sm75, float, false, true, false, 128, 64, 1024>>(typename AttentionBackwardKernel<cutlass::arch::Sm75, float, false, true, false, 128, 64, 1024>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm75, float, false, true, true, 128, 64, 1024>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm75, float, false, true, true, 128, 64, 1024>::kMinBlocksPerSm)
attention_kernel_backward_batched<AttentionBackwardKernel<cutlass::arch::Sm75, float, false, true, true, 128, 64, 1024>>(typename AttentionBackwardKernel<cutlass::arch::Sm75, float, false, true, true, 128, 64, 1024>::Params params);


// ======== bf16 / sm80 ========

template<>
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 128, 64, 1024>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 128, 64, 1024>::kMinBlocksPerSm)
attention_kernel_backward_batched<AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 128, 64, 1024>>(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, false, 128, 64, 1024>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 64, 1024>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 64, 1024>::kMinBlocksPerSm)
attention_kernel_backward_batched<AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 64, 1024>>(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, false, true, 128, 64, 1024>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, false, 128, 64, 1024>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, false, 128, 64, 1024>::kMinBlocksPerSm)
attention_kernel_backward_batched<AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, false, 128, 64, 1024>>(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, false, 128, 64, 1024>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 128, 64, 1024>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 128, 64, 1024>::kMinBlocksPerSm)
attention_kernel_backward_batched<AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 128, 64, 1024>>(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::bfloat16_t, true, true, true, 128, 64, 1024>::Params params);


// ======== f16 / sm80 ========

template<>
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 128, 64, 1024>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 128, 64, 1024>::kMinBlocksPerSm)
attention_kernel_backward_batched<AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 128, 64, 1024>>(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, false, 128, 64, 1024>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 64, 1024>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 64, 1024>::kMinBlocksPerSm)
attention_kernel_backward_batched<AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 64, 1024>>(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, false, true, 128, 64, 1024>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 128, 64, 1024>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 128, 64, 1024>::kMinBlocksPerSm)
attention_kernel_backward_batched<AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 128, 64, 1024>>(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, false, 128, 64, 1024>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 128, 64, 1024>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 128, 64, 1024>::kMinBlocksPerSm)
attention_kernel_backward_batched<AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 128, 64, 1024>>(typename AttentionBackwardKernel<cutlass::arch::Sm80, cutlass::half_t, true, true, true, 128, 64, 1024>::Params params);


// ======== f32 / sm80 ========

template<>
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 128, 64, 1024>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 128, 64, 1024>::kMinBlocksPerSm)
attention_kernel_backward_batched<AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 128, 64, 1024>>(typename AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, false, 128, 64, 1024>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, true, 128, 64, 1024>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, true, 128, 64, 1024>::kMinBlocksPerSm)
attention_kernel_backward_batched<AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, true, 128, 64, 1024>>(typename AttentionBackwardKernel<cutlass::arch::Sm80, float, true, false, true, 128, 64, 1024>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 128, 64, 1024>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 128, 64, 1024>::kMinBlocksPerSm)
attention_kernel_backward_batched<AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 128, 64, 1024>>(typename AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, false, 128, 64, 1024>::Params params);

template<>
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, true, 128, 64, 1024>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, true, 128, 64, 1024>::kMinBlocksPerSm)
attention_kernel_backward_batched<AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, true, 128, 64, 1024>>(typename AttentionBackwardKernel<cutlass::arch::Sm80, float, true, true, true, 128, 64, 1024>::Params params);

#endif // XFORMERS_MEM_EFF_ATTENTION_DISABLE_BACKWARD
