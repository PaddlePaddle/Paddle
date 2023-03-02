// This file is auto-generated. See "generate_kernels.py"
#ifndef XFORMERS_MEM_EFF_ATTENTION_DISABLE_FORWARD
#include "../../kernel_forward.h"

template<>
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, true, false, false, false>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, true, false, false, false>::kMinBlocksPerSm)
attention_kernel_batched<AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, true, false, false, false>>(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, true, false, false, false>::Params params) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700 && __CUDA_ARCH__ < 750
  if (!params.advance_to_block()) {
    return;
  }
  AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, true, false, false, false>::attention_kernel(params);
  return;
#endif
    printf(
        "FATAL: kernel `attention_kernel_batched` is for sm70-sm75, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}

template<>
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, true, false, false, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, true, false, false, true>::kMinBlocksPerSm)
attention_kernel_batched<AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, true, false, false, true>>(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, true, false, false, true>::Params params) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700 && __CUDA_ARCH__ < 750
  if (!params.advance_to_block()) {
    return;
  }
  AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, true, false, false, true>::attention_kernel(params);
  return;
#endif
    printf(
        "FATAL: kernel `attention_kernel_batched` is for sm70-sm75, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}

template<>
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, true, false, true, false>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, true, false, true, false>::kMinBlocksPerSm)
attention_kernel_batched<AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, true, false, true, false>>(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, true, false, true, false>::Params params) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700 && __CUDA_ARCH__ < 750
  if (!params.advance_to_block()) {
    return;
  }
  AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, true, false, true, false>::attention_kernel(params);
  return;
#endif
    printf(
        "FATAL: kernel `attention_kernel_batched` is for sm70-sm75, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}

template<>
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, true, false, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, true, false, true, true>::kMinBlocksPerSm)
attention_kernel_batched<AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, true, false, true, true>>(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, true, false, true, true>::Params params) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700 && __CUDA_ARCH__ < 750
  if (!params.advance_to_block()) {
    return;
  }
  AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, true, false, true, true>::attention_kernel(params);
  return;
#endif
    printf(
        "FATAL: kernel `attention_kernel_batched` is for sm70-sm75, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}

template<>
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, true, true, false, false>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, true, true, false, false>::kMinBlocksPerSm)
attention_kernel_batched<AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, true, true, false, false>>(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, true, true, false, false>::Params params) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700 && __CUDA_ARCH__ < 750
  if (!params.advance_to_block()) {
    return;
  }
  AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, true, true, false, false>::attention_kernel(params);
  return;
#endif
    printf(
        "FATAL: kernel `attention_kernel_batched` is for sm70-sm75, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}

template<>
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, true, true, false, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, true, true, false, true>::kMinBlocksPerSm)
attention_kernel_batched<AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, true, true, false, true>>(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, true, true, false, true>::Params params) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700 && __CUDA_ARCH__ < 750
  if (!params.advance_to_block()) {
    return;
  }
  AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, true, true, false, true>::attention_kernel(params);
  return;
#endif
    printf(
        "FATAL: kernel `attention_kernel_batched` is for sm70-sm75, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}

template<>
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, true, true, true, false>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, true, true, true, false>::kMinBlocksPerSm)
attention_kernel_batched<AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, true, true, true, false>>(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, true, true, true, false>::Params params) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700 && __CUDA_ARCH__ < 750
  if (!params.advance_to_block()) {
    return;
  }
  AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, true, true, true, false>::attention_kernel(params);
  return;
#endif
    printf(
        "FATAL: kernel `attention_kernel_batched` is for sm70-sm75, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}

template<>
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, true, true, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, true, true, true, true>::kMinBlocksPerSm)
attention_kernel_batched<AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, true, true, true, true>>(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, true, true, true, true>::Params params) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700 && __CUDA_ARCH__ < 750
  if (!params.advance_to_block()) {
    return;
  }
  AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, true, true, true, true>::attention_kernel(params);
  return;
#endif
    printf(
        "FATAL: kernel `attention_kernel_batched` is for sm70-sm75, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}

template<>
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, true, false, false, false>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, true, false, false, false>::kMinBlocksPerSm)
attention_kernel_batched<AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, true, false, false, false>>(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, true, false, false, false>::Params params) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700 && __CUDA_ARCH__ < 750
  if (!params.advance_to_block()) {
    return;
  }
  AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, true, false, false, false>::attention_kernel(params);
  return;
#endif
    printf(
        "FATAL: kernel `attention_kernel_batched` is for sm70-sm75, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}

template<>
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, true, false, false, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, true, false, false, true>::kMinBlocksPerSm)
attention_kernel_batched<AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, true, false, false, true>>(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, true, false, false, true>::Params params) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700 && __CUDA_ARCH__ < 750
  if (!params.advance_to_block()) {
    return;
  }
  AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, true, false, false, true>::attention_kernel(params);
  return;
#endif
    printf(
        "FATAL: kernel `attention_kernel_batched` is for sm70-sm75, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}

template<>
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, true, false, true, false>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, true, false, true, false>::kMinBlocksPerSm)
attention_kernel_batched<AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, true, false, true, false>>(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, true, false, true, false>::Params params) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700 && __CUDA_ARCH__ < 750
  if (!params.advance_to_block()) {
    return;
  }
  AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, true, false, true, false>::attention_kernel(params);
  return;
#endif
    printf(
        "FATAL: kernel `attention_kernel_batched` is for sm70-sm75, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}

template<>
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, true, false, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, true, false, true, true>::kMinBlocksPerSm)
attention_kernel_batched<AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, true, false, true, true>>(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, true, false, true, true>::Params params) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700 && __CUDA_ARCH__ < 750
  if (!params.advance_to_block()) {
    return;
  }
  AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, true, false, true, true>::attention_kernel(params);
  return;
#endif
    printf(
        "FATAL: kernel `attention_kernel_batched` is for sm70-sm75, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}

template<>
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, true, true, false, false>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, true, true, false, false>::kMinBlocksPerSm)
attention_kernel_batched<AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, true, true, false, false>>(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, true, true, false, false>::Params params) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700 && __CUDA_ARCH__ < 750
  if (!params.advance_to_block()) {
    return;
  }
  AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, true, true, false, false>::attention_kernel(params);
  return;
#endif
    printf(
        "FATAL: kernel `attention_kernel_batched` is for sm70-sm75, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}

template<>
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, true, true, false, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, true, true, false, true>::kMinBlocksPerSm)
attention_kernel_batched<AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, true, true, false, true>>(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, true, true, false, true>::Params params) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700 && __CUDA_ARCH__ < 750
  if (!params.advance_to_block()) {
    return;
  }
  AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, true, true, false, true>::attention_kernel(params);
  return;
#endif
    printf(
        "FATAL: kernel `attention_kernel_batched` is for sm70-sm75, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}

template<>
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, true, true, true, false>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, true, true, true, false>::kMinBlocksPerSm)
attention_kernel_batched<AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, true, true, true, false>>(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, true, true, true, false>::Params params) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700 && __CUDA_ARCH__ < 750
  if (!params.advance_to_block()) {
    return;
  }
  AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, true, true, true, false>::attention_kernel(params);
  return;
#endif
    printf(
        "FATAL: kernel `attention_kernel_batched` is for sm70-sm75, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}

template<>
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, true, true, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, true, true, true, true>::kMinBlocksPerSm)
attention_kernel_batched<AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, true, true, true, true>>(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, true, true, true, true>::Params params) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700 && __CUDA_ARCH__ < 750
  if (!params.advance_to_block()) {
    return;
  }
  AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, true, true, true, true>::attention_kernel(params);
  return;
#endif
    printf(
        "FATAL: kernel `attention_kernel_batched` is for sm70-sm75, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}

template<>
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, false, false, false, false>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, false, false, false, false>::kMinBlocksPerSm)
attention_kernel_batched<AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, false, false, false, false>>(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, false, false, false, false>::Params params) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700 && __CUDA_ARCH__ < 750
  if (!params.advance_to_block()) {
    return;
  }
  AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, false, false, false, false>::attention_kernel(params);
  return;
#endif
    printf(
        "FATAL: kernel `attention_kernel_batched` is for sm70-sm75, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}

template<>
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, false, false, false, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, false, false, false, true>::kMinBlocksPerSm)
attention_kernel_batched<AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, false, false, false, true>>(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, false, false, false, true>::Params params) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700 && __CUDA_ARCH__ < 750
  if (!params.advance_to_block()) {
    return;
  }
  AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, false, false, false, true>::attention_kernel(params);
  return;
#endif
    printf(
        "FATAL: kernel `attention_kernel_batched` is for sm70-sm75, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}

template<>
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, false, false, true, false>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, false, false, true, false>::kMinBlocksPerSm)
attention_kernel_batched<AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, false, false, true, false>>(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, false, false, true, false>::Params params) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700 && __CUDA_ARCH__ < 750
  if (!params.advance_to_block()) {
    return;
  }
  AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, false, false, true, false>::attention_kernel(params);
  return;
#endif
    printf(
        "FATAL: kernel `attention_kernel_batched` is for sm70-sm75, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}

template<>
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, false, false, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, false, false, true, true>::kMinBlocksPerSm)
attention_kernel_batched<AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, false, false, true, true>>(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, false, false, true, true>::Params params) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700 && __CUDA_ARCH__ < 750
  if (!params.advance_to_block()) {
    return;
  }
  AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, false, false, true, true>::attention_kernel(params);
  return;
#endif
    printf(
        "FATAL: kernel `attention_kernel_batched` is for sm70-sm75, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}

template<>
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, false, true, false, false>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, false, true, false, false>::kMinBlocksPerSm)
attention_kernel_batched<AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, false, true, false, false>>(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, false, true, false, false>::Params params) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700 && __CUDA_ARCH__ < 750
  if (!params.advance_to_block()) {
    return;
  }
  AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, false, true, false, false>::attention_kernel(params);
  return;
#endif
    printf(
        "FATAL: kernel `attention_kernel_batched` is for sm70-sm75, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}

template<>
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, false, true, false, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, false, true, false, true>::kMinBlocksPerSm)
attention_kernel_batched<AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, false, true, false, true>>(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, false, true, false, true>::Params params) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700 && __CUDA_ARCH__ < 750
  if (!params.advance_to_block()) {
    return;
  }
  AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, false, true, false, true>::attention_kernel(params);
  return;
#endif
    printf(
        "FATAL: kernel `attention_kernel_batched` is for sm70-sm75, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}

template<>
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, false, true, true, false>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, false, true, true, false>::kMinBlocksPerSm)
attention_kernel_batched<AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, false, true, true, false>>(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, false, true, true, false>::Params params) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700 && __CUDA_ARCH__ < 750
  if (!params.advance_to_block()) {
    return;
  }
  AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, false, true, true, false>::attention_kernel(params);
  return;
#endif
    printf(
        "FATAL: kernel `attention_kernel_batched` is for sm70-sm75, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}

template<>
__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, false, true, true, true>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, false, true, true, true>::kMinBlocksPerSm)
attention_kernel_batched<AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, false, true, true, true>>(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, false, true, true, true>::Params params) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700 && __CUDA_ARCH__ < 750
  if (!params.advance_to_block()) {
    return;
  }
  AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, false, true, true, true>::attention_kernel(params);
  return;
#endif
    printf(
        "FATAL: kernel `attention_kernel_batched` is for sm70-sm75, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}
#endif // XFORMERS_MEM_EFF_ATTENTION_DISABLE_FORWARD
