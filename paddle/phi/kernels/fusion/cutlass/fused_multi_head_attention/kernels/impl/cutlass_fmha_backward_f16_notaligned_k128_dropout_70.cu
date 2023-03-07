// This file is auto-generated. See "generate_kernels.py"
#ifndef XFORMERS_MEM_EFF_ATTENTION_DISABLE_BACKWARD
#include "../../kernel_backward.h"

template<>
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, false, 64, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, false, 64, 64, 128>::kMinBlocksPerSm)
attention_kernel_backward_batched<AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, false, 64, 64, 128>>(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, false, 64, 64, 128>::Params params) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700 && __CUDA_ARCH__ < 750
  if (!params.advance_to_block()) {
    return;
  }
  AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, false, 64, 64, 128>::attention_kernel(params);
  return;
#endif
    printf(
        "FATAL: kernel `attention_kernel_backward_batched` is for sm70-sm75, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}

template<>
__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, true, 64, 64, 128>::kNumThreads,
    AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, true, 64, 64, 128>::kMinBlocksPerSm)
attention_kernel_backward_batched<AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, true, 64, 64, 128>>(typename AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, true, 64, 64, 128>::Params params) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700 && __CUDA_ARCH__ < 750
  if (!params.advance_to_block()) {
    return;
  }
  AttentionBackwardKernel<cutlass::arch::Sm70, cutlass::half_t, false, true, true, 64, 64, 128>::attention_kernel(params);
  return;
#endif
    printf(
        "FATAL: kernel `attention_kernel_backward_batched` is for sm70-sm75, but was built for sm%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}
#endif // XFORMERS_MEM_EFF_ATTENTION_DISABLE_BACKWARD
