#pragma once

// All kernels are disabled by default
#define INSTANTIATE_ATTENTION_KERNEL_FORWARD_SM70(...) \
  INSTANTIATE_ATTENTION_KERNEL_FORWARD_DISABLED(70, __VA_ARGS__)
#define INSTANTIATE_ATTENTION_KERNEL_FORWARD_SM75(...) \
  INSTANTIATE_ATTENTION_KERNEL_FORWARD_DISABLED(75, __VA_ARGS__)
#define INSTANTIATE_ATTENTION_KERNEL_FORWARD_SM80(...) \
  INSTANTIATE_ATTENTION_KERNEL_FORWARD_DISABLED(80, __VA_ARGS__)

#include "../kernel_forward.h"

#define _ATTENTION_KERNEL_FORWARD_BEGIN(...)                                  \
  template <>                                                                 \
  __global__ void __launch_bounds__(                                          \
      __VA_ARGS__::kNumThreads, __VA_ARGS__::kMinBlocksPerSm)                 \
      attention_kernel_batched<__VA_ARGS__>(typename __VA_ARGS__::Params p) { \
    using Kernel = __VA_ARGS__;
#define _ATTENTION_KERNEL_FORWARD_END() }

#ifdef __CUDA_ARCH__
#define __CUDA_ARCH_OR_ZERO__ __CUDA_ARCH__
#else
#define __CUDA_ARCH_OR_ZERO__ 0
#endif

#define INSTANTIATE_ATTENTION_KERNEL_FORWARD(              \
    ARCH,                                                  \
    SCALAR_T,                                              \
    IS_ALIGNED,                                            \
    QUERIES_PER_BLOCK,                                     \
    KEYS_PER_BLOCK,                                        \
    SINGLE_VALUE_ITER, \
    ADD_MASK, \
    MASK_BROADCAST_ROW)                                     \
  _ATTENTION_KERNEL_FORWARD_BEGIN(AttentionKernel<         \
                                  SCALAR_T,                \
                                  cutlass::arch::Sm##ARCH, \
                                  IS_ALIGNED,              \
                                  QUERIES_PER_BLOCK,       \
                                  KEYS_PER_BLOCK,          \
                                  SINGLE_VALUE_ITER, \
                                  ADD_MASK, \
                                  MASK_BROADCAST_ROW>)      \
  if (!p.advance_to_block()) {                             \
    return;                                                \
  }                                                        \
  Kernel::attention_kernel(p);                             \
  _ATTENTION_KERNEL_FORWARD_END();

#define INSTANTIATE_ATTENTION_KERNEL_FORWARD_DISABLED(              \
    ARCH,                                                           \
    SCALAR_T,                                                       \
    IS_ALIGNED,                                                     \
    QUERIES_PER_BLOCK,                                              \
    KEYS_PER_BLOCK,                                                 \
    SINGLE_VALUE_ITER, \
    ADD_MASK, \
    MASK_BROADCAST_ROW)                                              \
  _ATTENTION_KERNEL_FORWARD_BEGIN(AttentionKernel<                  \
                                  SCALAR_T,                         \
                                  cutlass::arch::Sm##ARCH,          \
                                  IS_ALIGNED,                       \
                                  QUERIES_PER_BLOCK,                \
                                  KEYS_PER_BLOCK,                   \
                                  SINGLE_VALUE_ITER, \
                                  ADD_MASK, \
                                  MASK_BROADCAST_ROW>)      \
  printf(                                                           \
      "FATAL: this function is for sm%d, but was built for sm%d\n", \
      int(ARCH),                                                    \
      int(__CUDA_ARCH_OR_ZERO__));                                  \
  _ATTENTION_KERNEL_FORWARD_END();

// Enable the right one based on __CUDA_ARCH__
#ifndef __CUDA_ARCH__
#elif __CUDA_ARCH__ < 700
#error "Need cuda arch at least 5.0"
#elif __CUDA_ARCH__ < 750
#undef INSTANTIATE_ATTENTION_KERNEL_FORWARD_SM70
#define INSTANTIATE_ATTENTION_KERNEL_FORWARD_SM70(...) \
  INSTANTIATE_ATTENTION_KERNEL_FORWARD(70, __VA_ARGS__)
#elif __CUDA_ARCH__ < 800
#undef INSTANTIATE_ATTENTION_KERNEL_FORWARD_SM75
#define INSTANTIATE_ATTENTION_KERNEL_FORWARD_SM75(...) \
  INSTANTIATE_ATTENTION_KERNEL_FORWARD(75, __VA_ARGS__)
#elif __CUDA_ARCH__ >= 800
#undef INSTANTIATE_ATTENTION_KERNEL_FORWARD_SM80
#define INSTANTIATE_ATTENTION_KERNEL_FORWARD_SM80(...) \
  INSTANTIATE_ATTENTION_KERNEL_FORWARD(80, __VA_ARGS__)
#endif
