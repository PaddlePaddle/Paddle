// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/kernels/sparse/unary_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/sparse_coo_tensor.h"
#include "paddle/phi/core/sparse_csr_tensor.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/activation_kernel.h"
#include "paddle/phi/kernels/empty_kernel.h"

#define DEFINE_SPARSE_UNARY_KERNEL(DenseKernelFunc)                      \
  namespace phi {                                                        \
  namespace sparse {                                                     \
                                                                         \
  template <typename T, typename Context>                                \
  void SparseCoo##DenseKernelFunc(const Context& dev_ctx,                \
                                  const SparseCooTensor& x,              \
                                  SparseCooTensor* out) {                \
    DenseTensor non_zero_indices =                                       \
        phi::EmptyLike<T, Context>(dev_ctx, x.non_zero_indices());       \
    DenseTensor non_zero_elements =                                      \
        phi::EmptyLike<T, Context>(dev_ctx, x.non_zero_elements());      \
    phi::Copy(dev_ctx,                                                   \
              x.non_zero_indices(),                                      \
              dev_ctx.GetPlace(),                                        \
              false,                                                     \
              &non_zero_indices);                                        \
    phi::DenseKernelFunc<T, Context>(                                    \
        dev_ctx, x.non_zero_elements(), &non_zero_elements);             \
    out->SetMember(non_zero_indices, non_zero_elements, x.dims(), true); \
  }                                                                      \
                                                                         \
  template <typename T, typename Context>                                \
  void SparseCsr##DenseKernelFunc(const Context& dev_ctx,                \
                                  const SparseCsrTensor& x,              \
                                  SparseCsrTensor* out) {                \
    DenseTensor non_zero_crows =                                         \
        phi::EmptyLike<T, Context>(dev_ctx, x.non_zero_crows());         \
    DenseTensor non_zero_cols =                                          \
        phi::EmptyLike<T, Context>(dev_ctx, x.non_zero_cols());          \
    DenseTensor non_zero_elements =                                      \
        phi::EmptyLike<T, Context>(dev_ctx, x.non_zero_elements());      \
    phi::Copy(dev_ctx,                                                   \
              x.non_zero_crows(),                                        \
              dev_ctx.GetPlace(),                                        \
              false,                                                     \
              &non_zero_crows);                                          \
    phi::Copy(dev_ctx,                                                   \
              x.non_zero_cols(),                                         \
              dev_ctx.GetPlace(),                                        \
              false,                                                     \
              &non_zero_cols);                                           \
    phi::DenseKernelFunc<T, Context>(                                    \
        dev_ctx, x.non_zero_elements(), &non_zero_elements);             \
    out->SetMember(                                                      \
        non_zero_crows, non_zero_cols, non_zero_elements, x.dims());     \
  }                                                                      \
  }                                                                      \
  }

#define REGISTER_CPU_SPARSE_UNARY_KERNEL(kernel_name, DenseKernelFunc) \
  PD_REGISTER_KERNEL(sparse_coo_##kernel_name,                         \
                     CPU,                                              \
                     ALL_LAYOUT,                                       \
                     phi::sparse::SparseCoo##DenseKernelFunc,          \
                     float,                                            \
                     double) {                                         \
    kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);     \
  }                                                                    \
  PD_REGISTER_KERNEL(sparse_csr_##kernel_name,                         \
                     CPU,                                              \
                     ALL_LAYOUT,                                       \
                     phi::sparse::SparseCsr##DenseKernelFunc,          \
                     float,                                            \
                     double) {                                         \
    kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);     \
  }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#define REGISTER_GPU_SPARSE_UNARY_KERNEL(kernel_name, DenseKernelFunc) \
  PD_REGISTER_KERNEL(sparse_coo_##kernel_name,                         \
                     GPU,                                              \
                     ALL_LAYOUT,                                       \
                     phi::sparse::SparseCoo##DenseKernelFunc,          \
                     float,                                            \
                     double,                                           \
                     phi::dtype::float16) {                            \
    kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);     \
  }                                                                    \
                                                                       \
  PD_REGISTER_KERNEL(sparse_csr_##kernel_name,                         \
                     GPU,                                              \
                     ALL_LAYOUT,                                       \
                     phi::sparse::SparseCsr##DenseKernelFunc,          \
                     float,                                            \
                     double,                                           \
                     phi::dtype::float16) {                            \
    kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);     \
  }
#else
// This macro definition is empty when GPU is disabled
#define REGISTER_GPU_SPARSE_UNARY_KERNEL(sparse_kernel_name, DenseKernelFunc)
#endif

#define REGISTER_SPARSE_UNARY_KERNEL(kernel_name, DenseKernelFunc) \
  REGISTER_CPU_SPARSE_UNARY_KERNEL(kernel_name, DenseKernelFunc)   \
  REGISTER_GPU_SPARSE_UNARY_KERNEL(kernel_name, DenseKernelFunc)

#define DEFINE_AND_REGISTER_SPARSE_UNARY_KERNEL(kernel_name, DenseKernelFunc) \
  DEFINE_SPARSE_UNARY_KERNEL(DenseKernelFunc)                                 \
  REGISTER_SPARSE_UNARY_KERNEL(kernel_name, DenseKernelFunc)

// NOTE: the following code is to bypass the restriction of Paddle
// kernel registration mechanism. Do NOT refactor them unless you
// know what you are doing.
// If you want to implement any new kernel, please follow `sin`,
// `tanh` etc, do NOT follow `sqrt`.
DEFINE_SPARSE_UNARY_KERNEL(SqrtKernel)

PD_REGISTER_KERNEL(sparse_coo_sqrt,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::SparseCooSqrtKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}
PD_REGISTER_KERNEL(sparse_csr_sqrt,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::SparseCsrSqrtKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(sparse_coo_sqrt,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::SparseCooSqrtKernel,
                   float,
                   double,
                   phi::dtype::float16) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

PD_REGISTER_KERNEL(sparse_csr_sqrt,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::SparseCsrSqrtKernel,
                   float,
                   double,
                   phi::dtype::float16) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}

#endif

DEFINE_AND_REGISTER_SPARSE_UNARY_KERNEL(sin, SinKernel)
DEFINE_AND_REGISTER_SPARSE_UNARY_KERNEL(tanh, TanhKernel)
DEFINE_AND_REGISTER_SPARSE_UNARY_KERNEL(relu, ReluKernel)
