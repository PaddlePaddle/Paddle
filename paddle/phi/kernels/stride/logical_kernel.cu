// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include "paddle/phi/kernels/logical_kernel.h"
#include "paddle/common/flags.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/bitwise_kernel.h"
#include "paddle/phi/kernels/funcs/logical_functor.h"
#include "paddle/phi/kernels/stride/elementwise_stride_base.cu.h"
#if defined(__NVCC__) || defined(__HIPCC__) || defined(__xpu__)
#include "paddle/phi/kernels/funcs/dims_simplifier.h"
#endif
COMMON_DECLARE_bool(use_stride_kernel);
COMMON_DECLARE_bool(use_stride_compute_kernel);
COMMON_DECLARE_bool(force_stride_compute_contig_out);
namespace phi {

template <typename T, typename Context, typename Functor>
void LaunchLogicalNotStrideKernel(const Context &dev_ctx,
                                  const DenseTensor &x,
                                  Functor func,
                                  DenseTensor *out) {
  std::vector<const DenseTensor *> inputs = {&x};
  std::vector<DenseTensor *> outputs = {out};
  dev_ctx.template Alloc<bool>(out);
  UnaryStrideElementwiseKernel<bool, Context>(dev_ctx, inputs, &outputs, func);
}

template <typename T, typename Context, typename Functor>
void LogicalKernelStrideImpl(const Context &dev_ctx,
                             const DenseTensor &x,
                             const DenseTensor &y,
                             DenseTensor *out) {
  dev_ctx.template Alloc<bool>(out);
  Functor binary_func;
  std::vector<const DenseTensor *> inputs = {&x, &y};
  std::vector<DenseTensor *> outputs = {out};
  BinaryStrideBroadcastKernel<bool, Context>(
      dev_ctx, inputs, &outputs, binary_func, -1);
}
template <typename T, typename Context, typename Functor>
void InplaceLogicalKernelStrideImpl(const Context &dev_ctx,
                                    const DenseTensor &x,
                                    const DenseTensor &y,
                                    DenseTensor *out) {
  auto x_origin = x;
  dev_ctx.template Alloc<bool>(out);
  out->set_type(phi::DataType::BOOL);
  Functor binary_func;
  std::vector<const DenseTensor *> inputs = {&x, &y};
  std::vector<DenseTensor *> outputs = {out};
  BinaryStrideBroadcastKernel<bool, Context>(
      dev_ctx, inputs, &outputs, binary_func, -1);
}

#define DEFINE_CUDA_BINARY_LOGICAL_STRIDE_OP(name)                            \
  template <typename T, typename Context>                                     \
  void Logical##name##StrideKernel(const Context &dev_ctx,                    \
                                   const DenseTensor &x,                      \
                                   const DenseTensor &y,                      \
                                   DenseTensor *out) {                        \
    if (!FLAGS_use_stride_kernel) {                                           \
      PADDLE_THROW(common::errors::Fatal(                                     \
          "FLAGS_use_stride_kernel is closed. Strided kernel "                \
          "be called, something wrong has happened!"));                       \
    }                                                                         \
    DenseTensor x_;                                                           \
    DenseTensor y_;                                                           \
    if (!FLAGS_use_stride_compute_kernel) {                                   \
      if (!x.meta().is_contiguous()) {                                        \
        x_ = Tensor2Contiguous<Context>(dev_ctx, x);                          \
      } else {                                                                \
        x_ = x;                                                               \
      }                                                                       \
      if (!y.meta().is_contiguous()) {                                        \
        y_ = Tensor2Contiguous<Context>(dev_ctx, y);                          \
      } else {                                                                \
        y_ = y;                                                               \
      }                                                                       \
    } else {                                                                  \
      x_ = x;                                                                 \
      y_ = y;                                                                 \
    }                                                                         \
    if (x_.meta().is_contiguous() && y_.meta().is_contiguous()) {             \
      auto meta = out->meta();                                                \
      meta.strides = meta.calc_strides(out->dims());                          \
      out->set_meta(meta);                                                    \
      phi::Logical##name##Kernel<T, Context>(dev_ctx, x_, y_, out);           \
      return;                                                                 \
    }                                                                         \
    if (!FLAGS_use_stride_compute_kernel) {                                   \
      PADDLE_THROW(                                                           \
          common::errors::Fatal("FLAGS_use_stride_compute_kernel is closed. " \
                                "Kernel using DenseTensorIterator "           \
                                "be called, something wrong has happened!")); \
    }                                                                         \
    if (FLAGS_force_stride_compute_contig_out) {                              \
      auto meta = out->meta();                                                \
      meta.strides = meta.calc_strides(out->dims());                          \
      out->set_meta(meta);                                                    \
    }                                                                         \
    if (out->IsSharedWith(x_)) {                                              \
      InplaceLogicalKernelStrideImpl<T,                                       \
                                     Context,                                 \
                                     funcs::Logical##name##Functor<T>>(       \
          dev_ctx, x_, y_, out);                                              \
    } else {                                                                  \
      LogicalKernelStrideImpl<T, Context, funcs::Logical##name##Functor<T>>(  \
          dev_ctx, x_, y_, out);                                              \
    }                                                                         \
  }
DEFINE_CUDA_BINARY_LOGICAL_STRIDE_OP(And)
DEFINE_CUDA_BINARY_LOGICAL_STRIDE_OP(Or)
DEFINE_CUDA_BINARY_LOGICAL_STRIDE_OP(Xor)
#undef DEFINE_CUDA_BINARY_LOGICAL_STRIDE_OP

template <typename T, typename Context>
void LogicalNotStrideKernel(const Context &dev_ctx,
                            const DenseTensor &x,
                            DenseTensor *out) {
  if (!FLAGS_use_stride_kernel) {
    PADDLE_THROW(common::errors::Fatal(
        "FLAGS_use_stride_kernel is closed. Strided kernel "
        "be called, something wrong has happened!"));
  }
  DenseTensor x_;
  if (!FLAGS_use_stride_compute_kernel) {
    if (!x.meta().is_contiguous()) {
      x_ = Tensor2Contiguous<Context>(dev_ctx, x);
    } else {
      x_ = x;
    }
  } else {
    x_ = x;
  }

  if (x_.meta().is_contiguous()) {
    auto meta = out->meta();
    meta.strides = meta.calc_strides(out->dims());
    out->set_meta(meta);
    phi::LogicalNotKernel<T, Context>(dev_ctx, x_, out);
    return;
  }

  if (!FLAGS_use_stride_compute_kernel) {
    PADDLE_THROW(
        common::errors::Fatal("FLAGS_use_stride_compute_kernel is closed. "
                              "Kernel using DenseTensorIterator "
                              "be called, something wrong has happened!"));
  }

  if (FLAGS_force_stride_compute_contig_out) {
    auto meta = out->meta();
    meta.strides = meta.calc_strides(out->dims());
    out->set_meta(meta);
  }
  if (!out->IsSharedWith(x_)) {
    LaunchLogicalNotStrideKernel<T, Context>(
        dev_ctx, x_, funcs::LogicalNotFunctor<T>(), out);
  } else {
    auto x_origin = x_;
    out->set_type(phi::DataType::BOOL);
    LaunchLogicalNotStrideKernel<T, Context>(
        dev_ctx, x_origin, funcs::LogicalNotFunctor<T>(), out);
  }
}

}  // namespace phi

#define REGISTER_LOGICAL_CUDA_STRIDE_KERNEL(logical_and, func_type) \
  PD_REGISTER_KERNEL(logical_and,                                   \
                     GPU,                                           \
                     STRIDED,                                       \
                     phi::Logical##func_type##StrideKernel,         \
                     float,                                         \
                     phi::float16,                                  \
                     phi::bfloat16,                                 \
                     double,                                        \
                     bool,                                          \
                     int64_t,                                       \
                     int,                                           \
                     int8_t,                                        \
                     phi::complex64,                                \
                     phi::complex128,                               \
                     int16_t) {                                     \
    kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);           \
  }
REGISTER_LOGICAL_CUDA_STRIDE_KERNEL(logical_and, And)
REGISTER_LOGICAL_CUDA_STRIDE_KERNEL(logical_or, Or)
REGISTER_LOGICAL_CUDA_STRIDE_KERNEL(logical_xor, Xor)
REGISTER_LOGICAL_CUDA_STRIDE_KERNEL(logical_not, Not)
#undef REGISTER_LOGICAL_CUDA_STRIDE_KERNEL
#endif
