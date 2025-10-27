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

#include "paddle/common/flags.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/contiguous_kernel.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"
#include "paddle/phi/kernels/elementwise_divide_kernel.h"
#include "paddle/phi/kernels/elementwise_multiply_kernel.h"
#include "paddle/phi/kernels/elementwise_subtract_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/dense_tensor_iterator.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/funcs/index_elementwise.cu.h"
#include "paddle/phi/kernels/impl/elementwise_kernel_impl.h"
#include "paddle/phi/kernels/scale_kernel.h"
#include "paddle/phi/kernels/stride/elementwise_stride_base.cu.h"

#if defined(__NVCC__) || defined(__HIPCC__) || defined(__xpu__)
#include "paddle/phi/kernels/funcs/dims_simplifier.h"

#endif

COMMON_DECLARE_bool(use_stride_kernel);
COMMON_DECLARE_bool(use_stride_compute_kernel);
COMMON_DECLARE_bool(force_stride_compute_contig_out);

namespace phi {
#define DEFINE_CUDA_BINARY_ELEMENTWISE_STRIDE_OP(name, functor_name)          \
  template <typename T, typename Context>                                     \
  void name##StrideKernel(const Context &dev_ctx,                             \
                          const DenseTensor &x,                               \
                          const DenseTensor &y,                               \
                          DenseTensor *out) {                                 \
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
      phi::name##Kernel<T, Context>(dev_ctx, x_, y_, out);                    \
      return;                                                                 \
    }                                                                         \
    if (!FLAGS_use_stride_compute_kernel) {                                   \
      PADDLE_THROW(                                                           \
          common::errors::Fatal("FLAGS_use_stride_compute_kernel is closed. " \
                                "Kernel using DenseTensorIterator "           \
                                "be called, something wrong has happened!")); \
    }                                                                         \
                                                                              \
    if (FLAGS_force_stride_compute_contig_out) {                              \
      auto meta = out->meta();                                                \
      meta.strides = meta.calc_strides(out->dims());                          \
      out->set_meta(meta);                                                    \
    }                                                                         \
    LaunchBinaryElementwiseStrideKernel<T, Context>(                          \
        dev_ctx, x_, y_, funcs::functor_name##Functor<T>(), -1, out);         \
  }

DEFINE_CUDA_BINARY_ELEMENTWISE_STRIDE_OP(Subtract, Subtract)
DEFINE_CUDA_BINARY_ELEMENTWISE_STRIDE_OP(Multiply, Multiply)
DEFINE_CUDA_BINARY_ELEMENTWISE_STRIDE_OP(Divide, Divide)
DEFINE_CUDA_BINARY_ELEMENTWISE_STRIDE_OP(CopySign, CopySign)
DEFINE_CUDA_BINARY_ELEMENTWISE_STRIDE_OP(Remainder, Remainder)
DEFINE_CUDA_BINARY_ELEMENTWISE_STRIDE_OP(Maximum, Maximum)
DEFINE_CUDA_BINARY_ELEMENTWISE_STRIDE_OP(Minimum, Minimum)
DEFINE_CUDA_BINARY_ELEMENTWISE_STRIDE_OP(FloorDivide, FloorDivide)
DEFINE_CUDA_BINARY_ELEMENTWISE_STRIDE_OP(Heaviside, ElementwiseHeaviside)
DEFINE_CUDA_BINARY_ELEMENTWISE_STRIDE_OP(FMax, FMax)
DEFINE_CUDA_BINARY_ELEMENTWISE_STRIDE_OP(FMin, FMin)
#undef DEFINE_CUDA_BINARY_ELEMENTWISE_STRIDE_OP

template <typename T, typename Context>
void AddStrideKernel(const Context &dev_ctx,
                     const DenseTensor &x,
                     const DenseTensor &y,
                     DenseTensor *out) {
  if (!FLAGS_use_stride_kernel) {
    PADDLE_THROW(common::errors::Fatal(
        "FLAGS_use_stride_kernel is closed. Strided kernel "
        "be called, something wrong has happened!"));
  }
  DenseTensor x_;
  DenseTensor y_;
  if (!FLAGS_use_stride_compute_kernel) {
    if (!x.meta().is_contiguous()) {
      x_ = Tensor2Contiguous<Context>(dev_ctx, x);
    } else {
      x_ = x;
    }
    if (!y.meta().is_contiguous()) {
      y_ = Tensor2Contiguous<Context>(dev_ctx, y);
    } else {
      y_ = y;
    }
  } else {
    x_ = x;
    y_ = y;
  }
  if (x_.meta().is_contiguous() && y_.meta().is_contiguous()) {
    auto meta = out->meta();
    meta.strides = meta.calc_strides(out->dims());
    out->set_meta(meta);
    phi::AddKernel<T, Context>(dev_ctx, x_, y_, out);
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

  if (x_.dtype() == phi::DataType::FLOAT32 &&
      y_.dtype() == phi::DataType::BFLOAT16) {
    LaunchBinaryElementwiseStrideKernel<T, Context>(
        dev_ctx,
        x_,
        y_,
        funcs::MultiPrecisionAddFunctor<T, phi::bfloat16>(),
        -1,
        out);
  } else if (x_.dtype() == phi::DataType::FLOAT32 &&
             y_.dtype() == phi::DataType::FLOAT16) {
    LaunchBinaryElementwiseStrideKernel<T, Context>(
        dev_ctx,
        x_,
        y_,
        funcs::MultiPrecisionAddFunctor<T, phi::float16>(),
        -1,
        out);
  } else {
    LaunchBinaryElementwiseStrideKernel<T, Context>(
        dev_ctx, x_, y_, funcs::AddFunctor<T>(), -1, out);
  }
}

template <typename DataT, typename ParamT>
struct ScaleFunctor {
  ParamT bias;
  ParamT scale;
  bool bias_after_scale;

  ScaleFunctor(ParamT scale_data, ParamT bias_data, bool is_bias_after_scale)
      : bias(bias_data),
        scale(scale_data),
        bias_after_scale(is_bias_after_scale) {}

  __device__ __forceinline__ DataT operator()(const DataT x) const {
    if (bias_after_scale) {
      return static_cast<DataT>(scale * static_cast<ParamT>(x) + bias);
    } else {
      return static_cast<DataT>(scale * (static_cast<ParamT>(x) + bias));
    }
  }
};

template <typename T, typename Context>
void ScaleStrideKernel(const Context &dev_ctx,
                       const DenseTensor &x,
                       const Scalar &scale,
                       const Scalar &bias,
                       bool bias_after_scale,
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
    phi::ScaleKernel<T, Context>(
        dev_ctx, x_, scale, bias, bias_after_scale, out);
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

  if (x.numel() <= 0 || (!x.IsInitialized())) {
    dev_ctx.template Alloc<T>(out);
    return;
  }

  using MT = typename phi::dtype::MPTypeTrait<T>::Type;
  LaunchUnaryElementwiseStrideKernel<T, Context>(
      dev_ctx,
      x_,
      ScaleFunctor<T, MT>(scale.to<MT>(), bias.to<MT>(), bias_after_scale),
      out);
}

template <typename T, typename Context>
void FullStrideKernel(const Context &dev_ctx,
                      const IntArray &shape,
                      const Scalar &val,
                      DataType dtype,
                      DenseTensor *out) {
  auto meta = out->meta();
  meta.strides = meta.calc_strides(out->dims());
  out->set_meta(meta);
  FullKernel<T, Context>(dev_ctx, shape, val, dtype, out);
}

template <typename T, typename Context>
void FullLikeStrideKernel(const Context &dev_ctx,
                          const DenseTensor &x,
                          const Scalar &val,
                          DataType dtype,
                          DenseTensor *out) {
  // Is this correct?
  // In fact, both ones_like and full_like can only generate contiguous tensors,
  // which differs from common sense, where both strides and shapes are
  // considered.
  auto meta = out->meta();
  meta.strides = meta.calc_strides(out->dims());
  out->set_meta(meta);
  FullLikeKernel<T, Context>(dev_ctx, x, val, dtype, out);
}

}  // namespace phi

using float16 = phi::float16;
using bfloat16 = phi::bfloat16;
using complex64 = phi::complex64;
using complex128 = phi::complex128;

PD_REGISTER_KERNEL(scale,
                   GPU,
                   STRIDED,
                   phi::ScaleStrideKernel,
                   bool,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16,
                   phi::float8_e4m3fn,
                   phi::float8_e5m2,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t,
                   phi::complex64,
                   phi::complex128) {}

PD_REGISTER_KERNEL(full,
                   GPU,
                   STRIDED,
                   phi::FullStrideKernel,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool,
                   phi::float8_e4m3fn,
                   phi::float8_e5m2,
                   phi::float16,
                   phi::bfloat16,
                   phi::complex64,
                   phi::complex128) {}

PD_REGISTER_KERNEL(full_like,
                   GPU,
                   STRIDED,
                   phi::FullLikeStrideKernel,
                   bool,
                   float,
                   double,
                   int,
                   int8_t,
                   int64_t,
                   int16_t,
                   uint8_t,
                   phi::float8_e4m3fn,
                   phi::float16,
                   phi::bfloat16,
                   phi::complex64,
                   phi::complex128) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
}

PD_REGISTER_KERNEL(add,
                   GPU,
                   STRIDED,
                   phi::AddStrideKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   bool,
                   uint8_t,
                   int8_t,
                   int64_t,
                   phi::float16,
                   phi::bfloat16,
                   complex64,
                   complex128) {}

PD_REGISTER_KERNEL(subtract,
                   GPU,
                   STRIDED,
                   phi::SubtractStrideKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t,
                   float16,
                   bfloat16,
                   complex64,
                   complex128) {}

PD_REGISTER_KERNEL(multiply,
                   GPU,
                   STRIDED,
                   phi::MultiplyStrideKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   bool,
                   float16,
                   complex64,
                   complex128,
                   bfloat16) {}

PD_REGISTER_KERNEL(divide,
                   GPU,
                   STRIDED,
                   phi::DivideStrideKernel,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool,
                   float16,
                   bfloat16,
                   complex64,
                   complex128) {}

PD_REGISTER_KERNEL(copysign,
                   GPU,
                   STRIDED,
                   phi::CopySignStrideKernel,
                   bool,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16) {}

PD_REGISTER_KERNEL(remainder,
                   GPU,
                   STRIDED,
                   phi::RemainderStrideKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::float16,
                   phi::complex64,
                   phi::complex128,
                   phi::bfloat16) {}

PD_REGISTER_KERNEL(maximum,
                   GPU,
                   STRIDED,
                   phi::MaximumStrideKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::float16,
                   phi::bfloat16) {}

PD_REGISTER_KERNEL(minimum,
                   GPU,
                   STRIDED,
                   phi::MinimumStrideKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::float16,
                   phi::bfloat16) {}

PD_REGISTER_KERNEL(floor_divide,
                   GPU,
                   STRIDED,
                   phi::FloorDivideStrideKernel,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16) {}

PD_REGISTER_KERNEL(heaviside,
                   GPU,
                   STRIDED,
                   phi::HeavisideStrideKernel,
                   float,
                   double,
                   int,
                   float16,
                   bfloat16,
                   int64_t) {}

PD_REGISTER_KERNEL(fmax,
                   GPU,
                   STRIDED,
                   phi::FMaxStrideKernel,
                   float,
                   double,
                   int,
                   float16,
                   bfloat16,
                   int64_t) {}

PD_REGISTER_KERNEL(fmin,
                   GPU,
                   STRIDED,
                   phi::FMinStrideKernel,
                   float,
                   double,
                   int,
                   float16,
                   bfloat16,
                   int64_t) {}

#endif
