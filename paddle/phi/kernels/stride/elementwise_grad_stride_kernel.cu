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
#include "paddle/phi/kernels/elementwise_add_grad_kernel.h"
#include "paddle/phi/kernels/elementwise_multiply_grad_kernel.h"
#include "paddle/phi/kernels/elementwise_multiply_kernel.h"
#include "paddle/phi/kernels/elementwise_subtract_grad_kernel.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/gpu/elementwise_grad.h"
#include "paddle/phi/kernels/scale_kernel.h"

#if defined(__NVCC__) || defined(__HIPCC__) || defined(__xpu__)
#include "paddle/phi/kernels/funcs/dims_simplifier.h"

#endif

COMMON_DECLARE_bool(use_stride_kernel);
COMMON_DECLARE_bool(use_stride_compute_kernel);

namespace phi {

template <typename Context>
phi::DenseTensor Tensor2Contiguous(const Context& dev_ctx,
                                   const phi::DenseTensor& tensor) {
  phi::DenseTensor dense_out;
  phi::MetaTensor meta_input(tensor);
  phi::MetaTensor meta_out(&dense_out);
  UnchangedInferMeta(meta_input, &meta_out);
  PD_VISIT_ALL_TYPES(tensor.dtype(), "Tensor2Contiguous", ([&] {
                       phi::ContiguousKernel<data_t, Context>(
                           dev_ctx, tensor, &dense_out);
                     }));
  return dense_out;
}

template <typename T, typename Context>
void AddGradStrideKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& y,
                         const DenseTensor& dout,
                         int axis,
                         DenseTensor* dx,
                         DenseTensor* dy) {
  if (!FLAGS_use_stride_kernel) {
    PADDLE_THROW(common::errors::Fatal(
        "FLAGS_use_stride_kernel is closed. Strided kernel "
        "be called, something wrong has happened!"));
  }

  DenseTensor x_;
  DenseTensor y_;
  DenseTensor dout_;

  // avoid inplace
  bool inplace_add = false;
  if (dx && dx->IsSharedBufferWith(dout)) inplace_add = true;

  if (FLAGS_use_stride_compute_kernel && !inplace_add) {
    auto meta = dout.meta();
    if (dx != nullptr && dy == nullptr && dx->dims() == dout.dims()) {
      dx->set_meta(meta);
      dx->ResetHolder(dout.Holder());
      dx->ShareInplaceVersionCounterWith(dout);
      return;
    }
    if (dy != nullptr && dx == nullptr && dy->dims() == dout.dims()) {
      dy->set_meta(meta);
      dy->ResetHolder(dout.Holder());
      dy->ShareInplaceVersionCounterWith(dout);
      return;
    }
  }

  if (x.initialized() && !x.meta().is_contiguous()) {
    x_ = Tensor2Contiguous<Context>(dev_ctx, x);
  } else {
    x_ = x;
  }
  if (y.initialized() && !y.meta().is_contiguous()) {
    y_ = Tensor2Contiguous<Context>(dev_ctx, y);
  } else {
    y_ = y;
  }
  if (dout.initialized() && !dout.meta().is_contiguous()) {
    dout_ = Tensor2Contiguous<Context>(dev_ctx, dout);
  } else {
    dout_ = dout;
  }

  if (dx) {
    auto dx_meta = dx->meta();
    dx_meta.strides = dx_meta.calc_strides(dx->dims());
    dx->set_meta(dx_meta);
  }

  if (dy) {
    auto dy_meta = dy->meta();
    dy_meta.strides = dy_meta.calc_strides(dy->dims());
    dy->set_meta(dy_meta);
  }
  phi::AddGradKernel<T>(dev_ctx, x_, y_, dout_, axis, dx, dy);
}

template <typename T, typename Context>
void SubtractGradStrideKernel(const Context& dev_ctx,
                              const DenseTensor& x,
                              const DenseTensor& y,
                              const DenseTensor& dout,
                              int axis,
                              DenseTensor* dx,
                              DenseTensor* dy) {
  if (!FLAGS_use_stride_kernel) {
    PADDLE_THROW(common::errors::Fatal(
        "FLAGS_use_stride_kernel is closed. Strided kernel "
        "be called, something wrong has happened!"));
  }

  DenseTensor x_;
  DenseTensor y_;
  DenseTensor dout_;

  if (FLAGS_use_stride_compute_kernel) {
    auto meta = dout.meta();
    if (dx != nullptr && dy != nullptr && dx->dims() == dout.dims() &&
        dy->dims() == dout.dims()) {
      dx->set_meta(meta);
      dx->ResetHolder(dout.Holder());
      dx->ShareInplaceVersionCounterWith(dout);
      phi::ScaleStrideKernel<T, Context>(dev_ctx, dout, -1, 0, false, dy);
      return;
    }
    if (dx != nullptr && dy == nullptr && dx->dims() == dout.dims()) {
      dx->set_meta(meta);
      dx->ResetHolder(dout.Holder());
      dx->ShareInplaceVersionCounterWith(dout);
      return;
    }
    if (dy != nullptr && dx == nullptr && dy->dims() == dout.dims()) {
      phi::ScaleStrideKernel<T, Context>(dev_ctx, dout, -1, 0, false, dy);
      return;
    }
  }

  if (x.initialized() && !x.meta().is_contiguous()) {
    x_ = Tensor2Contiguous<Context>(dev_ctx, x);
  } else {
    x_ = x;
  }
  if (y.initialized() && !y.meta().is_contiguous()) {
    y_ = Tensor2Contiguous<Context>(dev_ctx, y);
  } else {
    y_ = y;
  }
  if (dout.initialized() && !dout.meta().is_contiguous()) {
    dout_ = Tensor2Contiguous<Context>(dev_ctx, dout);
  } else {
    dout_ = dout;
  }

  if (dx) {
    auto dx_meta = dx->meta();
    dx_meta.strides = dx_meta.calc_strides(dx->dims());
    dx->set_meta(dx_meta);
  }

  if (dy) {
    auto dy_meta = dy->meta();
    dy_meta.strides = dy_meta.calc_strides(dy->dims());
    dy->set_meta(dy_meta);
  }
  phi::SubtractGradKernel<T>(dev_ctx, x_, y_, dout_, axis, dx, dy);
}

template <typename T, typename Context>
void MultiplyGradStrideKernel(const Context& dev_ctx,
                              const DenseTensor& x,
                              const DenseTensor& y,
                              const DenseTensor& dout,
                              int axis,
                              DenseTensor* dx,
                              DenseTensor* dy) {
  if (!FLAGS_use_stride_kernel) {
    PADDLE_THROW(common::errors::Fatal(
        "FLAGS_use_stride_kernel is closed. Strided kernel "
        "be called, something wrong has happened!"));
  }

  DenseTensor x_;
  DenseTensor y_;
  DenseTensor dout_;

  bool invalid_stride = false;
  if (IsComplexType(x.dtype())) {
    invalid_stride = true;
  }
  if (IsComplexType(y.dtype())) {
    invalid_stride = true;
  }

  if (FLAGS_use_stride_compute_kernel && dout.initialized() &&
      dout.numel() != 0 && !invalid_stride) {
    auto broadcast_dim = dout.dims();
    if (x.initialized() && y.initialized() && dx != nullptr && dy != nullptr &&
        broadcast_dim == dx->dims() && broadcast_dim == dy->dims()) {
      phi::MultiplyStrideKernel<T, Context>(dev_ctx, dout, y, dx);
      phi::MultiplyStrideKernel<T, Context>(dev_ctx, dout, x, dy);
      return;
    }

    if (y.initialized() && dx != nullptr && dy == nullptr &&
        broadcast_dim == dx->dims()) {
      phi::MultiplyStrideKernel<T, Context>(dev_ctx, dout, y, dx);
      return;
    }

    if (x.initialized() && dy != nullptr && dx == nullptr &&
        broadcast_dim == dy->dims()) {
      phi::MultiplyStrideKernel<T, Context>(dev_ctx, dout, x, dy);
      return;
    }
  }

  if (x.initialized() && !x.meta().is_contiguous()) {
    x_ = Tensor2Contiguous<Context>(dev_ctx, x);
  } else {
    x_ = x;
  }

  if (y.initialized() && !y.meta().is_contiguous()) {
    y_ = Tensor2Contiguous<Context>(dev_ctx, y);
  } else {
    y_ = y;
  }

  if (dout.initialized() && !dout.meta().is_contiguous()) {
    dout_ = Tensor2Contiguous<Context>(dev_ctx, dout);
  } else {
    dout_ = dout;
  }

  if (dx) {
    auto dx_meta = dx->meta();
    dx_meta.strides = dx_meta.calc_strides(dx->dims());
    dx->set_meta(dx_meta);
  }

  if (dy) {
    auto dy_meta = dy->meta();
    dy_meta.strides = dy_meta.calc_strides(dy->dims());
    dy->set_meta(dy_meta);
  }
  phi::MultiplyGradKernel<T>(dev_ctx, x_, y_, dout_, axis, dx, dy);
}

}  // namespace phi

using float16 = phi::float16;
using bfloat16 = phi::bfloat16;
using complex64 = ::phi::complex64;
using complex128 = ::phi::complex128;

PD_REGISTER_KERNEL(add_grad,
                   GPU,
                   STRIDED,
                   phi::AddGradStrideKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::float16,
                   phi::bfloat16,
                   phi::complex64,
                   phi::complex128) {}

PD_REGISTER_KERNEL(subtract_grad,
                   GPU,
                   STRIDED,
                   phi::SubtractGradStrideKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::float16,
                   phi::bfloat16,
                   phi::complex64,
                   phi::complex128) {}

PD_REGISTER_KERNEL(multiply_grad,
                   GPU,
                   STRIDED,
                   phi::MultiplyGradStrideKernel,
                   float,
                   phi::float16,
                   double,
                   int,
                   int64_t,
                   bool,
                   phi::bfloat16,
                   phi::complex64,
                   phi::complex128) {}

#endif
