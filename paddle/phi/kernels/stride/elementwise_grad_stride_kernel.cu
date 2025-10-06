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
  if (FLAGS_use_stride_compute_kernel && !dx->IsSharedBufferWith(dout)) {
    printf("enter add grad stride\n");
    auto meta = dout.meta();

    printf("axis:%d\n", axis);

    printf("out grad shape\n");
    for (int i = 0; i < dout.dims().size(); i++) {
      printf("%d ", dout.dims()[i]);
    }
    printf("\n");

    printf("out grad stride\n");
    for (int i = 0; i < dout.strides().size(); i++) {
      printf("%d ", dout.strides()[i]);
    }
    printf("\n");

    printf("out grad offset:%d\n", dout.offset());

    printf("x shape\n");
    for (int i = 0; i < x.dims().size(); i++) {
      printf("%d ", x.dims()[i]);
    }
    printf("\n");

    printf("x stride\n");
    for (int i = 0; i < x.strides().size(); i++) {
      printf("%d ", x.strides()[i]);
    }
    printf("\n");

    printf("y shape\n");
    for (int i = 0; i < y.dims().size(); i++) {
      printf("%d ", y.dims()[i]);
    }
    printf("\n");

    printf("y stride\n");
    for (int i = 0; i < y.strides().size(); i++) {
      printf("%d ", y.strides()[i]);
    }
    printf("\n");

    printf("x_grad shape\n");
    for (int i = 0; i < dx->dims().size(); i++) {
      printf("%d ", dx->dims()[i]);
    }
    printf("\n");

    printf("x_grad stride\n");
    for (int i = 0; i < dx->strides().size(); i++) {
      printf("%d ", dx->strides()[i]);
    }
    printf("\n");

    printf("y_grad shape\n");
    for (int i = 0; i < dy->dims().size(); i++) {
      printf("%d ", dy->dims()[i]);
    }
    printf("\n");

    printf("y_grad stride\n");
    for (int i = 0; i < dy->strides().size(); i++) {
      printf("%d ", dy->strides()[i]);
    }
    printf("\n");

    if (dx->dtype() != dout.dtype()) {
      printf("auto promote dx\n");
    }

    if (dy->dtype() != dout.dtype()) {
      printf("auto promote dy\n");
    }

    if (dx->IsSharedBufferWith(dout)) {
      printf("dx inplace out\n");
    }

    if (dy->IsSharedBufferWith(dout)) {
      printf("dy inplace out\n");
    }
    // auto x_meta = dx->meta();
    // x_meta.strides = meta.calc_strides(dx->dims());
    // dx->set_meta(x_meta);

    // auto y_meta = dy->meta();
    // y_meta.strides = meta.calc_strides(dy->dims());
    // dx->set_meta(y_meta);

    if (dx != nullptr && dy != nullptr && dx->dims() == dout.dims() &&
        dy->dims() == dout.dims()) {
      printf("branch 1\n");
      dx->set_meta(meta);
      dx->ResetHolder(dout.Holder());
      dx->ShareInplaceVersionCounterWith(dout);
      dy->set_meta(meta);
      dy->ResetHolder(dout.Holder());
      dy->ShareInplaceVersionCounterWith(dout);
      printf("after x_grad shape\n");
      for (int i = 0; i < dx->dims().size(); i++) {
        printf("%d ", dx->dims()[i]);
      }
      printf("\n");

      printf("after x_grad stride\n");
      for (int i = 0; i < dx->strides().size(); i++) {
        printf("%d ", dx->strides()[i]);
      }
      printf("\n");

      printf("after y_grad shape\n");
      for (int i = 0; i < dy->dims().size(); i++) {
        printf("%d ", dy->dims()[i]);
      }
      printf("\n");

      printf("after y_grad stride\n");
      for (int i = 0; i < dy->strides().size(); i++) {
        printf("%d ", dy->strides()[i]);
      }
      printf("\n");
      return;
    }
    if (dx != nullptr && dy == nullptr && dx->dims() == dout.dims()) {
      printf("branch 2\n");
      dx->set_meta(meta);
      dx->ResetHolder(dout.Holder());
      dx->ShareInplaceVersionCounterWith(dout);
      return;
    }
    if (dy != nullptr && dx == nullptr && dy->dims() == dout.dims()) {
      printf("branch 3\n");
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
    printf("enter sub grad stride\n");
    auto meta = dout.meta();
    if (dx != nullptr && dy != nullptr && dx->dims() == dout.dims()) {
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
    if (dy != nullptr && dx == nullptr) {
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

  if (FLAGS_use_stride_compute_kernel && dout.initialized() &&
      dout.numel() != 0) {
    printf("enter mul grad\n");
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
