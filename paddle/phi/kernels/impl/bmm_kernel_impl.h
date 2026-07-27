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

#pragma once

#include "paddle/phi/kernels/bmm_kernel.h"

#include "paddle/phi/kernels/contiguous_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"

namespace phi {

template <typename T, typename Context>
void BmmKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const DenseTensor& y,
               DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  if (x.numel() == 0 || y.numel() == 0) {
    return;
  }

  auto blas = funcs::GetBlas<Context, T>(dev_ctx);

  auto mat_dim_a = funcs::CreateMatrixDescriptor(x.dims(), 0, false);
  auto mat_dim_b = funcs::CreateMatrixDescriptor(y.dims(), 0, false);

  blas.MatMul(x, mat_dim_a, y, mat_dim_b, T(1), out, T(0));
}

template <typename T, typename Context>
void BmmOutDtypeKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const DenseTensor& y,
                       DataType out_dtype,
                       DenseTensor* out) {
  PADDLE_ENFORCE_EQ(
      out_dtype,
      DataType::FLOAT32,
      common::errors::InvalidArgument(
          "The out_dtype of paddle.bmm currently only supports float32."));
  PADDLE_ENFORCE_EQ(
      x.dtype(),
      DataType::BFLOAT16,
      common::errors::InvalidArgument(
          "The out_dtype of paddle.bmm currently only supports bfloat16 "
          "Input(X)."));
  PADDLE_ENFORCE_EQ(
      y.dtype(),
      DataType::BFLOAT16,
      common::errors::InvalidArgument(
          "The out_dtype of paddle.bmm currently only supports bfloat16 "
          "Input(Y)."));

  const auto x_dims = x.dims();
  const auto y_dims = y.dims();
  PADDLE_ENFORCE_EQ(
      x_dims.size(),
      3,
      common::errors::InvalidArgument(
          "The out_dtype of paddle.bmm currently only supports 3-D Input(X)."));
  PADDLE_ENFORCE_EQ(
      y_dims.size(),
      3,
      common::errors::InvalidArgument(
          "The out_dtype of paddle.bmm currently only supports 3-D Input(Y)."));
  PADDLE_ENFORCE_EQ(
      x_dims[0],
      y_dims[0],
      common::errors::InvalidArgument(
          "Input(X) and Input(Y) must have the same batch size, but received "
          "%d and %d.",
          x_dims[0],
          y_dims[0]));
  PADDLE_ENFORCE_EQ(
      x_dims[2],
      y_dims[1],
      common::errors::InvalidArgument(
          "Input(X)'s width must equal Input(Y)'s height, but received %d "
          "and %d.",
          x_dims[2],
          y_dims[1]));

#if defined(PADDLE_WITH_CUDA) && !defined(PADDLE_WITH_HIP)
  if (x.numel() == 0 || y.numel() == 0) {
    Full<float, Context>(dev_ctx, out->dims(), 0, out);
    return;
  }

  DenseTensor x_contiguous;
  DenseTensor y_contiguous;
  const DenseTensor* x_ptr = &x;
  const DenseTensor* y_ptr = &y;
  if (!x.meta().is_contiguous()) {
    ContiguousKernel<T, Context>(dev_ctx, x, &x_contiguous);
    x_ptr = &x_contiguous;
  }
  if (!y.meta().is_contiguous()) {
    ContiguousKernel<T, Context>(dev_ctx, y, &y_contiguous);
    y_ptr = &y_contiguous;
  }

  const int64_t batch_count = x_dims[0];
  const int64_t m = x_dims[1];
  const int64_t k = x_dims[2];
  const int64_t n = y_dims[2];
  dev_ctx.template Alloc<float>(out);
  funcs::Blas<Context> blas(dev_ctx);
  blas.BatchedGEMM(CblasNoTrans,
                   CblasNoTrans,
                   m,
                   n,
                   k,
                   1.0f,
                   x_ptr->data<phi::bfloat16>(),
                   y_ptr->data<phi::bfloat16>(),
                   0.0f,
                   out->data<float>(),
                   batch_count,
                   m * k,
                   k * n);
#else
  PADDLE_THROW(common::errors::Unimplemented(
      "The out_dtype of paddle.bmm currently only supports CUDA bfloat16 "
      "inputs."));
#endif
}

}  // namespace phi
