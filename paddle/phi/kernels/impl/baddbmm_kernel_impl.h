/* Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <type_traits>

#include "glog/logging.h"

#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/kernels/baddbmm_kernel.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/contiguous_kernel.h"
#include "paddle/phi/kernels/expand_kernel.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/for_range.h"

namespace phi {

template <typename T>
struct BaddbmmScaleFunctor {
  BaddbmmScaleFunctor(double scale, T* output)
      : scale_(scale), output_(output) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    output_[idx] = scale_ == 0.0f ? static_cast<T>(0)
                                  : output_[idx] * static_cast<T>(scale_);
  }

 private:
  double scale_;
  T* output_;
};

template <typename T, typename Context>
void BaddbmmKernel(const Context& dev_ctx,
                   const DenseTensor& input,
                   const DenseTensor& x,
                   const DenseTensor& y,
                   double beta,
                   double alpha,
                   DenseTensor* out) {
  auto input_dims = input.dims();
  auto x_dims = x.dims();
  auto y_dims = y.dims();

  DenseTensor input_3d(input);
  if (input_dims.size() < 3) {
    std::vector<int64_t> padded_dims(3 - input_dims.size(), 1);
    const auto input_dims_vec = common::vectorize(input_dims);
    padded_dims.insert(
        padded_dims.end(), input_dims_vec.begin(), input_dims_vec.end());
    input_dims = common::make_ddim(padded_dims);
    input_3d.Resize(input_dims);
  }

  // broadcast mode check
  if (x_dims[0] != input_dims[0]) {
    PADDLE_ENFORCE_EQ(input_dims[0],
                      1,
                      errors::InvalidArgument(
                          "When x_dims[0] is not equal with input_dims[0], "
                          "input_dims[0] must be 1 but got %s",
                          input_dims[0]));
    PADDLE_ENFORCE_EQ(
        (x_dims[1] == input_dims[1] || input_dims[1] == 1) &&
            (y_dims[2] == input_dims[2] || input_dims[2] == 1),
        true,
        errors::InvalidArgument(
            "When x_dims[0] is not equal with input_dims[0], "
            "x_dims[1] and y_dims[2] must be equal with input_dims[1] and "
            "input_dims[2] respectively, or input_dims[1] and input_dims[2] "
            "must be 1. But got x_dims[1] = %s, input_dims[1] = %s, y_dims[2] "
            "= %s, input_dims[2] = %s",
            x_dims[1],
            input_dims[1],
            y_dims[2],
            input_dims[2]));
  }

  if (x_dims[1] != input_dims[1]) {
    PADDLE_ENFORCE_EQ(input_dims[1],
                      1,
                      errors::InvalidArgument(
                          "When x_dims[1] is not equal with input_dims[1], "
                          "input_dims[1] must be 1 but got %s",
                          input_dims[1]));
    PADDLE_ENFORCE_EQ(
        (x_dims[0] == input_dims[0] || input_dims[0] == 1) &&
            (y_dims[2] == input_dims[2] || input_dims[2] == 1),
        true,
        errors::InvalidArgument(
            "When x_dims[1] is not equal with input_dims[1], "
            "x_dims[0] and y_dims[2] must be equal with input_dims[0] and "
            "input_dims[2] respectively, or input_dims[0] and input_dims[2] "
            "must be 1. But got x_dims[0] = %s, input_dims[0] = %s, y_dims[2] "
            "= %s, input_dims[2] = %s",
            x_dims[0],
            input_dims[0],
            y_dims[2],
            input_dims[2]));
  }

  if (y_dims[2] != input_dims[2]) {
    PADDLE_ENFORCE_EQ(input_dims[2],
                      1,
                      errors::InvalidArgument(
                          "When y_dims[2] is not equal with input_dims[2], "
                          "input_dims[2] must be 1 but got %s",
                          input_dims[2]));
    PADDLE_ENFORCE_EQ(
        (x_dims[0] == input_dims[0] || input_dims[0] == 1) &&
            (x_dims[1] == input_dims[1] || input_dims[1] == 1),
        true,
        errors::InvalidArgument(
            "When y_dims[2] is not equal with input_dims[2], "
            "x_dims[0] and x_dims[1] must be equal with input_dims[0] and "
            "input_dims[1] respectively, or input_dims[0] and input_dims[1] "
            "must be 1. But got x_dims[0] = %s, input_dims[0] = %s, x_dims[1] "
            "= %s, input_dims[1] = %s",
            x_dims[0],
            input_dims[0],
            x_dims[1],
            input_dims[1]));
  }
  PADDLE_ENFORCE_EQ(
      x_dims[2],
      y_dims[1],
      errors::InvalidArgument(
          "The input tensor X's width must be equal with matrix Y' height. "
          "But received X's shape = [%s], Y's shape = [%s].",
          x_dims[2],
          y_dims[1]));

  dev_ctx.template Alloc<T>(out);
  // Degenerate shape (B/M/N == 0): out is empty, nothing to compute. Must
  // return before the bcast_dims integer division below, which would divide by
  // zero (SIGFPE) when a broadcast input dim is 0.
  if (out->numel() == 0) {
    return;
  }
  auto blas = funcs::GetBlas<Context, T>(dev_ctx);

  VLOG(3) << "broadcast input to [" << x_dims[0] << "," << x_dims[1] << ","
          << y_dims[2] << "]";
  ExpandKernel<T, Context>(
      dev_ctx, input_3d, IntArray({x_dims[0], x_dims[1], y_dims[2]}), out);

  // When K == 0 (x or y empty), the x@y term contributes nothing, so
  // out = beta * input. Feeding K == 0 into (Batched)GEMM otherwise triggers a
  // cuBLAS INVALID_VALUE error. Align with PyTorch: beta == 0 zeros the result
  // so that nan/inf in input does not propagate (nan * 0 = nan otherwise).
  if (x.numel() == 0 || y.numel() == 0) {
    funcs::ForRange<Context> for_range(dev_ctx, out->numel());
    BaddbmmScaleFunctor<T> functor(beta, out->data<T>());
    for_range(functor);
    return;
  }

  using MPType = typename MPTypeTrait<T>::Type;

  // special case for MPType
  if constexpr (std::is_same_v<MPType, float>) {
    VLOG(4) << "Function: baddbmm, Type of T: " << typeid(T).name();
    VLOG(4) << "Function: baddbmm, Type of MPType: " << typeid(MPType).name();
    float t_alpha = static_cast<float>(alpha);
    float t_beta = static_cast<float>(beta);
    if (x_dims[0] == 1) {
      blas.GEMM(CblasNoTrans,
                CblasNoTrans,
                x_dims[1],
                y_dims[2],
                x_dims[2],
                t_alpha,
                x.data<T>(),
                y.data<T>(),
                t_beta,
                out->data<T>());
    } else {
      blas.BatchedGEMM(CblasNoTrans,
                       CblasNoTrans,
                       x_dims[1],
                       y_dims[2],
                       x_dims[2],
                       t_alpha,
                       x.data<T>(),
                       y.data<T>(),
                       t_beta,
                       out->data<T>(),
                       x_dims[0],
                       x_dims[1] * x_dims[2],
                       x_dims[2] * y_dims[2]);
    }
  } else {
    T t_alpha = static_cast<T>(alpha);
    T t_beta = static_cast<T>(beta);
    if (x_dims[0] == 1) {
      blas.GEMM(CblasNoTrans,
                CblasNoTrans,
                x_dims[1],
                y_dims[2],
                x_dims[2],
                t_alpha,
                x.data<T>(),
                y.data<T>(),
                t_beta,
                out->data<T>());
    } else {
      blas.BatchedGEMM(CblasNoTrans,
                       CblasNoTrans,
                       x_dims[1],
                       y_dims[2],
                       x_dims[2],
                       t_alpha,
                       x.data<T>(),
                       y.data<T>(),
                       t_beta,
                       out->data<T>(),
                       x_dims[0],
                       x_dims[1] * x_dims[2],
                       x_dims[2] * y_dims[2]);
      // x_dims[2] == y_dims[1]
    }
  }
}

template <typename T, typename Context>
void BaddbmmOutDtypeKernel(const Context& dev_ctx,
                           const DenseTensor& input,
                           const DenseTensor& x,
                           const DenseTensor& y,
                           DataType out_dtype,
                           double beta,
                           double alpha,
                           DenseTensor* out) {
#if defined(PADDLE_WITH_CUDA) && !defined(PADDLE_WITH_HIP)
  PADDLE_ENFORCE_EQ(
      x.dtype(),
      y.dtype(),
      errors::InvalidArgument(
          "Input(X) and Input(Y) must have the same dtype when out_dtype is "
          "specified for paddle.baddbmm."));
  PADDLE_ENFORCE_EQ(
      out_dtype,
      DataType::FLOAT32,
      errors::InvalidArgument(
          "The mixed out_dtype path of paddle.baddbmm only supports "
          "float32."));
  PADDLE_ENFORCE_EQ(
      x.dtype() == DataType::FLOAT16 || x.dtype() == DataType::BFLOAT16,
      true,
      errors::InvalidArgument(
          "The mixed out_dtype path of paddle.baddbmm only supports float16 "
          "or bfloat16 Input(X)."));
  PADDLE_ENFORCE_EQ(
      input.dtype() == x.dtype() || input.dtype() == out_dtype,
      true,
      errors::InvalidArgument(
          "Input(input)'s dtype must match Input(X)'s dtype or out_dtype when "
          "out_dtype is specified for paddle.baddbmm."));

  auto input_dims = input.dims();
  const auto x_dims = x.dims();
  const auto y_dims = y.dims();
  PADDLE_ENFORCE_LE(
      input_dims.size(),
      3,
      errors::InvalidArgument(
          "The dimension of input must not exceed 3 when out_dtype is "
          "specified for paddle.baddbmm."));
  PADDLE_ENFORCE_EQ(
      x_dims.size(),
      3,
      errors::InvalidArgument(
          "The dimension of x must be 3 when out_dtype is specified for "
          "paddle.baddbmm."));
  PADDLE_ENFORCE_EQ(
      y_dims.size(),
      3,
      errors::InvalidArgument(
          "The dimension of y must be 3 when out_dtype is specified for "
          "paddle.baddbmm."));
  PADDLE_ENFORCE_EQ(
      x_dims[0],
      y_dims[0],
      errors::InvalidArgument(
          "Input(X) and Input(Y) must have the same batch size when out_dtype "
          "is specified for paddle.baddbmm."));
  PADDLE_ENFORCE_EQ(
      x_dims[2],
      y_dims[1],
      errors::InvalidArgument(
          "Input(X)'s width must equal Input(Y)'s height when out_dtype is "
          "specified for paddle.baddbmm."));
  DenseTensor input_3d(input);
  if (input_dims.size() < 3) {
    std::vector<int64_t> padded_dims(3 - input_dims.size(), 1);
    const auto input_dims_vec = common::vectorize(input_dims);
    padded_dims.insert(
        padded_dims.end(), input_dims_vec.begin(), input_dims_vec.end());
    input_dims = common::make_ddim(padded_dims);
    input_3d.Resize(input_dims);
  }

  DenseTensor input_float;
  const DenseTensor* input_ptr = &input_3d;
  if (input.dtype() != DataType::FLOAT32) {
    input_float = Cast<T, Context>(dev_ctx, input_3d, DataType::FLOAT32);
    input_ptr = &input_float;
  }

  ExpandKernel<float>(
      dev_ctx, *input_ptr, IntArray({x_dims[0], x_dims[1], y_dims[2]}), out);
  if (out->numel() == 0) {
    return;
  }
  if (x.numel() == 0 || y.numel() == 0) {
    funcs::ForRange<Context> for_range(dev_ctx, out->numel());
    BaddbmmScaleFunctor<float> functor(beta, out->data<float>());
    for_range(functor);
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

  funcs::Blas<Context> blas(dev_ctx);
  if (x_dims[0] == 1) {
    blas.GEMM(CblasNoTrans,
              CblasNoTrans,
              x_dims[1],
              y_dims[2],
              x_dims[2],
              static_cast<float>(alpha),
              x_ptr->data<T>(),
              y_ptr->data<T>(),
              static_cast<float>(beta),
              out->data<float>());
  } else {
    blas.BatchedGEMM(CblasNoTrans,
                     CblasNoTrans,
                     x_dims[1],
                     y_dims[2],
                     x_dims[2],
                     static_cast<float>(alpha),
                     x_ptr->data<T>(),
                     y_ptr->data<T>(),
                     static_cast<float>(beta),
                     out->data<float>(),
                     x_dims[0],
                     x_dims[1] * x_dims[2],
                     x_dims[2] * y_dims[2]);
  }
#else
  PADDLE_THROW(errors::Unimplemented(
      "The out_dtype of paddle.baddbmm currently only supports CUDA."));
#endif
}

}  // namespace phi
