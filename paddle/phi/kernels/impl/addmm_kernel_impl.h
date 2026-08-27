/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/phi/kernels/addmm_kernel.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/contiguous_kernel.h"
#include "paddle/phi/kernels/expand_kernel.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/for_range.h"

namespace phi {

template <typename T>
struct AddmmScaleFunctor {
  AddmmScaleFunctor(double scale, T* output) : scale_(scale), output_(output) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    output_[idx] = scale_ == 0.0f ? static_cast<T>(0)
                                  : output_[idx] * static_cast<T>(scale_);
  }

 private:
  double scale_;
  T* output_;
};

template <typename T, typename Context>
void AddmmKernel(const Context& dev_ctx,
                 const DenseTensor& input,
                 const DenseTensor& x,
                 const DenseTensor& y,
                 double beta,
                 double alpha,
                 DenseTensor* out) {
  auto input_dims = input.dims();
  auto x_dims = x.dims();
  auto y_dims = y.dims();

  DenseTensor input_2d(input);
  if (input.dims().size() == 1) {
    input_dims = {1, input.dims()[0]};
    input_2d.Resize(input_dims);
  }

  // broadcast mode check
  if (x_dims[0] != input_dims[0]) {
    PADDLE_ENFORCE_EQ(input_dims[0],
                      1,
                      errors::InvalidArgument(
                          "When x_dims[0] is not equal with input_dims[0], "
                          "input_dims[0] must be 1 but got %s",
                          input_dims[0]));
    PADDLE_ENFORCE_EQ(y_dims[1] == input_dims[1] || input_dims[1] == 1,
                      true,
                      errors::InvalidArgument(
                          "The input tensor shape mismatch, input shape=[%s], "
                          "x shape=[%s], y shape=[%s]",
                          input_dims,
                          x_dims,
                          y_dims));
  }
  // broadcast mode check
  if (y_dims[1] != input_dims[1]) {
    PADDLE_ENFORCE_EQ(input_dims[1],
                      1,
                      errors::InvalidArgument(
                          "When y_dims[1] is not equal with input_dims[0], "
                          "input_dims[0] must be 1 but got %s",
                          input_dims[1]));
    PADDLE_ENFORCE_EQ(x_dims[0] == input_dims[0] || input_dims[0] == 1,
                      true,
                      errors::InvalidArgument(
                          "The input tensor shape mismatch, input shape=[%s], "
                          "x shape=[%s], y shape=[%s]",
                          input_dims,
                          x_dims,
                          y_dims));
  }
  // broadcast mode check
  PADDLE_ENFORCE_EQ(
      x_dims[1],
      y_dims[0],
      errors::InvalidArgument(
          "The input tensor X's width must be equal with matrix Y' height. "
          "But received X's shape = [%s], Y's shape = [%s].",
          x_dims[1],
          y_dims[0]));

  dev_ctx.template Alloc<T>(out);
  if (out->numel() == 0) return;

  auto blas = funcs::GetBlas<Context, T>(dev_ctx);

  VLOG(3) << "broadcast input to [" << x_dims[0] << "," << y_dims[1] << "]";
  ExpandKernel<T, Context>(
      dev_ctx, input_2d, IntArray({x_dims[0], y_dims[1]}), out);

  // When K == 0 (x or y empty), the x@y term contributes nothing, so
  // out = beta * input. Align with PyTorch: when beta == 0 the input term is
  // dropped entirely and out is zeroed, otherwise nan/inf in input would
  // propagate through nan * 0 = nan.
  if (x.numel() == 0 || y.numel() == 0) {
    funcs::ForRange<Context> for_range(dev_ctx, out->numel());
    AddmmScaleFunctor<T> functor(beta, out->data<T>());
    for_range(functor);
    return;
  }

  using MPType = typename MPTypeTrait<T>::Type;
  if constexpr (std::is_same_v<MPType, float>) {
    float t_alpha = static_cast<float>(alpha);
    float t_beta = static_cast<float>(beta);
    blas.GEMM(CblasNoTrans,
              CblasNoTrans,
              x_dims[0],
              y_dims[1],
              x_dims[1],
              t_alpha,
              x.data<T>(),
              y.data<T>(),
              t_beta,
              out->data<T>());
  } else {
    T t_alpha = static_cast<T>(alpha);
    T t_beta = static_cast<T>(beta);
    blas.GEMM(false,
              false,
              x_dims[0],
              y_dims[1],
              x_dims[1],
              t_alpha,
              x.data<T>(),
              x_dims[1],
              y.data<T>(),
              y_dims[1],
              t_beta,
              out->data<T>(),
              y_dims[1]);
  }
}

template <typename T, typename Context>
void AddmmOutDtypeKernel(const Context& dev_ctx,
                         const DenseTensor& input,
                         const DenseTensor& x,
                         const DenseTensor& y,
                         DataType out_dtype,
                         double beta,
                         double alpha,
                         DenseTensor* out) {
#if defined(PADDLE_WITH_CUDA) && !defined(PADDLE_WITH_HIP)
  PADDLE_ENFORCE_EQ(
      out_dtype,
      DataType::FLOAT32,
      errors::InvalidArgument(
          "The out_dtype of paddle.addmm currently only supports float32."));
  PADDLE_ENFORCE_EQ(
      x.dtype() == DataType::FLOAT16 || x.dtype() == DataType::BFLOAT16,
      true,
      errors::InvalidArgument(
          "The out_dtype of paddle.addmm currently only supports float16 or "
          "bfloat16 Input(X)."));
  PADDLE_ENFORCE_EQ(
      x.dtype(),
      y.dtype(),
      errors::InvalidArgument(
          "Input(X) and Input(Y) must have the same dtype when out_dtype is "
          "specified for paddle.addmm."));
  PADDLE_ENFORCE_EQ(
      input.dtype() == x.dtype() || input.dtype() == DataType::FLOAT32,
      true,
      errors::InvalidArgument(
          "Input(input) must have the same dtype as Input(X) or float32 when "
          "out_dtype is specified for paddle.addmm."));

  const auto input_dims = input.dims();
  const auto x_dims = x.dims();
  const auto y_dims = y.dims();
  PADDLE_ENFORCE_EQ(
      input_dims.size() == 1 || input_dims.size() == 2,
      true,
      errors::InvalidArgument(
          "The dimension of input must be 1 or 2 when out_dtype is specified "
          "for paddle.addmm."));
  PADDLE_ENFORCE_EQ(
      x_dims.size(),
      2,
      errors::InvalidArgument(
          "The dimension of x must be 2 when out_dtype is specified for "
          "paddle.addmm."));
  PADDLE_ENFORCE_EQ(
      y_dims.size(),
      2,
      errors::InvalidArgument(
          "The dimension of y must be 2 when out_dtype is specified for "
          "paddle.addmm."));
  PADDLE_ENFORCE_EQ(
      x_dims[1],
      y_dims[0],
      errors::InvalidArgument(
          "Input(X)'s width must equal Input(Y)'s height when out_dtype is "
          "specified for paddle.addmm."));

  DenseTensor input_2d(input);
  if (input_dims.size() == 1) {
    input_2d.Resize({1, input_dims[0]});
  }

  DenseTensor input_float;
  const DenseTensor* input_ptr = &input_2d;
  if (input.dtype() != DataType::FLOAT32) {
    input_float = Cast<T, Context>(dev_ctx, input_2d, DataType::FLOAT32);
    input_ptr = &input_float;
  }

  ExpandKernel<float>(
      dev_ctx, *input_ptr, IntArray({x_dims[0], y_dims[1]}), out);
  if (out->numel() == 0) {
    return;
  }
  if (x.numel() == 0 || y.numel() == 0) {
    funcs::ForRange<Context> for_range(dev_ctx, out->numel());
    AddmmScaleFunctor<float> functor(beta, out->data<float>());
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
  blas.GEMM(CblasNoTrans,
            CblasNoTrans,
            x_dims[0],
            y_dims[1],
            x_dims[1],
            static_cast<float>(alpha),
            x_ptr->data<T>(),
            y_ptr->data<T>(),
            static_cast<float>(beta),
            out->data<float>());
#else
  PADDLE_THROW(errors::Unimplemented(
      "The out_dtype of paddle.addmm currently only supports CUDA float16 or "
      "bfloat16 inputs."));
#endif
}

}  // namespace phi
