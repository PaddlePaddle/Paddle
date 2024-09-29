// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>

#include "paddle/phi/core/enforce.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename Context, typename T>
struct LRNFunctor {
  void operator()(const Context& ctx,
                  const phi::DenseTensor& input,
                  phi::DenseTensor* out,
                  phi::DenseTensor* mid,
                  int N,
                  int C,
                  int H,
                  int W,
                  int n,
                  T k,
                  T alpha,
                  T beta,
                  const DataLayout data_layout = DataLayout::kAnyLayout);
};

template <typename T, typename Context>
void LRNKernel(const Context& dev_ctx,
               const DenseTensor& x,
               int n,
               T k,
               T alpha,
               T beta,
               const std::string& data_format,
               DenseTensor* out,
               DenseTensor* mid_out) {
  // f(x) = x * ( k + alpha * SUM((x)^2) )^(-beta)
  // x represents inputs
  // f(x) represents outputs
  // input
  auto x_dims = x.dims();

  const std::string data_layout_str = data_format;
  const phi::DataLayout data_layout =
      common::StringToDataLayout(data_layout_str);
  // NCHW
  int N = x_dims[0];
  int C = (data_layout != DataLayout::kNHWC ? x_dims[1] : x_dims[3]);
  int H = (data_layout != DataLayout::kNHWC ? x_dims[2] : x_dims[1]);
  int W = (data_layout != DataLayout::kNHWC ? x_dims[3] : x_dims[2]);

  dev_ctx.template Alloc<T>(out);

  // MidOut save the intermediate result for backward
  phi::DenseTensor* mid = mid_out;
  dev_ctx.template Alloc<T>(mid);

  PADDLE_ENFORCE_GE(
      alpha,
      0UL,
      common::errors::InvalidArgument("Argument(alpha) should >= 0.0, "
                                      "but received alpha(%d) less than 0",
                                      alpha));
  PADDLE_ENFORCE_GE(
      beta,
      0UL,
      common::errors::InvalidArgument("Argument(beta) should >= 0.0, "
                                      "but received beta(%d) less than 0",
                                      beta));
  PADDLE_ENFORCE_GE(
      k,
      0UL,
      common::errors::InvalidArgument("Argument(k) should >= 0.0, "
                                      "but received k(%d) less than 0",
                                      k));

  LRNFunctor<Context, T> f;
  f(dev_ctx, x, out, mid, N, C, H, W, n, k, alpha, beta, data_layout);
}

template <typename Context, typename T>
struct LRNGradFunctor {
  void operator()(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::DenseTensor& out,
                  const phi::DenseTensor& mid,
                  phi::DenseTensor* x_g,
                  const phi::DenseTensor& out_g,
                  int N,
                  int C,
                  int H,
                  int W,
                  int n,
                  T alpha,
                  T beta,
                  const DataLayout data_layout = DataLayout::kAnyLayout);
};

/**
 * \brief Backward calculation for normalization with across maps.
 *
 * Function implementation:
 *
 * The implementation of this Function is derived from the
 * CrossMapNormalFunc implementation.
 *
 * InputGrad = OutputGrad * MidOut ^ (-beta)
 *    -- upper
 *  + > (OutputGrad * OutputValue * (-2 * alpha * beta) / MidOut) * InputValue
 *    -- lower
 *
 * The data of inputs/outputs format is the same as the forward interface
 * and is NCHW.
 *
 * The upper and lower is the same as forward. The logic of the sum
 * is also the same as forward.
 */
template <typename T, typename Context>
void LRNGradKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& out,
                   const DenseTensor& mid_out,
                   const DenseTensor& out_grad,
                   int n,
                   T k,
                   T alpha,
                   T beta,
                   const std::string& data_format,
                   DenseTensor* x_grad) {
  const phi::DenseTensor& out_g = out_grad;
  const phi::DenseTensor& mid = mid_out;
  const std::string data_layout_str = data_format;
  const phi::DataLayout data_layout =
      common::StringToDataLayout(data_layout_str);

  auto x_g = x_grad;
  dev_ctx.template Alloc<T>(x_g);

  auto x_dims = x.dims();
  int N = x_dims[0];
  int C = (data_layout != DataLayout::kNHWC ? x_dims[1] : x_dims[3]);
  int H = (data_layout != DataLayout::kNHWC ? x_dims[2] : x_dims[1]);
  int W = (data_layout != DataLayout::kNHWC ? x_dims[3] : x_dims[2]);

  LRNGradFunctor<Context, T> f;
  f(dev_ctx, x, out, mid, x_g, out_g, N, C, H, W, n, alpha, beta, data_layout);
}
}  // namespace phi
