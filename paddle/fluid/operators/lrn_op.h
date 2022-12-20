/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include <string>

#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

using DataLayout = phi::DataLayout;

template <typename place, typename T>
struct LRNFunctor {
  void operator()(const framework::ExecutionContext& ctx,
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

template <typename DeviceContext, typename T>
class LRNKernel : public framework::OpKernel<T> {
 public:
  // f(x) = x * ( k + alpha * SUM((x)^2) )^(-beta)
  // x represents inputs
  // f(x) represents outputs
  void Compute(const framework::ExecutionContext& ctx) const override {
    // input
    const phi::DenseTensor& x = *ctx.Input<phi::DenseTensor>("X");
    auto x_dims = x.dims();

    const std::string data_layout_str = ctx.Attr<std::string>("data_format");
    const phi::DataLayout data_layout =
        phi::StringToDataLayout(data_layout_str);
    // NCHW
    int N = x_dims[0];
    int C = (data_layout != DataLayout::kNHWC ? x_dims[1] : x_dims[3]);
    int H = (data_layout != DataLayout::kNHWC ? x_dims[2] : x_dims[1]);
    int W = (data_layout != DataLayout::kNHWC ? x_dims[3] : x_dims[2]);

    phi::DenseTensor* out = ctx.Output<phi::DenseTensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());

    // MidOut save the intermediate result for backward
    phi::DenseTensor* mid = ctx.Output<phi::DenseTensor>("MidOut");
    mid->mutable_data<T>(ctx.GetPlace());

    int n = ctx.Attr<int>("n");
    T alpha = ctx.Attr<float>("alpha");
    T beta = ctx.Attr<float>("beta");
    T k = ctx.Attr<float>("k");

    PADDLE_ENFORCE_GE(
        alpha,
        0UL,
        platform::errors::InvalidArgument("Argument(alpha) should >= 0.0, "
                                          "but received alpha(%d) less than 0",
                                          alpha));
    PADDLE_ENFORCE_GE(
        beta,
        0UL,
        platform::errors::InvalidArgument("Argument(beta) should >= 0.0, "
                                          "but received beta(%d) less than 0",
                                          beta));
    PADDLE_ENFORCE_GE(
        k,
        0UL,
        platform::errors::InvalidArgument("Argument(k) should >= 0.0, "
                                          "but received k(%d) less than 0",
                                          k));

    LRNFunctor<DeviceContext, T> f;
    f(ctx, x, out, mid, N, C, H, W, n, k, alpha, beta, data_layout);
  }
};

template <typename DeviceContext, typename T>
struct LRNGradFunctor {
  void operator()(const framework::ExecutionContext& ctx,
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
template <typename DeviceContext, typename T>
class LRNGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const phi::DenseTensor& x = *ctx.Input<phi::DenseTensor>("X");
    const phi::DenseTensor& out = *ctx.Input<phi::DenseTensor>("Out");
    const phi::DenseTensor& out_g =
        *ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    const phi::DenseTensor& mid = *ctx.Input<phi::DenseTensor>("MidOut");
    const std::string data_layout_str = ctx.Attr<std::string>("data_format");
    const phi::DataLayout data_layout =
        phi::StringToDataLayout(data_layout_str);

    auto x_g = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    x_g->mutable_data<T>(ctx.GetPlace());

    auto x_dims = x.dims();
    int N = x_dims[0];
    int C = (data_layout != DataLayout::kNHWC ? x_dims[1] : x_dims[3]);
    int H = (data_layout != DataLayout::kNHWC ? x_dims[2] : x_dims[1]);
    int W = (data_layout != DataLayout::kNHWC ? x_dims[3] : x_dims[2]);

    int n = ctx.Attr<int>("n");
    T alpha = ctx.Attr<T>("alpha");
    T beta = ctx.Attr<T>("beta");

    PADDLE_ENFORCE_EQ(
        !ctx.Attr<bool>("is_test"),
        true,
        platform::errors::InvalidArgument(
            "is_test attribute should be set to False in training phase. "
            "but received is_test == True in training phase."));

    LRNGradFunctor<DeviceContext, T> f;
    f(ctx, x, out, mid, x_g, out_g, N, C, H, W, n, alpha, beta, data_layout);
  }
};

}  // namespace operators
}  // namespace paddle
