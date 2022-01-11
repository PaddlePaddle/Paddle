// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES  // use M_2_SQRTPI on Windows
#endif
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace operators {

// ndtri(x * 0.5 + 0.5) / sqrt(2)
template <typename DeviceContext, typename T>
class ErfinvKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto in = ctx.Input<framework::Tensor>("X");
    auto out = ctx.Output<framework::Tensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());

    auto eigen_in = framework::EigenVector<T>::Flatten(*in);
    auto eigen_out = framework::EigenVector<T>::Flatten(*out);
    auto& place = *ctx.template device_context<DeviceContext>().eigen_device();
    constexpr T half = static_cast<T>(0.5);
    constexpr T half_sqrt = static_cast<T>(M_SQRT1_2);
    eigen_out.device(place) = (eigen_in * half + half).ndtri() * half_sqrt;
  }
};

// sqrt(pi) / 2 * exp(square(out)) * grad
template <typename DeviceContext, typename T>
class ErfinvGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto out = ctx.Input<framework::Tensor>("Out");
    auto dout = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto dx = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    dx->mutable_data<T>(ctx.GetPlace());

    auto eigen_out = framework::EigenVector<T>::Flatten(*out);
    auto eigen_dout = framework::EigenVector<T>::Flatten(*dout);
    auto eigen_dx = framework::EigenVector<T>::Flatten(*dx);
    auto& place = *ctx.template device_context<DeviceContext>().eigen_device();

    constexpr T half_sqrt_pi = static_cast<T>(1 / M_2_SQRTPI);
    eigen_dx.device(place) =
        half_sqrt_pi * eigen_dout * eigen_out.square().exp();
  }
};

}  // namespace operators
}  // namespace paddle
