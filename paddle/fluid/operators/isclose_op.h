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

#include <string>
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
struct GetTensorValue {
  T operator()(const platform::DeviceContext& ctx,
               const framework::Tensor& tensor) const;
};

template <typename DeviceContext, typename T>
struct IscloseFunctor {
  void operator()(const DeviceContext& ctx, const framework::Tensor& in,
                  const framework::Tensor& other, const float rtol,
                  const float atol, bool equal_nan, framework::Tensor* output);
};

template <typename DeviceContext, typename T>
class IscloseKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    // get attrs
    bool equal_nan = ctx.Attr<bool>("equal_nan");
    // get input/output
    const auto* input = ctx.Input<Tensor>("Input");
    const auto* other = ctx.Input<Tensor>("Other");
    auto* out = ctx.Output<Tensor>("Out");

    double rtol_v = std::stod(ctx.Attr<std::string>("rtol"));
    double atol_v = std::stod(ctx.Attr<std::string>("atol"));

    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    GetTensorValue<DeviceContext, double> get_tensor_value;
    if (ctx.HasInput("Rtol")) {
      const auto* rtol = ctx.Input<Tensor>("Rtol");
      PADDLE_ENFORCE_EQ(
          rtol->numel(), 1,
          platform::errors::InvalidArgument(
              "Input(Rtol) size must be 1, but get %d.", rtol->numel()));
      PADDLE_ENFORCE_EQ(rtol->type(), framework::proto::VarType::FP64,
                        platform::errors::InvalidArgument(
                            "Input(Rtol) type must be double, but get %s.",
                            framework::DataTypeToString(rtol->type())));
      rtol_v = get_tensor_value(dev_ctx, *rtol);
    }
    if (ctx.HasInput("Atol")) {
      const auto* atol = ctx.Input<Tensor>("Atol");
      PADDLE_ENFORCE_EQ(
          atol->numel(), 1,
          platform::errors::InvalidArgument(
              "Input(Atol) size must be 1, but get %d", atol->numel()));
      PADDLE_ENFORCE_EQ(atol->type(), framework::proto::VarType::FP64,
                        platform::errors::InvalidArgument(
                            "Input(Atol) type must be double, but get %s",
                            framework::DataTypeToString(atol->type())));
      atol_v = get_tensor_value(dev_ctx, *atol);
    }

    IscloseFunctor<DeviceContext, T>()(dev_ctx, *input, *other, rtol_v, atol_v,
                                       equal_nan, out);
  }
};

}  // namespace operators
}  // namespace paddle
