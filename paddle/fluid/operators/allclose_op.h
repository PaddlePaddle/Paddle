// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
struct AllcloseFunctor {
  void operator()(const DeviceContext& ctx, const framework::Tensor& in,
                  const framework::Tensor& other, const float rtol,
                  const float atol, bool equal_nan, framework::Tensor* output);
};

template <typename DeviceContext, typename T>
class AllcloseKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    // get attrs
    bool equal_nan = ctx.Attr<bool>("equal_nan");
    // get input/output
    const auto* input = ctx.Input<Tensor>("Input");
    const auto* other = ctx.Input<Tensor>("Other");
    const auto* rtol = ctx.Input<Tensor>("Rtol");
    const auto* atol = ctx.Input<Tensor>("Atol");
    auto* out = ctx.Output<Tensor>("Out");
    auto& dev_ctx = ctx.template device_context<DeviceContext>();

    GetTensorValue<DeviceContext, double> get_tensor_value;
    double rtol_v = get_tensor_value(dev_ctx, *rtol);
    double atol_v = get_tensor_value(dev_ctx, *atol);
    AllcloseFunctor<DeviceContext, T>()(dev_ctx, *input, *other, rtol_v, atol_v,
                                        equal_nan, out);
  }
};

}  // namespace operators
}  // namespace paddle
