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

#include <vector>
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
struct DequantizeFunctor {
  void operator()(const DeviceContext& dev_ctx, const framework::Tensor* in,
                  const framework::Tensor* scale, T max_range,
                  framework::Tensor* out);
};

template <typename DeviceContext, typename T>
struct ChannelDequantizeFunctor {
  void operator()(const DeviceContext& dev_ctx, const framework::Tensor* in,
                  const framework::Tensor** scales, const int scale_num,
                  T max_range, framework::Tensor* out);
};

template <typename DeviceContext, typename T>
class FakeDequantizeMaxAbsKernel : public framework::OpKernel<T> {
 public:
  virtual void Compute(const framework::ExecutionContext& ctx) const {
    auto* in = ctx.Input<framework::Tensor>("X");
    auto* scale = ctx.Input<framework::Tensor>("Scale");
    auto* out = ctx.Output<framework::Tensor>("Out");

    float max_range = ctx.Attr<float>("max_range");

    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    out->mutable_data<T>(dev_ctx.GetPlace());

    DequantizeFunctor<DeviceContext, T>()(dev_ctx, in, scale,
                                          static_cast<T>(max_range), out);
  }
};

template <typename DeviceContext, typename T>
class FakeChannelWiseDequantizeMaxAbsKernel : public framework::OpKernel<T> {
 public:
  virtual void Compute(const framework::ExecutionContext& ctx) const {
    auto* in = ctx.Input<framework::Tensor>("X");
    auto scales = ctx.MultiInput<framework::Tensor>("Scales");
    auto* out = ctx.Output<framework::Tensor>("Out");

    auto quant_bits = ctx.Attr<std::vector<int>>("quant_bits");
    int max_range = 1;

    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    out->mutable_data<T>(dev_ctx.GetPlace());
    int scale_num = scales.size();
    if (scale_num == 1) {
      PADDLE_ENFORCE_EQ(
          scales[0]->numel(), in->dims()[0],
          "The number of first scale values must be the same with "
          "first dimension value of Input(X) when the `Scales` has only one "
          "element.");
      max_range *= (std::pow(2, quant_bits[0] - 1) - 1);
    } else if (scale_num == 2) {
      PADDLE_ENFORCE_EQ(
          scales[0]->numel(), in->dims()[1],
          "The number of first scale values must be the same with "
          "second dimension value of Input(X) when the `Scales` has two "
          "elements.");
      PADDLE_ENFORCE_EQ(
          scales[1]->numel(), 1,
          "The second scale tensor should only have one value at now.");
      max_range *= (std::pow(2, quant_bits[0] - 1) - 1) *
                   (std::pow(2, quant_bits[1] - 1) - 1);
    }
    ChannelDequantizeFunctor<DeviceContext, T>()(
        dev_ctx, in, scales.data(), scale_num, static_cast<T>(max_range), out);
  }
};

}  // namespace operators
}  // namespace paddle
