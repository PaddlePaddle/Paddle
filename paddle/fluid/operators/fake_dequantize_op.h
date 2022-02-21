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
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/ddim.h"

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
                  T max_range, const int quant_axis, const int x_num_col_dims,
                  framework::Tensor* out);
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
    auto quant_axis = ctx.Attr<int>("quant_axis");
    auto x_num_col_dims = ctx.Attr<int>("x_num_col_dims");
    int max_range = 1;

    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    out->mutable_data<T>(dev_ctx.GetPlace());
    int scale_num = scales.size();
    if (scale_num == 1) {
      PADDLE_ENFORCE_EQ(
          scales[0]->numel(), in->dims()[quant_axis],
          platform::errors::PreconditionNotMet(
              "The number of first scale values must be the same with "
              "quant_axis dimension value of Input(X) when the `Scales` has "
              "only one element, but %ld != %ld here.",
              scales[0]->numel(), in->dims()[quant_axis]));
      max_range *= (std::pow(2, quant_bits[0] - 1) - 1);
    } else if (scale_num == 2) {
      PADDLE_ENFORCE_EQ(
          scales[0]->numel(), in->dims()[x_num_col_dims],
          platform::errors::PreconditionNotMet(
              "The number of first scale values must be the same with "
              "corresponding dimension value of Input(X) when the `Scales` "
              "has two elements, but %ld != %ld here.",
              scales[0]->numel(), in->dims()[1]));
      PADDLE_ENFORCE_EQ(scales[1]->numel(), 1,
                        platform::errors::PreconditionNotMet(
                            "The second scale tensor should only have one "
                            "value at now, but it has %ld values here.",
                            scales[1]->numel()));
      max_range *= (std::pow(2, quant_bits[0] - 1) - 1) *
                   (std::pow(2, quant_bits[1] - 1) - 1);
    }
    ChannelDequantizeFunctor<DeviceContext, T>()(
        dev_ctx, in, scales.data(), scale_num, static_cast<T>(max_range),
        quant_axis, x_num_col_dims, out);
  }
};

}  // namespace operators
}  // namespace paddle
