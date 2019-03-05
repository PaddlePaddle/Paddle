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
    auto* weight_scales = ctx.Input<framework::Tensor>("WeightScales");
    auto* out = ctx.Output<framework::Tensor>("Out");

    PADDLE_ENFORCE_EQ(weight_scales->numel(), in->dims()[0],
                      "The weight uses the per-channel quantization type, so "
                      "the number of weight scale values must be the same with "
                      "first dimension value of Input(X).");

    int ativation_bits = ctx.Attr<int>("activation_bits");
    int weight_bits = ctx.Attr<int>("weight_bits");
    int range = std::pow(2, weight_bits - 1) - 1;

    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    out->mutable_data<T>(dev_ctx.GetPlace());

    auto dequant = DequantizeFunctor<DeviceContext, T>();
    if (ctx.HasInput("ActivationScale")) {
      auto* activation_scale = ctx.Input<framework::Tensor>("ActivationScale");
      PADDLE_ENFORCE_EQ(activation_scale->numel(), 1,
                        "The activation uses per-layer quantization type, so "
                        "it must have only one value.");
      framework::Tensor cpu_weigth_scales;
      framework::TensorCopy(*weight_scales, platform::CPUPlace(),
                            &cpu_weigth_scales);
      dev_ctx.Wait();
      const T* weight_scales_data = cpu_weigth_scales.data<T>();
      range *= (std::pow(2, ativation_bits - 1) - 1);
      for (int64_t i = 0; i < in->dims()[0]; i++) {
        framework::Tensor one_channel_in = in->Slice(i, i + 1);
        framework::Tensor one_channel_out = out->Slice(i, i + 1);
        auto max_range = range / weight_scales_data[i];
        dequant(dev_ctx, &one_channel_in, activation_scale,
                static_cast<T>(max_range), &one_channel_out);
      }
    } else {
      for (int64_t i = 0; i < in->dims()[0]; i++) {
        framework::Tensor one_channel_in = in->Slice(i, i + 1);
        framework::Tensor one_channel_out = out->Slice(i, i + 1);
        framework::Tensor one_channel_scale = weight_scales->Slice(i, i + 1);
        dequant(dev_ctx, &one_channel_in, &one_channel_scale,
                static_cast<T>(range), &one_channel_out);
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
