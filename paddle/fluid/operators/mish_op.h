/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/framework/op_registry.h"
namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
HOSTDEVICE static T CalcSoftplus(T x, float threshold) {
  if (threshold > 0 && x > threshold) {
    return x;
  } else if (threshold > 0 && x < -threshold) {
    return exp(x);
  } else {
    return log1p(exp(x));
  }
}

// expf instead of exp should be used for float type, complement
// and register float kernel separatelly
HOSTDEVICE static float CalcSoftplusFP32(float x, float threshold) {
  if (threshold > 0 && x > threshold) {
    return x;
  } else if (threshold > 0 && x < -threshold) {
    return expf(x);
  } else {
    return log1pf(expf(x));
  }
}

template <typename DeviceContext, typename T>
class MishCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* out = ctx.Output<Tensor>("Out");

    const float threshold = ctx.Attr<float>("threshold");

    const T* x_data = x->data<T>();
    T* out_data = out->mutable_data<T>(ctx.GetPlace());

    int numel = x->numel();
    for (int i = 0; i < numel; i++) {
      T x_d = x_data[i];
      T sp = CalcSoftplus<T>(x_d, threshold);
      out_data[i] = x_d * std::tanh(sp);
    }
  }
};

template <typename DeviceContext>
class MishFP32CPUKernel : public framework::OpKernel<float> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* out = ctx.Output<Tensor>("Out");

    const float threshold = ctx.Attr<float>("threshold");

    const float* x_data = x->data<float>();
    float* out_data = out->mutable_data<float>(ctx.GetPlace());

    int numel = x->numel();
    for (int i = 0; i < numel; i++) {
      float x_d = x_data[i];
      float sp = CalcSoftplusFP32(x_d, threshold);
      out_data[i] = x_d * std::tanh(sp);
    }
  }
};

template <typename DeviceContext, typename T>
class MishGradCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));

    auto threshold = ctx.Attr<float>("threshold");

    const T* x_data = x->data<T>();
    const T* dout_data = dout->data<T>();
    T* dx_data = dx->mutable_data<T>(ctx.GetPlace());

    int numel = x->numel();
    for (int i = 0; i < numel; i++) {
      T x_d = x_data[i];
      T sp = CalcSoftplus<T>(x_d, threshold);
      T tsp = std::tanh(sp);
      T grad_sp = -std::expm1(-sp);
      T grad_tsp = (static_cast<T>(1) - tsp * tsp) * grad_sp;
      dx_data[i] = dout_data[i] * (x_d * grad_tsp + tsp);
    }
  }
};

template <typename DeviceContext>
class MishGradFP32CPUKernel : public framework::OpKernel<float> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));

    auto threshold = ctx.Attr<float>("threshold");

    const float* x_data = x->data<float>();
    const float* dout_data = dout->data<float>();
    float* dx_data = dx->mutable_data<float>(ctx.GetPlace());

    int numel = x->numel();
    for (int i = 0; i < numel; i++) {
      float x_d = x_data[i];
      float sp = CalcSoftplusFP32(x_d, threshold);
      float tsp = std::tanh(sp);
      float grad_sp = -std::expm1f(-sp);
      float grad_tsp = (static_cast<float>(1) - tsp * tsp) * grad_sp;
      dx_data[i] = dout_data[i] * (x_d * grad_tsp + tsp);
    }
  }
};

}  // namespace operators
}  // namespace paddle
