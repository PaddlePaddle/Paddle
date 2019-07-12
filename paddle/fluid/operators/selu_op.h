/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

template <typename T>
struct SeluFunctor {
  SeluFunctor(const T* x_data_ptr, float alpha, float scale, T* y_data_ptr)
      : x_data_ptr_(x_data_ptr),
        alpha_(alpha),
        scale_(scale),
        y_data_ptr_(y_data_ptr) {}

  HOSTDEVICE void operator()(size_t idx) const {
    T x_ele = x_data_ptr_[idx];
    if (x_ele <= 0) {
      x_ele = alpha_ * real_exp(x_ele) - alpha_;
    }
    y_data_ptr_[idx] = scale_ * x_ele;
  }
  const T* x_data_ptr_;
  const float alpha_;
  const float scale_;
  T* y_data_ptr_;
};

template <typename T>
struct SeluGradFunctor {
  SeluGradFunctor(const T* y_data_ptr, const T* dy_data_ptr, float alpha,
                  float scale, T* dx_data_ptr)
      : y_data_ptr_(y_data_ptr),
        dy_data_ptr_(dy_data_ptr),
        alpha_(alpha),
        scale_(scale),
        la_(alpha * scale),
        dx_data_ptr_(dx_data_ptr) {}

  HOSTDEVICE void operator()(size_t idx) const {
    T y_ele = y_data_ptr_[idx];
    T dy_ele = dy_data_ptr_[idx];

    float tmp = scale_;
    if (y_ele <= 0) {
      tmp = y_ele + la_;
    }
    dx_data_ptr_[idx] = dy_ele * tmp;
  }
  const T* y_data_ptr_;
  const T* dy_data_ptr_;
  const float alpha_;
  const float scale_;
  const float la_;
  T* dx_data_ptr_;
};

template <typename DeviceContext, typename T>
class SeluKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    using Tensor = framework::Tensor;

    auto* x = context.Input<Tensor>("X");
    auto* out = context.Output<Tensor>("Out");

    float alpha = context.Attr<float>("alpha");
    float scale = context.Attr<float>("scale");

    auto out_ptr = out->mutable_data<T>(context.GetPlace());

    SeluFunctor<T> functor(x->data<T>(), alpha, scale, out_ptr);

    auto& dev_ctx = context.template device_context<DeviceContext>();
    size_t limit = static_cast<size_t>(x->numel());
    platform::ForRange<DeviceContext> for_range(dev_ctx, limit);
    for_range(functor);
  }
};

template <typename DeviceContext, typename T>
class SeluGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    using Tensor = framework::Tensor;

    auto* out = context.Input<Tensor>("Out");
    auto* dout = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = context.Output<Tensor>(framework::GradVarName("X"));

    float alpha = context.Attr<float>("alpha");
    float scale = context.Attr<float>("scale");

    auto dx_ptr = dx->mutable_data<T>(context.GetPlace());

    SeluGradFunctor<T> functor(out->data<T>(), dout->data<T>(), alpha, scale,
                               dx_ptr);

    auto& dev_ctx = context.template device_context<DeviceContext>();
    size_t limit = static_cast<size_t>(out->numel());
    platform::ForRange<DeviceContext> for_range(dev_ctx, limit);
    for_range(functor);
  }
};

}  // namespace operators
}  // namespace paddle
