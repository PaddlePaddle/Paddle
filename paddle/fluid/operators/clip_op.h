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
#include "paddle/fluid/operators/math/selected_rows_functor.h"
#include "paddle/fluid/platform/transform.h"

namespace paddle {
namespace operators {

using framework::Tensor;
using platform::Transform;

template <typename T>
class ClipFunctor {
 public:
  explicit ClipFunctor(const T min, const T max) : min_(min), max_(max) {}
  HOSTDEVICE T operator()(const T& x) const {
    if (x < min_)
      return min_;
    else if (x > max_)
      return max_;
    else
      return x;
  }

 private:
  T min_;
  T max_;
};

template <typename T>
class ClipGradFunctor {
 public:
  explicit ClipGradFunctor(const T min, const T max) : min_(min), max_(max) {}
  HOSTDEVICE T operator()(const T& x, const T& y) const {
    return (y > min_ && y < max_) ? x : 0;
  }

 private:
  T min_;
  T max_;
};

template <typename DeviceContext, typename T>
class ClipKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto max = context.Attr<T>("max");
    auto min = context.Attr<T>("min");
    auto* x_var = context.InputVar("X");
    if (x_var->IsType<framework::LoDTensor>()) {
      auto* x = context.Input<framework::LoDTensor>("X");
      auto* out = context.Output<framework::LoDTensor>("Out");
      T* out_data = out->mutable_data<T>(context.GetPlace());
      const T* x_data = x->data<T>();
      int64_t numel = x->numel();
      Transform<DeviceContext> trans;
      trans(context.template device_context<DeviceContext>(), x_data,
            x_data + numel, out_data, ClipFunctor<T>(min, max));
    } else if (x_var->IsType<framework::SelectedRows>()) {
      auto* x = context.Input<framework::SelectedRows>("X");
      auto* out = context.Output<framework::SelectedRows>("Out");
      PADDLE_ENFORCE_NE(x, out,
                        "Inplace clip is not allowed when x is SelectedRows");
      math::scatter::MergeAdd<DeviceContext, T> merge_func;
      merge_func(context.template device_context<DeviceContext>(), *x, out);
      auto* out_tensor = out->mutable_value();
      auto* out_data = out_tensor->data<T>();
      int64_t numel = out_tensor->numel();
      Transform<DeviceContext> trans;
      trans(context.template device_context<DeviceContext>(), out_data,
            out_data + numel, out_data, ClipFunctor<T>(min, max));
    } else {
      PADDLE_THROW("ClipOp only supports LoDTensor and SelectedRows");
    }
  }
};

template <typename DeviceContext, typename T>
class ClipGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto max = context.Attr<T>("max");
    auto min = context.Attr<T>("min");
    auto* d_out =
        context.Input<framework::LoDTensor>(framework::GradVarName("Out"));
    auto* d_x =
        context.Output<framework::LoDTensor>(framework::GradVarName("X"));
    if (d_x != nullptr) {
      auto* x = context.Input<framework::LoDTensor>("X");
      int64_t numel = d_out->numel();
      auto* d_x_data = d_x->mutable_data<T>(context.GetPlace());
      const T* d_out_data = d_out->data<T>();
      const T* x_data = x->data<T>();
      Transform<DeviceContext> trans;
      trans(context.template device_context<DeviceContext>(), d_out_data,
            d_out_data + numel, x_data, d_x_data, ClipGradFunctor<T>(min, max));
    }
  }
};

}  // namespace operators
}  // namespace paddle
