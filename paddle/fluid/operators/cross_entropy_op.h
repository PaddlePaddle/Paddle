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
#include "paddle/fluid/operators/math/cross_entropy.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class CrossEntropyOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* labels = ctx.Input<Tensor>("Label");
    auto* y = ctx.Output<Tensor>("Y");
    y->mutable_data<T>(ctx.GetPlace());

    math::CrossEntropyFunctor<DeviceContext, T>()(
        ctx.template device_context<DeviceContext>(), y, x, labels,
        ctx.Attr<bool>("soft_label"));
  }
};

template <typename T>
class XeSoftlabelGradFunctor {
 public:
  XeSoftlabelGradFunctor(T* dx,
                         const T* dy,     // NOLINT
                         const T* x,      // NOLINT
                         const T* label,  // NOLINT
                         size_t num_classes)
      : dx_(dx), dy_(dy), x_(x), label_(label), num_classes_(num_classes) {}

  HOSTDEVICE void operator()(size_t i) {
    auto row_ids = i / num_classes_;
    dx_[i] = -label_[i] * dy_[row_ids] / x_[i];
  }

 private:
  T* dx_;
  const T* dy_;
  const T* x_;
  const T* label_;
  size_t num_classes_;
};

template <typename T>
class XeGradFunctor {
 public:
  XeGradFunctor(T* dx,
                const T* dy,           // NOLINT
                const T* x,            // NOLINT
                const int64_t* label,  // NOLINT
                size_t num_classes)
      : dx_(dx), dy_(dy), x_(x), label_(label), num_classes_(num_classes) {}

  HOSTDEVICE void operator()(size_t sample_id) {
    auto x_is_true_offset = sample_id * num_classes_ + label_[sample_id];
    for (size_t x_offset = sample_id * num_classes_;
         x_offset < (sample_id + 1) * num_classes_; ++x_offset) {
      dx_[x_offset] = x_offset != x_is_true_offset
                          ? static_cast<T>(0)
                          : -dy_[sample_id] / x_[x_offset];
    }
  }

 private:
  T* dx_;
  const T* dy_;
  const T* x_;
  const int64_t* label_;
  size_t num_classes_;
};

template <typename DeviceContext, typename T>
class CrossEntropyGradientOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* dy = ctx.Input<Tensor>(framework::GradVarName("Y"));
    auto* label = ctx.Input<Tensor>("Label");
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dx_data = dx->mutable_data<T>(ctx.GetPlace());

    int64_t class_num = x->dims()[1];
    if (ctx.Attr<bool>("soft_label")) {
      XeSoftlabelGradFunctor<T> functor(dx_data, dy->data<T>(), x->data<T>(),
                                        label->data<T>(),
                                        static_cast<size_t>(class_num));
      platform::ForRange<DeviceContext> for_range(
          ctx.template device_context<DeviceContext>(),
          static_cast<size_t>(dx->numel()));
      for_range(functor);
    } else {
      XeGradFunctor<T> functor(dx_data, dy->data<T>(), x->data<T>(),
                               label->data<int64_t>(),
                               static_cast<size_t>(class_num));
      platform::ForRange<DeviceContext> for_range(
          ctx.template device_context<DeviceContext>(),
          static_cast<size_t>(dy->numel()));
      for_range(functor);
    }
  }
};

}  // namespace operators
}  // namespace paddle
