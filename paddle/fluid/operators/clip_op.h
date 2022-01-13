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
#if defined(__NVCC__) || defined(__HIPCC__)
#include "paddle/fluid/operators/elementwise/elementwise_op_impl.cu.h"
#endif

namespace paddle {
namespace operators {

using framework::Tensor;
using platform::Transform;

template <typename T>
class ClipFunctor {
 public:
  explicit ClipFunctor(const T min, const T max) : min_(min), max_(max) {}
  HOSTDEVICE T operator()(const T x) const {
    return x < min_ ? min_ : x > max_ ? max_ : x;
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
    return (y > min_ && y < max_) ? x : static_cast<T>(0);
  }

 private:
  T min_;
  T max_;
};

template <typename DeviceContext, typename T>
class ClipKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto max = static_cast<T>(context.Attr<float>("max"));
    Tensor max_cpu;
    if (context.HasInput("Max")) {
      auto* max_t = context.Input<Tensor>("Max");
      auto* max_data = max_t->data<T>();
      if (platform::is_gpu_place(max_t->place())) {
        paddle::framework::TensorCopySync(*max_t, platform::CPUPlace(),
                                          &max_cpu);
        max_data = max_cpu.data<T>();
      }
      max = max_data[0];
    }
    max = static_cast<T>(max);

    auto min = static_cast<T>(context.Attr<float>("min"));
    Tensor min_cpu;
    if (context.HasInput("Min")) {
      auto* min_t = context.Input<Tensor>("Min");
      auto* min_data = min_t->data<T>();
      if (platform::is_gpu_place(min_t->place())) {
        paddle::framework::TensorCopySync(*min_t, platform::CPUPlace(),
                                          &min_cpu);
        min_data = min_cpu.data<T>();
      }
      min = min_data[0];
    }

    PADDLE_ENFORCE_LE(min, max,
                      platform::errors::InvalidArgument(
                          "max should be greater than or equal to min. "
                          "But received min = %f, max = %f",
                          static_cast<float>(min), static_cast<float>(max)));

    auto* x_var = context.InputVar("X");
    if (x_var->IsType<framework::LoDTensor>()) {
      auto* x = context.Input<framework::LoDTensor>("X");
      auto* out = context.Output<framework::LoDTensor>("Out");
      T* out_data = out->mutable_data<T>(context.GetPlace());
      const T* x_data = x->data<T>();
      int64_t numel = x->numel();
      if (platform::is_gpu_place(context.GetPlace())) {
#if defined(__NVCC__) || defined(__HIPCC__)
        std::vector<const framework::Tensor*> ins = {x};
        std::vector<framework::Tensor*> outs = {out};
        auto functor = ClipFunctor<T>(min, max);
        paddle::operators::LaunchSameDimsElementwiseCudaKernel<
            ElementwiseType::kUnary, T, T>(
            context.template device_context<platform::CUDADeviceContext>(), ins,
            &outs, functor);
#endif
      } else {
        Transform<DeviceContext> trans;
        trans(context.template device_context<DeviceContext>(), x_data,
              x_data + numel, out_data, ClipFunctor<T>(min, max));
      }
    } else if (x_var->IsType<framework::SelectedRows>()) {
      auto* x = context.Input<framework::SelectedRows>("X");
      auto* out = context.Output<framework::SelectedRows>("Out");
      PADDLE_ENFORCE_NE(x, out, platform::errors::InvalidArgument(
                                    "Inplace clip is not allowed "
                                    "when x is SelectedRows"));
      math::scatter::MergeAdd<DeviceContext, T> merge_func;
      merge_func(context.template device_context<DeviceContext>(), *x, out);
      auto* out_tensor = out->mutable_value();
      auto* out_data = out_tensor->data<T>();
      int64_t numel = out_tensor->numel();
      Transform<DeviceContext> trans;
      trans(context.template device_context<DeviceContext>(), out_data,
            out_data + numel, out_data, ClipFunctor<T>(min, max));
    } else {
      PADDLE_THROW(platform::errors::Unavailable(
          "ClipOp only supports LoDTensor and SelectedRows."));
    }
  }
};

template <typename DeviceContext, typename T>
class ClipGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto max = static_cast<T>(context.Attr<float>("max"));
    Tensor max_cpu;
    if (context.HasInput("Max")) {
      auto* max_t = context.Input<Tensor>("Max");
      auto* max_data = max_t->data<T>();
      if (platform::is_gpu_place(max_t->place())) {
        paddle::framework::TensorCopySync(*max_t, platform::CPUPlace(),
                                          &max_cpu);
        max_data = max_cpu.data<T>();
      }
      max = max_data[0];
    }
    max = static_cast<T>(max);

    auto min = static_cast<T>(context.Attr<float>("min"));
    Tensor min_cpu;
    if (context.HasInput("Min")) {
      auto* min_t = context.Input<Tensor>("Min");
      auto* min_data = min_t->data<T>();
      if (platform::is_gpu_place(min_t->place())) {
        paddle::framework::TensorCopySync(*min_t, platform::CPUPlace(),
                                          &min_cpu);
        min_data = min_cpu.data<T>();
      }
      min = min_data[0];
    }
    min = static_cast<T>(min);

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
