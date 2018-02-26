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
#include "paddle/fluid/platform/transform.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using platform::Transform;

template <typename T>
class PReluFunctor {
 public:
  explicit PReluFunctor(const T* alpha) : alpha_(alpha) {}

  HOSTDEVICE T operator()(const T& x) const {
    if (x > 0)
      return x;
    else
      return x * (*alpha_);
  }

 private:
  const T* alpha_;
};

template <typename DeviceContext, typename T>
class PReluKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<Tensor>("X");
    auto* alpha = context.Input<Tensor>("Alpha");
    auto* out = context.Output<Tensor>("Out");

    const T* x_ptr = x->data<T>();
    T* o_ptr = out->mutable_data<T>(context.GetPlace());

    auto* alpha_ptr = alpha->data<T>();

    int numel = x->numel();

    Transform<DeviceContext> trans;
    trans(context.template device_context<DeviceContext>(), x_ptr,
          x_ptr + numel, o_ptr, PReluFunctor<T>(alpha_ptr));
  }
};

template <typename T>
class PReluGradFunctor {
 public:
  explicit PReluGradFunctor(const T* alpha) : alpha_(alpha) {}

  HOSTDEVICE T operator()(const T& out, const T& dout) const {
    if (out > 0)
      return dout;
    else
      return dout * (*alpha_);
  }

 private:
  const T* alpha_;
};

template <typename DeviceContext, typename T>
class PReluGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* dx = context.Output<Tensor>(framework::GradVarName("X"));
    auto* dout = context.Input<Tensor>(framework::GradVarName("Out"));

    auto* out = context.Input<Tensor>("Out");
    auto* alpha = context.Input<Tensor>("Alpha");
    auto* alpha_ptr = alpha->data<T>();

    T* dx_ptr = dx->mutable_data<T>(context.GetPlace());
    const T* dout_ptr = dout->data<T>();
    const T* out_ptr = out->data<T>();
    int numel = dx->numel();

    Transform<DeviceContext> trans;
    trans(context.template device_context<DeviceContext>(), out_ptr,
          out_ptr + numel, dout_ptr, dx_ptr, PReluGradFunctor<T>(alpha_ptr));

    // TODO(Zhuoyuan): add dalpha upgrade when GPU kernels ready
  }
};

}  // namespace operators
}  // namespace paddle
