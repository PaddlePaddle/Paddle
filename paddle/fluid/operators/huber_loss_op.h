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
#include "paddle/pten/core/hostdevice.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;

template <typename T>
struct HuberLossForward {
  HOSTDEVICE HuberLossForward(const T& delta) : delta(delta) {}

  HOSTDEVICE T operator()(const T& val) const {
    T abs_val = std::abs(val);
    if (abs_val <= delta) {
      return static_cast<T>(0.5) * val * val;
    } else {
      return delta * (abs_val - static_cast<T>(0.5) * delta);
    }
  }

  T delta;
};

template <typename DeviceContext, typename T>
class HuberLossKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in0 = context.Input<Tensor>("X");
    auto* in1 = context.Input<Tensor>("Y");
    auto* out0 = context.Output<Tensor>("Residual");
    auto* out1 = context.Output<Tensor>("Out");
    auto delta = static_cast<T>(context.Attr<float>("delta"));
    auto& place =
        *context.template device_context<DeviceContext>().eigen_device();

    auto x = EigenVector<T>::Flatten(*in0);
    auto y = EigenVector<T>::Flatten(*in1);
    out0->mutable_data<T>(context.GetPlace());
    auto residual = EigenVector<T>::Flatten(*out0);
    residual.device(place) = y - x;
    out1->mutable_data<T>(context.GetPlace());
    auto loss = EigenVector<T>::Flatten(*out1);
    loss.device(place) = residual.unaryExpr(HuberLossForward<T>(delta));
  }
};

template <typename T>
struct HuberLossBackward {
  HOSTDEVICE HuberLossBackward(const T& delta, T sign)
      : sign(sign), delta(delta) {}

  HOSTDEVICE T operator()(const T& val) const {
    T abs_val = std::abs(val);
    if (abs_val <= delta) {
      return sign * val;
    } else {
      if (val > 0) {
        return sign * delta;
      } else {
        return -1 * sign * delta;
      }
    }
  }

  T sign;
  T delta;
};

template <typename DeviceContext, typename T>
class HuberLossGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in0 = context.Input<Tensor>("Residual");
    auto* in1 = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* out0 = context.Output<Tensor>(framework::GradVarName("X"));
    auto* out1 = context.Output<Tensor>(framework::GradVarName("Y"));
    auto delta = static_cast<T>(context.Attr<float>("delta"));
    auto& place =
        *context.template device_context<DeviceContext>().eigen_device();

    auto residual = EigenVector<T>::Flatten(*in0);
    auto out_grad = EigenVector<T>::Flatten(*in1);

    if (out0) {
      out0->mutable_data<T>(context.GetPlace());
      auto x_grad = EigenVector<T>::Flatten(*out0);
      x_grad.device(place) =
          residual.unaryExpr(HuberLossBackward<T>(delta, -1.0));
      x_grad.device(place) = out_grad * x_grad;
    }

    if (out1) {
      out1->mutable_data<T>(context.GetPlace());
      auto y_grad = EigenVector<T>::Flatten(*out1);
      y_grad.device(place) =
          residual.unaryExpr(HuberLossBackward<T>(delta, 1.0));
      y_grad.device(place) = out_grad * y_grad;
    }
  }
};

}  // namespace operators
}  // namespace paddle
