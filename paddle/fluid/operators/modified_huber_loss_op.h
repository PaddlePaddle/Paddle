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
#include "paddle/phi/core/hostdevice.h"

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;
template <typename T,
          int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;

template <typename T>
struct CheckLabelValue {
  HOSTDEVICE T operator()(const T& val) const {
    PADDLE_ENFORCE_EQ(
        val == static_cast<T>(0) || val == static_cast<T>(1),
        true,
        platform::errors::InvalidArgument(
            "Input(label) value of modified_huber_loss_op expected to be 0 "
            "or 1, but got %ld. Please check label value.",
            val));
  }
};

template <typename T>
struct ModifiedHuberLossForward {
  HOSTDEVICE T operator()(const T& val) const {
    if (val < -1) {
      return -4 * val;
    } else if (val < 1) {
      return (1 - val) * (1 - val);
    } else {
      return static_cast<T>(0);
    }
  }
};

template <typename DeviceContext, typename T>
class ModifiedHuberLossKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in0 = context.Input<phi::DenseTensor>("X");
    auto* in1 = context.Input<phi::DenseTensor>("Y");
    auto* out0 = context.Output<phi::DenseTensor>("IntermediateVal");
    auto* out1 = context.Output<phi::DenseTensor>("Out");

    out0->mutable_data<T>(context.GetPlace());
    out1->mutable_data<T>(context.GetPlace());
    auto& place =
        *context.template device_context<DeviceContext>().eigen_device();

    auto x = EigenVector<T>::Flatten(*in0);
    auto y = EigenVector<T>::Flatten(*in1);
    // make sure value's of Y in {0, 1}
    y.unaryExpr(CheckLabelValue<T>());
    auto inter_val = EigenVector<T>::Flatten(*out0);
    // scale y to {-1, +1} and compute x * y
    inter_val.device(place) = x * (2 * y - static_cast<T>(1));
    auto loss = EigenVector<T>::Flatten(*out1);
    loss.device(place) = inter_val.unaryExpr(ModifiedHuberLossForward<T>());
  }
};

// CPU backward kernel
template <typename T>
class ModifiedHuberLossGradCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in0 = context.Input<phi::DenseTensor>("Y");
    auto* in1 = context.Input<phi::DenseTensor>("IntermediateVal");
    auto* in2 = context.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto* out0 = context.Output<phi::DenseTensor>(framework::GradVarName("X"));

    if (out0) {
      const T* y_ptr = in0->data<T>();
      const T* inter_val_ptr = in1->data<T>();
      const T* out_grad_ptr = in2->data<T>();
      size_t counts = static_cast<size_t>(phi::product(in1->dims()));
      T* x_grad_ptr = out0->mutable_data<T>(context.GetPlace());
      for (size_t i = 0; i < counts; ++i) {
        if (inter_val_ptr[i] < -1) {
          x_grad_ptr[i] = -4 * (2 * y_ptr[i] - 1) * out_grad_ptr[i];
        } else if (inter_val_ptr[i] < 1) {
          x_grad_ptr[i] = -2 * (1 - inter_val_ptr[i]) * (2 * y_ptr[i] - 1) *
                          out_grad_ptr[i];
        } else {
          x_grad_ptr[i] = 0;
        }
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
