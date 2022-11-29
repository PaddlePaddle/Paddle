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
#include "paddle/fluid/operators/eigen/eigen_function.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class MinusKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* left_tensor = context.Input<phi::DenseTensor>("X");
    auto* right_tensor = context.Input<phi::DenseTensor>("Y");
    auto* out_tensor = context.Output<phi::DenseTensor>("Out");

    out_tensor->mutable_data<T>(context.GetPlace());
    auto& dev =
        *context.template device_context<DeviceContext>().eigen_device();
    EigenSub<std::decay_t<decltype(dev)>, T>::Eval(
        dev,
        framework::EigenVector<T>::Flatten(*out_tensor),
        framework::EigenVector<T>::Flatten(*left_tensor),
        framework::EigenVector<T>::Flatten(*right_tensor));
  }
};

}  // namespace operators
}  // namespace paddle
