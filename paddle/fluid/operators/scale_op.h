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
class ScaleKernel : public framework::OpKernel<T> {
 public:
  virtual void Compute(const framework::ExecutionContext& context) const {
    auto* tensor = context.Output<framework::Tensor>("Out");
    auto* in = context.Input<framework::Tensor>("X");
    tensor->mutable_data<T>(in->place());

    auto scale = static_cast<T>(context.Attr<float>("scale"));

    auto eigen_out = framework::EigenVector<T>::Flatten(*tensor);
    auto eigen_in = framework::EigenVector<T>::Flatten(*in);
    auto& dev =
        *context.template device_context<DeviceContext>().eigen_device();
    eigen_out.device(dev) = scale * eigen_in;
  }
};

}  // namespace operators
}  // namespace paddle
