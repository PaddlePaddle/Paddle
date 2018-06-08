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
#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"
namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class FillZerosArrayKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto& input = *context.Input<framework::LoDTensorArray>("X");
    auto& output = *context.Output<framework::LoDTensorArray>("Out");
    output.resize(input.size());
    for (auto i = 0; i < input.size(); i++) {
      output[i].Resize(input[i].dims());
      output[i].set_lod(input[i].lod());
      output[i].mutable_data<T>(context.GetPlace());
      math::SetConstant<DeviceContext, T> setter;
      setter(context.template device_context<DeviceContext>(), &(output[i]),
             static_cast<T>(0));
    }
  }
};

}  // namespace operators
}  // namespace paddle
