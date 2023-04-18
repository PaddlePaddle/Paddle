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
#include <string>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/sequence_pooling.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class SequencePoolGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* out_g =
        context.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto* in_g = context.Output<phi::DenseTensor>(framework::GradVarName("X"));
    std::string pooltype = context.Attr<std::string>("pooltype");
    const phi::DenseTensor* index = nullptr;
    if (pooltype == "MAX") {
      index = context.Input<phi::DenseTensor>("MaxIndex");
    }
    in_g->mutable_data<T>(context.GetPlace());
    phi::funcs::SequencePoolGradFunctor<DeviceContext, T> pool;
    pool(context.template device_context<DeviceContext>(),
         pooltype,
         *out_g,
         in_g,
         index);
  }
};

}  // namespace operators
}  // namespace paddle
