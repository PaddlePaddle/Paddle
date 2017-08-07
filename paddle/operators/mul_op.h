/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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

#include "paddle/operators/math/math_function.h"
#include "paddle/operators/type_alias.h"

namespace paddle {
namespace operators {

template <typename Place, typename T>
class MulKernel : public OpKernel {
public:
  void Compute(const ExecutionContext& context) const override {
    auto* input0 = context.Input<Tensor>("X");
    auto* input1 = context.Input<Tensor>("Y");
    auto* output = context.Output<Tensor>(0);

    output->mutable_data<T>(context.GetPlace());

    paddle::operators::math::template matmul<Place, T>(
        *input0,
        false,
        *input1,
        false,
        1,
        output,
        0,
        &const_cast<platform::DeviceContext&>(context.device_context()));
  }
};

}  // namespace operators
}  // namespace paddle
