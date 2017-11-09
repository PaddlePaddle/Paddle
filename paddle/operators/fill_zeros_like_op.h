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
#include "paddle/framework/op_registry.h"
#include "paddle/operators/math/math_function.h"

namespace paddle {
namespace operators {

template <typename Place, typename T>
class FillZerosLikeKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* out = context.Output<framework::Tensor>("Y");
    out->mutable_data<T>(context.GetPlace());

    math::SetConstant<Place, T> setter;
    setter(context.device_context(), out, static_cast<T>(0));
  }
};

}  // namespace operators
}  // namespace paddle
