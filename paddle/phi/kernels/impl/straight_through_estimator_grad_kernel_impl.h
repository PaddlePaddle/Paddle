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

#include "paddle/common/hostdevice.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor_util.h"

namespace paddle {
namespace operators {

template <typename T, typename DeviceContext>
class StraightThroughEstimatorGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *d_out =
        context.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto x_grad_name = framework::GradVarName("X");
    auto *d_x = context.Output<phi::DenseTensor>(x_grad_name);
    PADDLE_ENFORCE_NOT_NULL(
        d_x,
        common::errors::PreconditionNotMet("StraightThroughEstimatorGradKernel "
                                           "doesn't have the output named %s.",
                                           x_grad_name));

    // Initialize dx as same as d_out
    d_x->mutable_data<T>(context.GetPlace());
    framework::TensorCopy(*d_out, context.GetPlace(), d_x);
  }
};

}  // namespace operators
}  // namespace paddle
