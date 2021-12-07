/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/optimizers/adam_op.h"

namespace paddle {
namespace operators {

namespace scatter = paddle::operators::math::scatter;

template <typename DeviceContext, typename T>
class MergedAdamOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto param = ctx.MultiInput<framework::Tensor>("Param");
    auto grad = ctx.MultiInput<framework::Tensor>("Grad");
    auto lr = ctx.MultiInput<framework::Tensor>("LearningRate");
    auto mom1 = ctx.MultiInput<framework::Tensor>("Moment1");
    auto mom2 = ctx.MultiInput<framework::Tensor>("Moment2");
    auto beta1_pow = ctx.MultiInput<framework::Tensor>("Beta1Pow");
    auto beta2_pow = ctx.MultiInput<framework::Tensor>("Beta2Pow");

    auto param_out = ctx.MultiOutput<framework::Tensor>("ParamOut");
    auto mom1_out = ctx.MultiOutput<framework::Tensor>("Moment1Out");
    auto mom2_out = ctx.MultiOutput<framework::Tensor>("Moment2Out");
    auto beta1_pow_out = ctx.MultiOutput<framework::Tensor>("Beta1PowOut");
    auto beta2_pow_out = ctx.MultiOutput<framework::Tensor>("Beta2PowOut");

    T beta1 = static_cast<T>(ctx.Attr<float>("beta1"));
    T beta2 = static_cast<T>(ctx.Attr<float>("beta2"));
    T epsilon = static_cast<T>(ctx.Attr<float>("epsilon"));

    size_t param_num = param.size();
    for (size_t idx = 0; idx < param_num; idx++) {
      AdamFunctor<T, CPUAdam> functor(
          beta1, beta2, epsilon, beta1_pow[idx]->data<T>(),
          beta2_pow[idx]->data<T>(), mom1[idx]->data<T>(),
          mom1_out[idx]->mutable_data<T>(ctx.GetPlace()), mom2[idx]->data<T>(),
          mom2_out[idx]->mutable_data<T>(ctx.GetPlace()), lr[idx]->data<T>(),
          grad[idx]->data<T>(), param[idx]->data<T>(),
          param_out[idx]->mutable_data<T>(ctx.GetPlace()));
      functor(param[idx]->numel());
    }
  }
};

}  // namespace operators
}  // namespace paddle
