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
    size_t n = param.size();
    auto grad = ctx.MultiInput<framework::Tensor>("Grad");
    PADDLE_ENFORCE_EQ(n, grad.size(),
                      platform::errors::InvalidArgument(
                          "The size of Input(Grad) must be equal to "
                          "Input(Param), but got the size of Input(Grad) "
                          "is %d, the size of Input(Param) is %d.",
                          grad.size(), n));
    auto lr = ctx.MultiInput<framework::Tensor>("LearningRate");
    PADDLE_ENFORCE_EQ(
        n, lr.size(),
        platform::errors::InvalidArgument(
            "The size of Input(LearningRate) must be equal to "
            "Input(Param), but got the size of Input(LearningRate) "
            "is %d, the size of Input(Param) is %d.",
            lr.size(), n));
    auto mom1 = ctx.MultiInput<framework::Tensor>("Moment1");
    PADDLE_ENFORCE_EQ(n, mom1.size(),
                      platform::errors::InvalidArgument(
                          "The size of Input(Moment1) must be equal to "
                          "Input(Param), but got the size of Input(Moment1) "
                          "is %d, the size of Input(Param) is %d.",
                          mom1.size(), n));
    auto mom2 = ctx.MultiInput<framework::Tensor>("Moment2");
    PADDLE_ENFORCE_EQ(n, mom2.size(),
                      platform::errors::InvalidArgument(
                          "The size of Input(Moment2) must be equal to "
                          "Input(Param), but got the size of Input(Moment2) "
                          "is %d, the size of Input(Param) is %d.",
                          mom2.size(), n));
    auto beta1_pow = ctx.MultiInput<framework::Tensor>("Beta1Pow");
    PADDLE_ENFORCE_EQ(n, beta1_pow.size(),
                      platform::errors::InvalidArgument(
                          "The size of Input(Beta1Pow) must be equal to "
                          "Input(Param), but got the size of Input(Beta1Pow) "
                          "is %d, the size of Input(Param) is %d.",
                          beta1_pow.size(), n));
    auto beta2_pow = ctx.MultiInput<framework::Tensor>("Beta2Pow");
    PADDLE_ENFORCE_EQ(n, beta2_pow.size(),
                      platform::errors::InvalidArgument(
                          "The size of Input(Beta2Pow) must be equal to "
                          "Input(Param), but got the size of Input(Beta2Pow) "
                          "is %d, the size of Input(Param) is %d.",
                          beta2_pow.size(), n));

    auto param_out = ctx.MultiOutput<framework::Tensor>("ParamOut");
    auto mom1_out = ctx.MultiOutput<framework::Tensor>("Moment1Out");
    auto mom2_out = ctx.MultiOutput<framework::Tensor>("Moment2Out");
    auto beta1_pow_out = ctx.MultiOutput<framework::Tensor>("Beta1PowOut");
    auto beta2_pow_out = ctx.MultiOutput<framework::Tensor>("Beta2PowOut");

    T beta1 = static_cast<T>(ctx.Attr<float>("beta1"));
    T beta2 = static_cast<T>(ctx.Attr<float>("beta2"));
    T epsilon = static_cast<T>(ctx.Attr<float>("epsilon"));
    bool use_global_beta_pow = ctx.Attr<bool>("use_global_beta_pow");
    VLOG(4) << "use_global_beta_pow:" << use_global_beta_pow;

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
      if (!use_global_beta_pow) {
        beta1_pow_out[idx]->mutable_data<T>(ctx.GetPlace())[0] =
            beta1 * beta1_pow[idx]->data<T>()[0];
        beta2_pow_out[idx]->mutable_data<T>(ctx.GetPlace())[0] =
            beta2 * beta2_pow[idx]->data<T>()[0];
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
