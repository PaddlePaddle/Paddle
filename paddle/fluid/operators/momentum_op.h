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

template <typename T>
class MomentumOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    T mu = static_cast<T>(ctx.Attr<float>("mu"));
    bool use_nesterov = ctx.Attr<bool>("use_nesterov");

    auto learning_rate = ctx.Input<framework::Tensor>("LearningRate");
    auto param = ctx.Input<framework::Tensor>("Param");
    auto param_out = ctx.Output<framework::Tensor>("ParamOut");
    auto* velocity_var = ctx.InputVar("Velocity");
    auto* grad_var = ctx.InputVar("Grad");
    if (grad_var->IsType<framework::LoDTensor>()) {
      PADDLE_ENFORCE(velocity_var->IsType<framework::LoDTensor>(),
                     "Unmatched Type of Param and Grad");
      auto velocity = ctx.Input<framework::Tensor>("Velocity");
      auto grad = ctx.Input<framework::Tensor>("Grad");
      auto velocity_out = ctx.Output<framework::Tensor>("VelocityOut");
      param_out->mutable_data<T>(ctx.GetPlace());
      velocity_out->mutable_data<T>(ctx.GetPlace());
      auto p_out = framework::EigenVector<T>::Flatten(*param_out);
      auto v_out = framework::EigenVector<T>::Flatten(*velocity_out);

      auto p = framework::EigenVector<T>::Flatten(*param);
      auto v = framework::EigenVector<T>::Flatten(*velocity);
      auto g = framework::EigenVector<T>::Flatten(*grad);
      auto* lr = learning_rate->data<T>();

      v_out = v * mu + g;
      if (use_nesterov) {
        p_out = p - (g + v_out * mu) * lr[0];
      } else {
        p_out = p - lr[0] * v_out;
      }
    } else if (grad_var->IsType<framework::SelectedRows>()) {
      // sparse update embedding with selectedrows
      PADDLE_ENFORCE(velocity_var->IsType<framework::SelectedRows>(),
                     "Unmatched Type of Param and Grad");
      auto velocity = ctx.Input<framework::SelectedRows>("Velocity");
      auto grad = ctx.Input<framework::SelectedRows>("Grad");
      auto velocity_out = ctx.Output<framework::SelectedRows>("VelocityOut");

      // sparse update maybe empty.
      if (grad->rows().size() == 0) {
        return;
      }
      PADDLE_ENFORCE(grad->height() == velocity->height(),
                     "Unmatched gradient and velocity.");
      auto* p_out = param_out->mutable_data<T>(ctx.GetPlace());
      auto* v_out =
          velocity_out->mutable_value()->mutable_data<T>(ctx.GetPlace());
      auto* lr = learning_rate->data<T>();
      auto* p = param->data<T>();
      auto* g = grad->value().data<T>();
      auto* v = velocity->value().data<T>();
      size_t grad_row_numel = grad->value().numel() / grad->rows().size();

      for (size_t i = 0; i < grad->rows().size(); ++i) {
        size_t grad_row_index = grad->rows()[i];
        for (size_t j = 0; j < grad_row_numel; ++j) {
          size_t p_i = grad_row_index * grad_row_numel + j;
          size_t g_i = i * grad_row_numel + j;
          v_out[g_i] = v[g_i] * mu + g[g_i];
          if (use_nesterov) {
            p_out[p_i] = p[p_i] - (g[g_i] + v_out[g_i] * mu) * lr[0];
          } else {
            p_out[p_i] = p[p_i] - v_out[g_i] * lr[0];
          }
        }
      }
    } else {
      PADDLE_THROW("Unsupported Variable Type of Grad");
    }
  }
};

}  // namespace operators
}  // namespace paddle
