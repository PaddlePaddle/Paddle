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
class LarsMomentumOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const bool merge_operation = ctx.Attr<bool>("merge_operation");
    auto param_out = ctx.MultiOutput<framework::LoDTensor>("ParamOut");
    auto velocity_out = ctx.MultiOutput<framework::LoDTensor>("VelocityOut");
    auto param = ctx.MultiInput<framework::LoDTensor>("Param");
    auto velocity = ctx.MultiInput<framework::LoDTensor>("Velocity");
    auto learning_rate = ctx.MultiInput<framework::LoDTensor>("LearningRate");
    auto grad_var = ctx.MultiInputVar("Grad");
    // only support dense for now.
    PADDLE_ENFORCE_EQ(grad_var[0]->IsType<framework::LoDTensor>(), true,
                      platform::errors::InvalidArgument(
                          "The Var(%s)'s type should be LoDTensor, "
                          "but the received is %s",
                          ctx.InputNames("Grad").front(),
                          framework::ToTypeName(grad_var[0]->Type())));
    auto grad = ctx.MultiInput<framework::LoDTensor>("Grad");

    T mu = static_cast<T>(ctx.Attr<float>("mu"));
    T lars_coeff = ctx.Attr<float>("lars_coeff");
    T epsilon = ctx.Attr<float>("epsilon");

    if (!merge_operation) {
      auto* lr = learning_rate[0]->data<T>();
      T lars_weight_decay =
          ctx.Attr<std::vector<float>>("lars_weight_decay")[0];
      param_out[0]->mutable_data<T>(ctx.GetPlace());
      velocity_out[0]->mutable_data<T>(ctx.GetPlace());

      auto p_out = framework::EigenVector<T>::Flatten(*(param_out[0]));
      auto v_out = framework::EigenVector<T>::Flatten(*(velocity_out[0]));
      auto p = framework::EigenVector<T>::Flatten(*(param[0]));
      auto v = framework::EigenVector<T>::Flatten(*(velocity[0]));
      auto g = framework::EigenVector<T>::Flatten(*(grad[0]));

      framework::Tensor p_norm_t, g_norm_t;
      p_norm_t.Resize({1});
      g_norm_t.Resize({1});
      p_norm_t.mutable_data<T>(ctx.GetPlace());
      g_norm_t.mutable_data<T>(ctx.GetPlace());
      auto ep_norm = framework::EigenScalar<T>::From(p_norm_t);
      auto eg_norm = framework::EigenScalar<T>::From(g_norm_t);
      ep_norm = p.square().sum().sqrt();
      eg_norm = g.square().sum().sqrt();

      T local_lr = lr[0];
      if (lars_weight_decay > 0 && ep_norm(0) > 0 && eg_norm(0) > 0) {
        local_lr = lr[0] * lars_coeff * ep_norm(0) /
                   (eg_norm(0) + lars_weight_decay * ep_norm(0) + epsilon);
      }
      v_out = v * mu + local_lr * (g + lars_weight_decay * p);
      p_out = p - v_out;
    } else {
      int op_num = param.size();
      auto weight_decay_arr = ctx.Attr<std::vector<float>>("lars_weight_decay");
      for (int i = 0; i < op_num; ++i) {
        auto* lr = learning_rate[i]->data<T>();
        T lars_weight_decay = weight_decay_arr[i];
        param_out[i]->mutable_data<T>(ctx.GetPlace());
        velocity_out[i]->mutable_data<T>(ctx.GetPlace());

        auto p_out = framework::EigenVector<T>::Flatten(*(param_out[i]));
        auto v_out = framework::EigenVector<T>::Flatten(*(velocity_out[i]));
        auto p = framework::EigenVector<T>::Flatten(*(param[i]));
        auto v = framework::EigenVector<T>::Flatten(*(velocity[i]));
        auto g = framework::EigenVector<T>::Flatten(*(grad[i]));

        framework::Tensor p_norm_t, g_norm_t;
        p_norm_t.Resize({1});
        g_norm_t.Resize({1});
        p_norm_t.mutable_data<T>(ctx.GetPlace());
        g_norm_t.mutable_data<T>(ctx.GetPlace());
        auto ep_norm = framework::EigenScalar<T>::From(p_norm_t);
        auto eg_norm = framework::EigenScalar<T>::From(g_norm_t);
        ep_norm = p.square().sum().sqrt();
        eg_norm = g.square().sum().sqrt();

        T local_lr = lr[0];
        if (lars_weight_decay > 0 && ep_norm(0) > 0 && eg_norm(0) > 0) {
          local_lr = lr[0] * lars_coeff * ep_norm(0) /
                     (eg_norm(0) + lars_weight_decay * ep_norm(0) + epsilon);
        }
        v_out = v * mu + local_lr * (g + lars_weight_decay * p);
        p_out = p - v_out;
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
