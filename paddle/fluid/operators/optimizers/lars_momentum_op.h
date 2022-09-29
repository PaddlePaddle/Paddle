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
    auto param_out = ctx.MultiOutput<framework::LoDTensor>("ParamOut");
    auto velocity_out = ctx.MultiOutput<framework::LoDTensor>("VelocityOut");
    auto param = ctx.MultiInput<framework::LoDTensor>("Param");
    auto velocity = ctx.MultiInput<framework::LoDTensor>("Velocity");
    auto learning_rate = ctx.MultiInput<framework::LoDTensor>("LearningRate");
    auto grad = ctx.MultiInput<framework::LoDTensor>("Grad");
    auto weight_decay_arr = ctx.Attr<std::vector<float>>("lars_weight_decay");
    T mu = static_cast<T>(ctx.Attr<float>("mu"));
    T lars_coeff = ctx.Attr<float>("lars_coeff");
    T epsilon = ctx.Attr<float>("epsilon");
    T rescale_grad = ctx.Attr<float>("rescale_grad");

    int op_num = param.size();
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
      auto rescale_g = rescale_grad * g;

      phi::DenseTensor p_norm_t, g_norm_t;
      p_norm_t.Resize({1});
      g_norm_t.Resize({1});
      p_norm_t.mutable_data<T>(ctx.GetPlace());
      g_norm_t.mutable_data<T>(ctx.GetPlace());
      auto ep_norm = framework::EigenScalar<T>::From(p_norm_t);
      auto eg_norm = framework::EigenScalar<T>::From(g_norm_t);
      ep_norm = p.square().sum().sqrt();
      eg_norm = rescale_g.square().sum().sqrt();

      T local_lr = lr[0];
      if (lars_weight_decay > 0 && ep_norm(0) > 0 && eg_norm(0) > 0) {
        local_lr = lr[0] * lars_coeff * ep_norm(0) /
                   (eg_norm(0) + lars_weight_decay * ep_norm(0) + epsilon);
      }
      v_out = v * mu + local_lr * (rescale_g + lars_weight_decay * p);
      p_out = p - v_out;
    }
  }
};

}  // namespace operators
}  // namespace paddle
