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

template <typename DeviceContext, typename T>
struct SparseAdagradFunctor {
  void operator()(const DeviceContext& context,
                  const framework::SelectedRows& grad,
                  const framework::Tensor& learning_rate, T epsilon,
                  framework::Tensor* moment, framework::Tensor* param);
};

template <typename DeviceContext, typename T>
class AdagradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* param_out_tensor = ctx.Output<framework::Tensor>("ParamOut");
    auto* moment_out_tensor = ctx.Output<framework::Tensor>("MomentOut");

    param_out_tensor->mutable_data<T>(ctx.GetPlace());
    moment_out_tensor->mutable_data<T>(ctx.GetPlace());

    T epsilon = static_cast<T>(ctx.Attr<float>("epsilon"));

    auto* grad_var = ctx.InputVar("Grad");
    if (grad_var->IsType<framework::LoDTensor>()) {
      auto param = framework::EigenVector<T>::Flatten(
          *ctx.Input<framework::Tensor>("Param"));
      auto grad = framework::EigenVector<T>::Flatten(
          *ctx.Input<framework::Tensor>("Grad"));
      auto moment = framework::EigenVector<T>::Flatten(
          *ctx.Input<framework::Tensor>("Moment"));
      auto* learning_rate = ctx.Input<framework::Tensor>("LearningRate");

      auto param_out = framework::EigenVector<T>::Flatten(*param_out_tensor);
      auto moment_out = framework::EigenVector<T>::Flatten(*moment_out_tensor);
      auto* place = ctx.template device_context<DeviceContext>().eigen_device();

      moment_out.device(*place) = moment + grad * grad;
      Eigen::DSizes<int, 1> m_dsize(moment_out_tensor->numel());
      if (platform::is_cpu_place(ctx.GetPlace())) {
        auto* lr = learning_rate->data<T>();
        param_out.device(*place) =
            param - lr[0] * grad / (moment_out.sqrt() + epsilon);
      } else {
        auto lr = framework::EigenVector<T>::Flatten(*learning_rate);
        param_out.device(*place) =
            param -
            lr.broadcast(m_dsize) * grad / (moment_out.sqrt() + epsilon);
      }
    } else if (grad_var->IsType<framework::SelectedRows>()) {
      auto* param_tensor = ctx.Input<framework::Tensor>("Param");
      PADDLE_ENFORCE_EQ(param_tensor, param_out_tensor);

      auto* moment_tensor = ctx.Input<framework::Tensor>("Moment");
      PADDLE_ENFORCE_EQ(moment_tensor, moment_out_tensor);

      SparseAdagradFunctor<DeviceContext, T> functor;
      functor(ctx.template device_context<DeviceContext>(),
              *ctx.Input<framework::SelectedRows>("Grad"),
              *ctx.Input<framework::Tensor>("LearningRate"), epsilon,
              moment_out_tensor, param_out_tensor);
    } else {
      PADDLE_THROW("Unsupported Variable Type of Grad");
    }
  }
};

}  // namespace operators
}  // namespace paddle
