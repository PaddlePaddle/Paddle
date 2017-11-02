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
#include "paddle/framework/eigen.h"
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

template <typename Place, typename T>
struct SparseAdagradFunctor {
  void operator()(const platform::DeviceContext& context,
                  const framework::SelectedRows& grad,
                  const framework::Tensor& learning_rate, T epsilon,
                  framework::Tensor* moment, framework::Tensor* param);
};

template <typename Place, typename T>
class AdagradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto param_out_tensor = ctx.Output<framework::Tensor>("ParamOut");
    auto moment_out_tensor = ctx.Output<framework::Tensor>("MomentOut");

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
      auto lr = framework::EigenVector<T>::Flatten(
          *ctx.Input<framework::Tensor>("LearningRate"));

      auto param_out = framework::EigenVector<T>::Flatten(*param_out_tensor);
      auto moment_out = framework::EigenVector<T>::Flatten(*moment_out_tensor);
      auto place = ctx.GetEigenDevice<Place>();

      moment_out.device(place) = moment + grad * grad;
      Eigen::DSizes<int, 1> m_dsize(moment_out_tensor->numel());
      param_out.device(place) =
          param - lr.broadcast(m_dsize) * grad / (moment_out.sqrt() + epsilon);
    } else if (grad_var->IsType<framework::SelectedRows>()) {
      auto* param = ctx.Input<framework::Tensor>("Param");
      auto* param_out = ctx.Output<framework::Tensor>("ParamOut");
      PADDLE_ENFORCE_EQ(param, param_out);

      auto* moment = ctx.Input<framework::Tensor>("Moment");
      auto* moment_out = ctx.Output<framework::Tensor>("MomentOut");
      PADDLE_ENFORCE_EQ(moment, moment_out);

      SparseAdagradFunctor<Place, T> functor;
      functor(ctx.device_context(), *ctx.Input<framework::SelectedRows>("Grad"),
              *ctx.Input<framework::Tensor>("LearningRate"), epsilon,
              moment_out, param_out);
    } else {
      PADDLE_THROW("Unsupported Variable Type of Grad");
    }
  }
};

}  // namespace operators
}  // namespace paddle
