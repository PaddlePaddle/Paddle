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
#include "paddle/framework/selected_rows.h"

namespace paddle {
namespace operators {

template <typename Place, typename T>
struct SparseSGDFunctor {
  void operator()(const platform::DeviceContext& context,
                  const framework::SelectedRows& input,
                  const framework::Tensor& learning_rate,
                  framework::Tensor* output);
};

template <typename Place, typename T>
class SGDOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* param = ctx.Input<framework::Tensor>("Param");
    auto* param_out = ctx.Output<framework::Tensor>("ParamOut");
    auto* learning_rate = ctx.Input<framework::Tensor>("LearningRate");

    auto* grad_var = ctx.InputVar("Grad");
    // Actually, all tensors are LoDTensor except SelectedRows.
    if (grad_var->IsType<framework::LoDTensor>()) {
      param_out->mutable_data<T>(ctx.GetPlace());
      auto* grad = ctx.Input<framework::Tensor>("Grad");

      auto p = framework::EigenVector<T>::Flatten(*param);
      auto g = framework::EigenVector<T>::Flatten(*grad);
      auto o = framework::EigenVector<T>::Flatten(*param_out);
      auto lr = framework::EigenVector<T>::Flatten(*learning_rate);
      auto place = ctx.GetEigenDevice<Place>();

      Eigen::DSizes<int, 1> grad_dsize(grad->numel());
      o.device(place) = p - lr.broadcast(grad_dsize) * g;
    } else if (grad_var->IsType<framework::SelectedRows>()) {
      // TODO(qijun): In Sparse SGD operator, in-place update is enforced.
      // This manual optimization brings difficulty to track data dependency.
      // It's better to find a more elegant solution.
      PADDLE_ENFORCE_EQ(param, param_out);
      auto* grad = ctx.Input<framework::SelectedRows>("Grad");
      SparseSGDFunctor<Place, T> functor;
      functor(ctx.device_context(), *grad, *learning_rate, param_out);
    } else {
      PADDLE_THROW("Unsupported Variable Type of Grad");
    }
  }
};
}  // namespace operators
}  // namespace paddle
