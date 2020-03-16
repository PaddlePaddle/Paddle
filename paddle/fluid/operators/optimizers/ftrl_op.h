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

using Tensor = framework::Tensor;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;

template <typename DeviceContext, typename T>
class FTRLOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto* param_var = ctx.InputVar("Param");
    PADDLE_ENFORCE(param_var->IsType<framework::LoDTensor>(),
                   "The Var(%s)'s type should be LoDTensor, "
                   "but the received is %s",
                   ctx.InputNames("Param").front(),
                   framework::ToTypeName(param_var->Type()));
    const auto* grad_var = ctx.InputVar("Grad");
    PADDLE_ENFORCE(grad_var->IsType<framework::LoDTensor>(),
                   "The Var(%s)'s type should be LoDTensor, "
                   "but the received is %s",
                   ctx.InputNames("Grad").front(),
                   framework::ToTypeName(grad_var->Type()));

    auto* param_out = ctx.Output<Tensor>("ParamOut");
    auto* sq_accum_out = ctx.Output<Tensor>("SquaredAccumOut");
    auto* lin_accum_out = ctx.Output<Tensor>("LinearAccumOut");

    param_out->mutable_data<T>(ctx.GetPlace());
    sq_accum_out->mutable_data<T>(ctx.GetPlace());
    lin_accum_out->mutable_data<T>(ctx.GetPlace());

    auto grad = ctx.Input<Tensor>("Grad");

    auto l1 = static_cast<T>(ctx.Attr<float>("l1"));
    auto l2 = static_cast<T>(ctx.Attr<float>("l2"));
    auto lr_power = static_cast<T>(ctx.Attr<float>("lr_power"));

    auto p = EigenVector<T>::Flatten(*ctx.Input<Tensor>("Param"));
    auto sq_accum =
        EigenVector<T>::Flatten(*ctx.Input<Tensor>("SquaredAccumulator"));
    auto lin_accum =
        EigenVector<T>::Flatten(*ctx.Input<Tensor>("LinearAccumulator"));
    auto g = EigenVector<T>::Flatten(*grad);
    auto lr = EigenVector<T>::Flatten(*ctx.Input<Tensor>("LearningRate"));

    auto p_out = EigenVector<T>::Flatten(*param_out);
    auto s_acc_out = EigenVector<T>::Flatten(*sq_accum_out);
    auto l_acc_out = EigenVector<T>::Flatten(*lin_accum_out);
    auto& place = *ctx.template device_context<DeviceContext>().eigen_device();

    Eigen::DSizes<int, 1> grad_dsize(grad->numel());

    auto new_accum = sq_accum + g * g;
    // Special case for lr_power = -0.5
    if (lr_power == static_cast<T>(-0.5)) {
      l_acc_out.device(place) =
          lin_accum + g -
          ((new_accum.sqrt() - sq_accum.sqrt()) / lr.broadcast(grad_dsize)) * p;
    } else {
      l_acc_out.device(place) =
          lin_accum + g -
          ((new_accum.pow(-lr_power) - sq_accum.pow(-lr_power)) /
           lr.broadcast(grad_dsize)) *
              p;
    }

    auto x = (l_acc_out.constant(l1) * l_acc_out.sign() - l_acc_out);
    if (lr_power == static_cast<T>(-0.5)) {
      auto y = (new_accum.sqrt() / lr.broadcast(grad_dsize)) +
               l_acc_out.constant(static_cast<T>(2) * l2);
      auto pre_shrink = x / y;
      p_out.device(place) =
          (l_acc_out.abs() > l_acc_out.constant(l1))
              .select(pre_shrink, p.constant(static_cast<T>(0)));
    } else {
      auto y = (new_accum.pow(-lr_power) / lr.broadcast(grad_dsize)) +
               l_acc_out.constant(static_cast<T>(2) * l2);
      auto pre_shrink = x / y;
      p_out.device(place) =
          (l_acc_out.abs() > l_acc_out.constant(l1))
              .select(pre_shrink, p.constant(static_cast<T>(0)));
    }

    s_acc_out.device(place) = sq_accum + g * g;
  }
};

}  // namespace operators
}  // namespace paddle
