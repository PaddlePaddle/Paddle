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
class ProximalGDOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* param_out = ctx.Output<Tensor>("ParamOut");

    param_out->mutable_data<T>(ctx.GetPlace());

    auto grad = ctx.Input<Tensor>("Grad");

    auto l1 = static_cast<T>(ctx.Attr<float>("l1"));
    auto l2 = static_cast<T>(ctx.Attr<float>("l2"));

    auto p = EigenVector<T>::Flatten(*ctx.Input<Tensor>("Param"));
    auto g = EigenVector<T>::Flatten(*grad);
    auto lr = EigenVector<T>::Flatten(*ctx.Input<Tensor>("LearningRate"));

    auto p_out = EigenVector<T>::Flatten(*param_out);
    auto& place = *ctx.template device_context<DeviceContext>().eigen_device();

    Eigen::DSizes<int, 1> grad_dsize(grad->numel());

    auto prox_param = p - lr.broadcast(grad_dsize) * g;
    if (l1 > 0) {
      p_out.device(place) =
          prox_param.sign() *
          (((prox_param.abs() - (lr * l1).broadcast(grad_dsize))
                .cwiseMax(T(0.0))) /
           (1.0 + (lr * l2).broadcast(grad_dsize)));
    } else {
      p_out.device(place) =
          prox_param / (1.0 + (lr * l2).broadcast(grad_dsize));
    }
  }
};

}  // namespace operators
}  // namespace paddle
