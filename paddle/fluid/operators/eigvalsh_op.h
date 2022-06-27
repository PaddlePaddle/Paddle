// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/eigen_values_vectors.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T,
          int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;

template <typename DeviceContext, typename ValueType, typename T>
class EigvalshKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto input = ctx.Input<Tensor>("X");
    auto output_w = ctx.Output<Tensor>("Eigenvalues");

    std::string lower = ctx.Attr<std::string>("UPLO");
    bool is_lower = (lower == "L");
    bool is_test = ctx.Attr<bool>("is_test");
    math::MatrixEighFunctor<DeviceContext, T> functor;
    if (is_test) {
      functor(ctx, *input, output_w, nullptr, is_lower, false);
    } else {
      auto output_v = ctx.Output<Tensor>("Eigenvectors");
      functor(ctx, *input, output_w, output_v, is_lower, true);
    }
  }
};

template <typename DeviceContext, typename ValueType, typename T>
class EigvalshGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& x_grad = *ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    auto& output_v = *ctx.Input<Tensor>("Eigenvectors");
    auto& output_w_grad =
        *ctx.Input<Tensor>(framework::GradVarName("Eigenvalues"));

    auto dito =
        math::DeviceIndependenceTensorOperations<DeviceContext, T, ValueType>(
            ctx);
    auto tV = dito.Transpose(dito.Conj(output_v));

    // compute elementwise multiply of output_v and output_w_grad
    x_grad.mutable_data<T>(output_v.dims(), ctx.GetPlace());
    auto output_v_vector = EigenVector<T>::Flatten(output_v);
    auto output_w_grad_vector = EigenVector<ValueType>::Flatten(output_w_grad);
    auto result_vector = EigenVector<T>::Flatten(x_grad);
    auto& place = *ctx.template device_context<DeviceContext>().eigen_device();
    std::vector<int> broadcast_factor;
    broadcast_factor.push_back(output_v.dims().at(output_v.dims().size() - 1));
    result_vector.device(place) =
        output_v_vector * output_w_grad_vector.broadcast(broadcast_factor);

    x_grad = dito.Matmul(x_grad, tV);
  }
};

}  // namespace operators
}  // namespace paddle
