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

template <typename DeviceContext, typename T>
class EighKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto input = ctx.Input<Tensor>("X");
    auto output_w = ctx.Output<Tensor>("Eigenvalues");
    auto output_v = ctx.Output<Tensor>("Eigenvectors");
    std::string lower = ctx.Attr<std::string>("UPLO");
    bool is_lower = (lower == "L");
    math::MatrixEighFunctor<DeviceContext, T> functor;
    functor(ctx, *input, output_w, output_v, is_lower, true);
  }
};

template <typename DeviceContext, typename T>
class EighGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using ValueType = phi::dtype::Real<T>;
    auto& x_grad = *ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    x_grad.mutable_data<T>(ctx.GetPlace());
    auto& output_w = *ctx.Input<Tensor>("Eigenvalues");
    auto& output_v = *ctx.Input<Tensor>("Eigenvectors");
    auto& output_w_grad =
        *ctx.Input<Tensor>(framework::GradVarName("Eigenvalues"));
    auto& output_v_grad =
        *ctx.Input<Tensor>(framework::GradVarName("Eigenvectors"));

    auto& dims = output_v.dims();
    const int m = dims[dims.size() - 1];
    auto dito =
        math::DeviceIndependenceTensorOperations<DeviceContext, T, ValueType>(
            ctx);
    auto tV = dito.Transpose(dito.Conj(output_v));
    auto W = dito.template Sub<ValueType>(dito.Unsqueeze(output_w, -2),
                                          dito.Unsqueeze(output_w, -1));
    Tensor result = dito.Matmul(tV, output_v_grad);
    result.mutable_data<T>(dims, ctx.GetPlace());
    std::vector<int> out_shape = phi::vectorize<int>(dims);
    auto constant = dito.Fill(out_shape, 0.5);
    result = dito.Sub(result, dito.Conj(dito.Transpose(result)));
    result = dito.Mul(result, constant);
    result = dito.Div(result, W);
    result = dito.DiagFill(m, m, m, 0, output_w_grad, result);
    x_grad = dito.Matmul(output_v, dito.Matmul(result, tV));
  }
};

}  // namespace operators
}  // namespace paddle
