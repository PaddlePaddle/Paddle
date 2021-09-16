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

template <typename T, size_t D, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenTensor = framework::EigenTensor<T, D, MajorType, IndexType>;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;

template <typename DeviceContext, typename ValueType, typename T>
class EighKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto input_var = ctx.Input<Tensor>("X");
    auto output_w_var = ctx.Output<Tensor>("Eigenvalues");
    auto output_v_var = ctx.Output<Tensor>("Eigenvectors");
    std::string lower = ctx.Attr<std::string>("UPLO");
    bool is_lower = (lower == "L");
    math::MatrixEighFunctorCPU<DeviceContext, ValueType, T> functor;
    functor(ctx, *input_var, output_w_var, output_v_var, is_lower, true);
  }
};

template <typename DeviceContext, typename ValueType, typename T>
class EighGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& x_grad = *ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    x_grad.mutable_data<T>(ctx.GetPlace());
    auto& output_w_var = *ctx.Input<Tensor>("Eigenvalues");
    auto& output_v_var = *ctx.Input<Tensor>("Eigenvectors");
    auto& output_w_grad =
        *ctx.Input<Tensor>(framework::GradVarName("Eigenvalues"));
    auto& output_v_grad =
        *ctx.Input<Tensor>(framework::GradVarName("Eigenvectors"));

    auto& dims = output_v_var.dims();
    const int m = dims[dims.size() - 1];
    auto dito =
        math::DeviceIndependenceTensorOperations<DeviceContext, T, ValueType>(
            ctx);
    auto tV = dito.Transpose(dito.Conj(output_v_var));
    auto W = dito.Sub_(dito.Unsqueeze(output_w_var, -2),
                       dito.Unsqueeze(output_w_var, -1));
    Tensor result = dito.Matmul(tV, output_v_grad);
    result.mutable_data<T>(dims, ctx.GetPlace());
    std::vector<int> out_shape = framework::vectorize<int>(dims);
    auto constant = dito.Fill(out_shape, 0.5);
    result = dito.Sub(result, dito.Conj(dito.Transpose(result)));
    result = dito.Mul(result, constant);
    result = dito.Div_(result, W);
    result = dito.DiagFill(m, m, m, 0, output_w_grad, result);
    x_grad = dito.Matmul(output_v_var, dito.Matmul(result, tV));
  }
};

}  // namespace operators
}  // namespace paddle
