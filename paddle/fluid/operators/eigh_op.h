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
#include "paddle/phi/kernels/complex_kernel.h"
#include "paddle/phi/kernels/funcs/diag_functor.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/transpose.h"
#include "paddle/phi/kernels/funcs/unsqueeze.h"
#include "paddle/phi/kernels/math_kernel.h"
#include "paddle/phi/kernels/matmul_kernel.h"

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
    using ValueType = phi::funcs::Real<T>;
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

    auto& dev_ctx = context.template device_context<DeviceContext>();
    auto& dev_ctx = static_cast<
        const typename framework::ConvertToPhiContext<DeviceContext>::TYPE&>(
        dev_ctx);
    auto tV = phi::funcs::TransposeLast2Dims<T>(
        dev_ctx, phi::Conj<T>(dev_ctx, output_v));

    auto W =
        phi::Subtract<ValueType>(dev_ctx, phi::funcs::Unsqueeze(output_w, -2),
                                 phi::funcs::Unsqueeze(output_w, -1));
    Tensor result = phi::Matmul<T>(dev_ctx, tV, output_v_grad, false, false);
    result.mutable_data<T>(dims, ctx.GetPlace());
    std::vector<int> out_shape = phi::vectorize<int>(dims);
    auto constant = phi::Full<T>(dev_ctx, out_shape, 0.5);
    result = phi::Subtract<T>(
        dev_ctx, result,
        phi::Conj<T>(dev_ctx,
                     phi::funcs::TransposeLast2Dims<T>(dev_ctx, result)));
    result = phi::Multiply<T>(dev_ctx, result, constant);
    result = phi::Divide<T>(dev_ctx, result, W);
    result = phi::funcs::DiagFill(dev_ctx, m, m, m, 0, output_w_grad, result);
    x_grad =
        phi::Matmul<T>(dev_ctx, output_v, phi::Matmul<T>(dev_ctx, result, tV));
  }
};

}  // namespace operators
}  // namespace paddle
