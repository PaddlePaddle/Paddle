/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/matrix_inverse.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class InverseKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input = context.Input<framework::Tensor>("Input");
    auto* output = context.Output<framework::Tensor>("Output");
    output->mutable_data<T>(context.GetPlace());

    auto& dev_ctx = context.template device_context<DeviceContext>();
    phi::funcs::MatrixInverseFunctor<DeviceContext, T> mat_inv;
    mat_inv(dev_ctx, *input, output);
  }
};

template <typename DeviceContext, typename T>
class InverseGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* a_inv = context.Input<framework::Tensor>("Output");
    auto* a_inv_grad =
        context.Input<framework::Tensor>(framework::GradVarName("Output"));
    auto* a_grad =
        context.Output<framework::Tensor>(framework::GradVarName("Input"));

    if (a_grad) {
      a_grad->mutable_data<T>(context.GetPlace());

      auto blas = phi::funcs::GetBlas<DeviceContext, T>(context);
      auto& dev_ctx = context.template device_context<DeviceContext>();
      framework::Tensor tmp_out =
          context.AllocateTmpTensor<T, DeviceContext>(a_inv->dims(), dev_ctx);

      auto mat_dim_a0 =
          phi::funcs::CreateMatrixDescriptor(a_inv_grad->dims(), 0, false);
      auto mat_dim_b0 =
          phi::funcs::CreateMatrixDescriptor(a_inv->dims(), 0, true);
      blas.MatMul(*a_inv_grad, mat_dim_a0, *a_inv, mat_dim_b0, T(1), &tmp_out,
                  T(0));

      auto mat_dim_a1 =
          phi::funcs::CreateMatrixDescriptor(a_inv->dims(), 0, true);
      auto mat_dim_b1 =
          phi::funcs::CreateMatrixDescriptor(tmp_out.dims(), 0, false);
      blas.MatMul(*a_inv, mat_dim_a1, tmp_out, mat_dim_b1, T(-1), a_grad, T(0));
    }
  }
};

}  // namespace operators
}  // namespace paddle
