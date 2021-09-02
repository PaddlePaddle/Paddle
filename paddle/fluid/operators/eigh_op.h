// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/operators/eigh_helper.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using DDim = framework::DDim;

template <typename DeviceContext, typename ValueType, typename T>
class EighKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& input_var = *ctx.Input<Tensor>("X");
    auto& output_w_var = *ctx.Output<Tensor>("OutValue");
    auto& output_v_var = *ctx.Output<Tensor>("OutVector");

    std::string lower = ctx.Attr<std::string>("UPLO");
    auto dims = input_var.dims();
    auto output_value_dim = output_w_var.dims();

    int64_t batch_size = 1;
    int dim_size = dims.size();
    for (int64_t i = 0; i < dim_size - 2; i++) {
      batch_size *= dims[i];
    }
    auto dito =
        math::DeviceIndependenceTensorOperations<DeviceContext, T, ValueType>(
            ctx);
    Tensor input = input_var;
    if (lower == "U") {
      input = dito.Transpose(input_var);
    }
    int rows = dims[dims.size() - 2];
    int cols = dims[dims.size() - 1];

    auto* value_data =
        output_w_var.mutable_data<ValueType>(output_value_dim, ctx.GetPlace());

    if (framework::IsComplexType(input_var.type())) {
      auto* x_data = input.mutable_data<T>(dims, ctx.GetPlace());
      auto* vector_data = output_v_var.mutable_data<T>(dims, ctx.GetPlace());
      math::BatchComplexValues<T, ValueType>(x_data, value_data, vector_data,
                                             batch_size, rows, cols);
    } else {
      auto* x_data = input.mutable_data<ValueType>(dims, ctx.GetPlace());
      auto* vector_data =
          output_v_var.mutable_data<ValueType>(dims, ctx.GetPlace());
      math::BatchEigenvalues<ValueType>(x_data, value_data, vector_data,
                                        batch_size, rows, cols);
    }
    output_v_var = dito.Transpose(output_v_var);
  }
};

template <typename DeviceContext, typename ValueType, typename T>
class EighGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& x_grad = *ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    x_grad.mutable_data<T>(ctx.GetPlace());
    auto& output_w_var = *ctx.Input<Tensor>("OutValue");   // ValueType
    auto& output_v_var = *ctx.Input<Tensor>("OutVector");  // T
    auto& output_w_grad =
        *ctx.Input<Tensor>(framework::GradVarName("OutValue"));
    auto& output_v_grad =
        *ctx.Input<Tensor>(framework::GradVarName("OutVector"));

    auto& dims = output_v_var.dims();
    int batch_size = 1;
    for (int i = 0; i < dims.size() - 2; i++) {
      batch_size *= dims[i];
    }
    int cols = dims[dims.size() - 1];
    auto dito =
        math::DeviceIndependenceTensorOperations<DeviceContext, T, ValueType>(
            ctx);

    Tensor conj_res = output_v_var;
    if (framework::IsComplexType(output_v_var.type())) {
      conj_res = dito.Conj(output_v_var);
    }
    auto tV = dito.Transpose(conj_res);
    Tensor w_sub;
    w_sub =
        dito.SubBroadcast(dito.Unsqueeze(output_w_var, -2),
                          dito.Unsqueeze(output_w_var, -1), batch_size, cols);

    Tensor result = dito.Matmul(tV, output_v_grad);
    auto res_trans = dito.Transpose(result);
    if (framework::IsComplexType(output_v_var.type())) {
      res_trans = dito.Conj(res_trans);
    }
    result = dito.Sub(result, res_trans);
    result = dito.Mul(result, 0.5);
    result = dito.Div(result, w_sub);
    result = dito.DiagFill(cols, cols, cols, 0, output_w_grad, result);
    x_grad = dito.Matmul(output_v_var, dito.Matmul(result, tV));
  }
};

}  // namespace operators
}  // namespace paddle
