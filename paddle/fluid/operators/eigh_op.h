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
#include "paddle/fluid/operators/transpose_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using DDim = framework::DDim;
inline DDim EigenvalueDim(const DDim& dim, int k) {
  auto vec = framework::vectorize(dim);
  vec.erase(vec.end() - 2, vec.end());
  vec.push_back(k);
  return framework::make_ddim(vec);
}

template <typename DeviceContext, typename ValueType, typename T>
class EighKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& input_var = *ctx.Input<Tensor>("X");
    auto& output_w_var = *ctx.Output<Tensor>("OutValue");
    auto& output_v_var = *ctx.Output<Tensor>("OutVector");

    std::string lower = ctx.Attr<std::string>("UPLO");
    auto dims = input_var.dims();
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
      input = dito.transpose(input_var);
    }
    int rows = dims[dims.size() - 2];
    int cols = dims[dims.size() - 1];
    int k = std::min(rows, cols);
    auto* x_data = input.mutable_data<T>(dims, ctx.GetPlace());

    auto* value_data = output_w_var.mutable_data<ValueType>(
        EigenvalueDim(dims, k), ctx.GetPlace());
    auto* vector_data = output_v_var.mutable_data<T>(dims, ctx.GetPlace());
    math::BatchEigenvalues<T, ValueType>(x_data, value_data, vector_data,
                                         batch_size, rows, cols, k);
    output_v_var = dito.transpose(output_v_var);
  }
};

template <typename DeviceContext, typename ValueType, typename T>
class EighGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& x_grad = *ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    x_grad.mutable_data<T>(ctx.GetPlace());
    auto& output_w_var = *ctx.Input<Tensor>("OutValue");
    auto& output_v_var = *ctx.Input<Tensor>("OutVector");
    auto& output_w_grad =
        *ctx.Input<Tensor>(framework::GradVarName("OutValue"));
    auto& output_v_grad =
        *ctx.Input<Tensor>(framework::GradVarName("OutVector"));

    auto& dims = output_v_var.dims();
    auto dito =
        math::DeviceIndependenceTensorOperations<DeviceContext, T, ValueType>(
            ctx);
    auto tV = dito.transpose(dito.conj_(output_v_var));
    auto W = dito.sub(dito.unsqueeze(output_w_var, -2),
                      dito.unsqueeze(output_w_var, -1));
    Tensor result = dito.matmul(tV, output_v_grad);
    // auto* result_data = result.mutable_data<T>(dims, ctx.GetPlace());
    //  std::cout << "\n>>>>result: >>>>>>>>>>\n";
    // for(int i=0; i < output_v_var.numel(); i++){
    //   std::cout << result_data[i] << "\t";
    // }
    std::vector<int> out_shape = framework::vectorize<int>(dims);
    auto constant = dito.zeros(out_shape, result.type(), 0.5);

    result = dito.sub(result, dito.conj_(dito.transpose(result)));
    result = dito.mul(result, constant);
    const int m = dims[dims.size() - 1];
    result = dito.div(result, W);
    result = dito.diag_copy(m, m, m, 0, output_w_grad, result);
    x_grad.ShareDataWith(dito.matmul(output_v_var, dito.matmul(result, tV)));
  }
};

}  // namespace operators
}  // namespace paddle
