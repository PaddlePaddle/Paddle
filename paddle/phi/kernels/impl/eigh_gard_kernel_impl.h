// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/kernels/funcs/complex_functors.h"

namespace phi {

template <typename T, typename Context>
void EighGardKernel(const Context& dev_ctx,
                    const DenseTensor& out_w,
                    const DenseTensor& out_v,
                    const DenseTensor& dout_w,
                    const DenseTensor& dout_v,
                    DenseTensor* dx) {
  using ValueType = phi::funcs::Real<T>;
  // auto& dx = *ctx.Output<framework::Tensor>(framework::GradVarName("X"));
  dev_ctx.template Alloc<T>(dx);
  // dx.mutable_data<T>(ctx.GetPlace());
  // auto& output_w = *ctx.Input<Tensor>("Eigenvalues");
  // auto& output_v = *ctx.Input<Tensor>("Eigenvectors");
  // auto& output_w_grad =
  //     *ctx.Input<Tensor>(framework::GradVarName("Eigenvalues"));
  // auto& output_v_grad =
  //     *ctx.Input<Tensor>(framework::GradVarName("Eigenvectors"));

  auto& dims = out_v.dims();
  const int m = dims[dims.size() - 1];
  auto dito =
      math::DeviceIndependenceTensorOperations<DeviceContext, T, ValueType>(
          ctx);
  auto tV = dito.Transpose(dito.Conj(out_v));
  auto W = dito.template Sub<ValueType>(dito.Unsqueeze(out_w, -2),
                                        dito.Unsqueeze(out_w, -1));
  DenseTensor result = dito.Matmul(tV, dout_v);

  // result.mutable_data<T>(dims, ctx.GetPlace());
  result.Resize(dims);
  dev_ctx.template Alloc<T>(&result);

  std::vector<int> out_shape = phi::vectorize<int>(dims);
  auto constant = dito.Fill(out_shape, 0.5);
  result = dito.Sub(result, dito.Conj(dito.Transpose(result)));
  result = dito.Mul(result, constant);
  result = dito.Div(result, W);
  result = dito.DiagFill(m, m, m, 0, dout_w, result);
  dx = dito.Matmul(out_v, dito.Matmul(result, tV));
}

}  // namespace phi
