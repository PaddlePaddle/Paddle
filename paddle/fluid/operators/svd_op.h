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

#include <cstdarg>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/svd_helper.h"
#include "paddle/fluid/platform/for_range.h"
#include "paddle/phi/kernels/funcs/complex_functors.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;
using DDim = framework::DDim;

template <typename T>
class SvdCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* x = context.Input<Tensor>("X");
    Tensor* U = context.Output<Tensor>("U");
    Tensor* VH = context.Output<Tensor>("VH");
    Tensor* S = context.Output<Tensor>("S");
    int full = context.Attr<bool>("full_matrices");

    /*Create Tensors and output, set the dim ...*/
    auto numel = x->numel();
    auto* x_data = x->data<T>();
    auto x_dims = x->dims();
    int rows = x_dims[x_dims.size() - 2];
    int cols = x_dims[x_dims.size() - 1];
    int k = std::min(rows, cols);
    int col_u = full ? rows : k;
    int col_v = full ? cols : k;
    int batches = numel / (rows * cols);
    auto* U_out = U->mutable_data<phi::dtype::Real<T>>(
        context.GetPlace(),
        size_t(batches * rows * col_u * sizeof(phi::dtype::Real<T>)));
    auto* VH_out = VH->mutable_data<phi::dtype::Real<T>>(
        context.GetPlace(),
        size_t(batches * col_v * cols * sizeof(phi::dtype::Real<T>)));
    auto* S_out = S->mutable_data<phi::dtype::Real<T>>(
        context.GetPlace(), size_t(batches * k * sizeof(phi::dtype::Real<T>)));
    /*SVD Use the Eigen Library*/
    math::BatchSvd<T>(x_data, U_out, VH_out, S_out, rows, cols, batches, full);
  }
};

template <typename DeviceContext, typename T>
class SvdGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    const framework::Tensor& U_const = *ctx.Input<framework::Tensor>("U");
    const framework::Tensor& VH_const = *ctx.Input<framework::Tensor>("VH");
    const framework::Tensor& S = *ctx.Input<framework::Tensor>("S");
    framework::Tensor& dX =
        *ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    const framework::Tensor& dU_const =
        *ctx.Input<framework::Tensor>(framework::GradVarName("U"));
    const framework::Tensor& dVH_const =
        *ctx.Input<framework::Tensor>(framework::GradVarName("VH"));

    const bool full = ctx.Attr<bool>("full_matrices");
    int m = dX.dims()[dX.dims().size() - 2];
    int n = dX.dims()[dX.dims().size() - 1];
    int k = S.dims()[S.dims().size() - 1];
    auto dito = math::DeviceIndependenceTensorOperations<DeviceContext, T>(ctx);
    framework::Tensor U, VH, dU, dV, dVH;
    if (full) {
      // if full_matrices is set, slice the U and VT to k columns
      U = dito.Slice(U_const, {-1}, {0}, {k});
      VH = dito.Slice(VH_const, {-2}, {0}, {k});
      dU = dito.Slice(dU_const, {-1}, {0}, {k});
      dVH = dito.Slice(dVH_const, {-2}, {0}, {k});
    } else {
      U = U_const;
      VH = VH_const;
      dU = dU_const;
      dVH = dVH_const;
    }
    auto s_inverse = dito.Pow(S, -1);
    auto s_square = dito.Pow(S, 2);
    auto F =
        dito.Sub(dito.Unsqueeze(s_square, -2), dito.Unsqueeze(s_square, -1));
    F = dito.Add(F, dito.Diag(dito.Infinits({k})));
    F = dito.Pow(F, -1);
    Tensor sigma_term;
    Tensor u_term;
    Tensor v_term;

    if (ctx.HasInput(framework::GradVarName("S"))) {
      const framework::Tensor& gS =
          *ctx.Input<framework::Tensor>(framework::GradVarName("S"));
      sigma_term = dito.Mul(dito.Unsqueeze(gS, -2), U);
      sigma_term = dito.Matmul(sigma_term, VH);
    }

    if (ctx.HasInput(framework::GradVarName("U"))) {
      auto UTG = dito.Matmul(U, dU, true, false);
      auto GTU = dito.Matmul(dU, U, true, false);
      u_term = dito.Mul(dito.Mul(dito.Sub(UTG, GTU), F), dito.Unsqueeze(S, -2));
      u_term = dito.Matmul(U, u_term);
      if (m > k) {
        auto project = dito.Sub(dito.Eye(m), dito.Matmul(U, U, false, true));
        u_term = dito.Add(u_term, dito.Mul(dito.Matmul(project, dU),
                                           dito.Unsqueeze(s_inverse, -2)));
      }
      u_term = dito.Matmul(u_term, VH);
    }

    if (ctx.HasInput(framework::GradVarName("VH"))) {
      auto UTG = dito.Matmul(VH, dVH, false, true);
      auto GTU = dito.Matmul(dVH, VH, false, true);
      v_term = dito.Mul(dito.Matmul(dito.Mul(dito.Sub(UTG, GTU), F), VH),
                        dito.Unsqueeze(S, -1));
      if (n > k) {
        auto project = dito.Sub(dito.Eye(n), dito.Matmul(VH, VH, true, false));
        v_term = dito.Add(v_term, dito.Mul(dito.Matmul(dVH, project),
                                           dito.Unsqueeze(s_inverse, -1)));
      }
      v_term = dito.Matmul(U, v_term);
    }

    dX.ShareDataWith(dito.Add(dito.Add(u_term, sigma_term), v_term));
  }
};

}  // namespace operators
}  // namespace paddle
