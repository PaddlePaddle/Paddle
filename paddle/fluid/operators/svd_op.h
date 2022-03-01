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
#include "paddle/fluid/platform/for_range.h"
#include "paddle/phi/kernels/funcs/complex_functors.h"
#include "paddle/phi/kernels/funcs/pow.h"
#include "paddle/phi/kernels/funcs/slice.h"
#include "paddle/phi/kernels/math_kernel.h"
#include "paddle/phi/kernels/matmul_kernel.h"

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
    auto* U_out = U->mutable_data<phi::funcs::Real<T>>(
        context.GetPlace(),
        size_t(batches * rows * col_u * sizeof(phi::funcs::Real<T>)));
    auto* VH_out = VH->mutable_data<phi::funcs::Real<T>>(
        context.GetPlace(),
        size_t(batches * col_v * cols * sizeof(phi::funcs::Real<T>)));
    auto* S_out = S->mutable_data<phi::funcs::Real<T>>(
        context.GetPlace(), size_t(batches * k * sizeof(phi::funcs::Real<T>)));
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

    auto& dev_ctx = context.template device_context<DeviceContext>();
    auto& dev_ctx = static_cast<
        const typename framework::ConvertToPhiContext<DeviceContext>::TYPE&>(
        dev_ctx);
    framework::Tensor U, VH, dU, dV, dVH;
    if (full) {
      // if full_matrices is set, slice the U and VT to k columns
      U = phi::funcs::Slice<T>(dev_ctx, U_const, {-1}, {0}, {k});
      VH = phi::funcs::Slice<T>(dev_ctx, VH_const, {-2}, {0}, {k});
      dU = phi::funcs::Slice<T>(dev_ctx, dU_const, {-1}, {0}, {k});
      dVH = phi::funcs::Slice<T>(dev_ctx, dVH_const, {-2}, {0}, {k});
    } else {
      U = U_const;
      VH = VH_const;
      dU = dU_const;
      dVH = dVH_const;
    }
    auto s_inverse = phi::funcs::Pow<T>(dev_ctx, S, -1);
    auto s_square = phi::funcs::Pow<T>(dev_ctx, S, 2);
    auto F =
        phi::Subtract<ValueType>(dev_ctx, phi::funcs::Unsqueeze(s_square, -2),
                                 phi::funcs::Unsqueeze(s_square, -1));
    F = phi::Add<ValueType>(
        dev_ctx, F,
        phi::funcs::Diag<T>(
            dev_ctx,
            phi::Full<T>(
                dev_ctx, {k},
                static_cast<T>(std::numeric_limits<double>::infinity()))));
    F = phi::funcs::Pow<T>(dev_ctx, F, -1);
    Tensor sigma_term;
    Tensor u_term;
    Tensor v_term;

    if (ctx.HasInput(framework::GradVarName("S"))) {
      const framework::Tensor& gS =
          *ctx.Input<framework::Tensor>(framework::GradVarName("S"));
      sigma_term = phi::Multiply<T>(dev_ctx, phi::funcs::Unsqueeze(gS, -2), U);
      sigma_term = phi::Matmul<T>(dev_ctx, sigma_term, VH);
    }

    if (ctx.HasInput(framework::GradVarName("U"))) {
      auto UTG = phi::Matmul<T>(dev_ctx, U, dU, true, false);
      auto GTU = phi::Matmul<T>(dev_ctx, dU, U, true, false);
      u_term = phi::Multiply<T>(
          dev_ctx,
          phi::Multiply<T>(dev_ctx, phi::Subtract<T>(dev_ctx, UTG, GTU), F),
          phi::funcs::Unsqueeze(S, -2));
      u_term = phi::Matmul<T>(dev_ctx, U, u_term);
      if (m > k) {
        auto project = phi::Subtract<T>(
            dev_ctx,
            phi::funcs::Diag<T>(dev_ctx, phi::Full<T>(dev_ctx, {m}, 1)),
            phi::Matmul<T>(dev_ctx, U, U, false, true));
        u_term = phi::Add<T>(
            dev_ctx, u_term,
            phi::Multiply<T>(dev_ctx,
                             phi::Matmul<T>(dev_ctx, project, dU, false, false),
                             phi::funcs::Unsqueeze(s_inverse, -2)));
      }
      u_term = phi::Matmul<T>(dev_ctx, u_term, VH);
    }

    if (ctx.HasInput(framework::GradVarName("VH"))) {
      auto UTG = phi::Matmul<T>(dev_ctx, VH, dVH, false, true);
      auto GTU = phi::Matmul<T>(dev_ctx, dVH, VH, false, true);
      v_term = phi::Multiply<T>(
          dev_ctx,
          phi::Matmul<T>(
              dev_ctx,
              phi::Multiply<T>(dev_ctx, phi::Subtract<T>(dev_ctx, UTG, GTU), F,
                               false, false),
              VH),
          phi::funcs::Unsqueeze(S, -1));
      if (n > k) {
        auto project = phi::Subtract<T>(
            dev_ctx,
            phi::funcs::Diag<T>(dev_ctx, phi::Full<T>(dev_ctx, {m}, 1)),
            phi::Matmul<T>(dev_ctx, VH, VH, true, false));
        v_term = phi::Add<T>(
            dev_ctx, v_term,
            phi::Multiply<T>(
                dev_ctx, phi::Matmul<T>(dev_ctx, dVH, project, false, false),
                phi::funcs::Unsqueeze(s_inverse, -1)));
      }
      v_term = phi::Matmul<T>(dev_ctx, U, v_term, false, false);
    }

    dX.ShareDataWith(
        phi::Add<T>(dev_ctx, phi::Add<T>(dev_ctx, u_term, sigma_term), v_term));
  }
};

}  // namespace operators
}  // namespace paddle
