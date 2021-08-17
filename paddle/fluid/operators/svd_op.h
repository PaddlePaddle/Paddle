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
#include "paddle/fluid/operators/math/complex_functors.h"
#include "paddle/fluid/operators/svd_helper.h"
#include "paddle/fluid/platform/for_range.h"

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
    PADDLE_ENFORCE_EQ(numel % (rows * cols), 0,
                      platform::errors::InvalidArgument(
                          "batches is not integer, bugs here!"));
    int col_u = full ? rows : k;
    int col_v = full ? cols : k;
    int batches = numel / (rows * cols);
    auto* U_out = U->mutable_data<math::Real<T>>(
        context.GetPlace(),
        size_t(batches * rows * col_u * sizeof(math::Real<T>)));
    auto* VH_out = VH->mutable_data<math::Real<T>>(
        context.GetPlace(),
        size_t(batches * col_v * cols * sizeof(math::Real<T>)));
    auto* S_out = S->mutable_data<math::Real<T>>(
        context.GetPlace(), size_t(batches * k * sizeof(math::Real<T>)));

    /*SVD Use the Eigen Library*/
    math::BatchSvd<T>(x_data, U_out, VH_out, S_out, rows, cols, batches, full);
    /*For Test dito library: TODO: delete*/
    /*
    auto dito =
    math::DeviceIndependenceTensorOperations<platform::CPUDeviceContext,
    T>(context) ;
    Tensor zeros = dito.eye(5, x->type()) ;
    dito.print_matrix(zeros, "Eye") ;
    dito.print_matrix(dito.slice(zeros, {0}, {2}, {5}), "EE") ;
    dito.print_matrix(dito.slice(zeros, {1}, {1}, {4}), "SE") ;
    dito.print_matrix(zeros, "Z")  ;
    dito.print_matrix(dito.matmul(zeros, *x, true, false), "Matmul") ;
    dito.print_matrix(dito.diag(dito.zeros({5}, x->type(), 1.0)), "Diagalize of
    I") ;
    dito.print_matrix(dito.diag(dito.zeros({5,5}, x->type(), 1.0)), "Diag of I")
    ;

    dito.print_matrix(dito.diag(dito.zeros({5,5}, x->type(), 1.0)), "Diag of I")
    ;
    dito.print_matrix(dito.pow(dito.zeros({2,3}, x->type(), 2.0), 0.5), "Test of
    pow of 2.0") ;
    dito.print_matrix(dito.unsqueeze(dito.zeros({5,5}, x->type(), 1.0), -1),
    "Unsqueeze -1") ;
    dito.print_matrix(dito.unsqueeze(dito.zeros({5,5}, x->type(), 1.0), -2),
    "Unsqueeze -2") ;
    */
    /*
    auto dito =
    math::DeviceIndependenceTensorOperations<platform::CPUDeviceContext,
    T>(context) ;
    dito.print_matrix(*x, "X") ;
    Tensor ones = dito.zeros({3, 3}, x->type(), 1.0) ;
    dito.print_matrix(dito.sub(ones, *x), "ones - X: Sub") ;
    */
  }
};

template <typename DeviceContext, typename T>
class SvdGradKernel : public framework::OpKernel<T> {
 private:
  bool GradExist(std::string name) { return true; }

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
      U = dito.slice(U_const, {-1}, {0}, {k});
      VH = dito.slice(VH_const, {-2}, {0}, {k});
      dU = dito.slice(dU_const, {-1}, {0}, {k});
      dVH = dito.slice(dVH_const, {-2}, {0}, {k});
    } else {
      U = U_const;
      VH = VH_const;
      dU = dU_const;
      dVH = dVH_const;
    }
    auto s_inverse = dito.pow(S, -1);
    auto s_square = dito.pow(S, 2);
    auto F =
        dito.sub(dito.unsqueeze(s_square, -2), dito.unsqueeze(s_square, -1));
    F = dito.add(F, dito.diag(dito.infinits({k}, U.type())));
    F = dito.pow(F, -1);
    Tensor sigma_term;
    Tensor u_term;
    Tensor v_term;

    if (ctx.HasInput(framework::GradVarName("S"))) {
      const framework::Tensor& gS =
          *ctx.Input<framework::Tensor>(framework::GradVarName("S"));
      sigma_term = dito.mul(dito.unsqueeze(gS, -2), U);
      sigma_term = dito.matmul(sigma_term, VH);
    }

    if (ctx.HasInput(framework::GradVarName("U"))) {
      auto UTG = dito.matmul(U, dU, true, false);
      auto GTU = dito.matmul(dU, U, true, false);
      u_term = dito.mul(dito.mul(dito.sub(UTG, GTU), F), dito.unsqueeze(S, -2));
      u_term = dito.matmul(U, u_term);
      if (m > k) {
        auto project =
            dito.sub(dito.eye(m, U.type()), dito.matmul(U, U, false, true));
        u_term = dito.add(u_term, dito.mul(dito.matmul(project, dU),
                                           dito.unsqueeze(s_inverse, -2)));
      }
      u_term = dito.matmul(u_term, VH);
    }

    if (ctx.HasInput(framework::GradVarName("VH"))) {
      auto UTG = dito.matmul(VH, dVH, false, true);
      auto GTU = dito.matmul(dVH, VH, false, true);
      v_term = dito.mul(dito.matmul(dito.mul(dito.sub(UTG, GTU), F), VH),
                        dito.unsqueeze(S, -1));
      if (n > k) {
        auto project =
            dito.sub(dito.eye(n, U.type()), dito.matmul(VH, VH, true, false));
        v_term = dito.add(v_term, dito.mul(dito.matmul(dVH, project),
                                           dito.unsqueeze(s_inverse, -1)));
      }
      v_term = dito.matmul(U, v_term);
    }

    dX.ShareDataWith(dito.add(dito.add(u_term, sigma_term), v_term));
  }
};

}  // namespace operators
}  // namespace paddle
