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

#include <Eigen/Dense>
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

static inline std::tuple<bool, bool> _parse_qr_mode(std::string mode) {
  bool compute_q;
  bool reduced;
  if (mode == "reduced") {
    compute_q = true;
    reduced = true;
  } else if (mode == "complete") {
    compute_q = true;
    reduced = false;
  } else if (mode == "r") {
    compute_q = false;
    reduced = true;
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "QR received unrecognized mode '%s'"
        " but expected one of 'reduced' (default), 'r', or 'complete'",
        mode));
  }
  return std::make_tuple(compute_q, reduced);
}

template <typename DeviceContext, typename T>
class QrGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    const framework::Tensor& Q = *ctx.Input<framework::Tensor>("Q");
    const framework::Tensor& R = *ctx.Input<framework::Tensor>("R");
    // Use a different name A instead of X
    const framework::Tensor& A = *ctx.Input<framework::Tensor>("X");
    const framework::Tensor& dQ =
        *ctx.Input<framework::Tensor>(framework::GradVarName("Q"));
    const framework::Tensor& dR =
        *ctx.Input<framework::Tensor>(framework::GradVarName("R"));
    // Use a different name dA instead of dX
    framework::Tensor& dA =
        *ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    dA.mutable_data<phi::dtype::Real<T>>(ctx.GetPlace());
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    phi::funcs::SetConstant<DeviceContext, T>()(dev_ctx, &dA, T(0));

    auto dito = math::DeviceIndependenceTensorOperations<DeviceContext, T>(ctx);

    std::string mode = ctx.Attr<std::string>("mode");
    bool compute_q, reduced;
    std::tie(compute_q, reduced) = _parse_qr_mode(mode);
    if (!compute_q) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "The derivative of qr is not implemented when mode='r'."));
    }

    auto a_dims = A.dims();
    int a_rank = a_dims.size();
    int m = a_dims[a_rank - 2];
    int n = a_dims[a_rank - 1];

    if ((m > n) && (!reduced)) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "The derivative of qr is not implemented when mode='complete' and "
          "nrows > ncols."));
    }

    // m >= n case
    auto m_gt_n_case = [](
        const framework::ExecutionContext& ctx,
        math::DeviceIndependenceTensorOperations<DeviceContext, T>& dito,
        const Tensor& dQ, const Tensor& dR, const Tensor& A, const Tensor& Q,
        const Tensor& R) -> framework::Tensor {
      // Hai-Jun Liao, Jin-Guo Liu, Lei Wang, Tao Xiang (2019). Differentiable
      // Programming Tensor Networks.
      // https://arxiv.org/abs/1903.09650 Section 3. QR factorization

      // dR^H
      framework::Tensor R_term;
      if (ctx.HasInput(framework::GradVarName("R"))) {
        R_term = dito.Matmul(R, dito.Transpose(dR));
      } else {
        R_term = dito.Fill(phi::vectorize<int>(R.dims()), 0);
      }

      // dQ^H * Q
      framework::Tensor Q_term;
      if (ctx.HasInput(framework::GradVarName("Q"))) {
        Q_term = dito.Matmul(dito.Transpose(dQ), Q);
      } else {
        Q_term = dito.Fill(phi::vectorize<int>(R.dims()), 0);
      }

      framework::Tensor M_tmp1 = dito.Sub(R_term, Q_term);

      // Compute M = (tril(M) + tril(M).mH()) * 0.5 Identity
      framework::Tensor M_tril_0 = dito.TrilTriu(M_tmp1, 0, true);
      framework::Tensor M_tril_1 = dito.TrilTriu(M_tmp1, -1, true);
      framework::Tensor M = dito.Add(M_tril_0, dito.Transpose(M_tril_1));

      framework::Tensor rhs_term;
      if (ctx.HasInput(framework::GradVarName("Q"))) {
        rhs_term = dito.Add(dQ, dito.Matmul(Q, M));
      } else {
        rhs_term = dito.Matmul(Q, M);
      }

      // dA * R^H = rhs_term
      auto dA =
          dito.TriangularSolve(dito.Transpose(dito.Conj(dito.Transpose(R))),
                               dito.Transpose(rhs_term),
                               /*upper=*/true,
                               /*transpose=*/false,
                               /*unitriangular=*/false);

      return dito.Transpose(dA);
    };

    if (m >= n) {
      auto dA_tmp = m_gt_n_case(ctx, dito, dQ, dR, A, Q, R);
      framework::TensorCopy(dA_tmp, dA.place(), &dA);
    } else {
      // If m < n for input matrices A, we partition A = [X|Y] and R = [U|V]
      // Calculate dX and dY individually and concatenate them to get dA
      dA.mutable_data<phi::dtype::Real<T>>(ctx.GetPlace());

      auto Y = dito.Slice(A, {-1}, {m}, {n});
      auto U = dito.Slice(R, {-1}, {0}, {m});
      framework::Tensor dY, dX, dV, dR_tmp, dQ_prime;

      if (ctx.HasInput(framework::GradVarName("R"))) {
        dV = dito.Slice(dR, {-1}, {m}, {n});
        dR_tmp = dito.Slice(dR, {-1}, {0}, {m});
        // Y * dV^H
        dQ_prime = dito.Matmul(Y, dito.Transpose(dV));
      } else {
        dV = dito.Fill(phi::vectorize<int>(Y.dims()), 0);
        dQ_prime = dito.Fill(phi::vectorize<int>(Q.dims()), 0);
      }

      if (ctx.HasInput(framework::GradVarName("Q"))) {
        dQ_prime = dito.Add(dQ_prime, dQ);
      }
      dX = m_gt_n_case(ctx, dito, dQ_prime, dR_tmp, A, Q, U);
      dY = dito.Matmul(Q, dV);
      // Concatenate dX and dY to get dA.
      auto dA_tmp = dito.ConcatTwoTensors(dX, dY, -1);
      framework::TensorCopy(dA_tmp, dA.place(), &dA);
    }
  }
};

template <typename DeviceContext, typename T>
void BatchedGeqrf(const DeviceContext& dev_ctx, int batch_size, int m, int n,
                  T* a, int lda, T* tau, int a_stride, int tau_stride);

template <typename DeviceContext, typename T>
void BatchedOrgqr(const DeviceContext& dev_ctx, int batch_size, int m, int n,
                  int k, T* a, int lda, T* tau, int a_stride, int tau_stride);

}  // namespace operators
}  // namespace paddle
