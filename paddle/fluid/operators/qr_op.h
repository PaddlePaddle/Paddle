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
#include "paddle/fluid/operators/math/complex_functors.h"
#include "paddle/fluid/platform/for_range.h"

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
    PADDLE_ENFORCE(
        false, "QR received unrecognized mode '", mode,
        "' but expected one of 'reduced' (default), 'r', or 'complete'");
  }
  return std::make_tuple(compute_q, reduced);
}

template <typename T>
class QrCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    bool compute_q;
    bool reduced_mode;
    const Tensor& x = *context.Input<Tensor>("X");
    Tensor& q = *context.Output<Tensor>("Q");
    Tensor& r = *context.Output<Tensor>("R");
    std::string mode = context.Attr<std::string>("mode");
    std::tie(compute_q, reduced_mode) = _parse_qr_mode(mode);

    auto numel = x.numel();
    PADDLE_ENFORCE_GT(numel, 0, platform::errors::PreconditionNotMet(
                                    "The input of QR is empty."));
    auto x_dims = x.dims();
    int x_rank = x_dims.size();
    int m = x_dims[x_rank - 2];
    int n = x_dims[x_rank - 1];
    int min_mn = std::min(m, n);
    int k = reduced_mode ? min_mn : m;
    int batch_size = numel / (m * n);
    int x_stride = m * n;
    int q_stride = m * k;
    int r_stride = k * n;

    auto* x_data = x.data<math::Real<T>>();
    T* q_data = nullptr;
    if (compute_q) {
      q_data = q.mutable_data<math::Real<T>>(
          context.GetPlace(),
          size_t(batch_size * m * k * sizeof(math::Real<T>)));
    }
    auto* r_data = r.mutable_data<math::Real<T>>(
        context.GetPlace(), size_t(batch_size * k * n * sizeof(math::Real<T>)));

    // Implement QR by calling Eigen
    for (int i = 0; i < batch_size; ++i) {
      const T* x_matrix_ptr = x_data + i * x_stride;
      T* r_matrix_ptr = r_data + i * r_stride;
      using EigenDynamicMatrix =
          Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
      auto x_matrix = Eigen::Map<const EigenDynamicMatrix>(x_matrix_ptr, m, n);
      Eigen::HouseholderQR<EigenDynamicMatrix> qr(x_matrix);
      if (reduced_mode) {
        auto qr_top_matrix = qr.matrixQR().block(0, 0, min_mn, n);
        auto r_matrix_view =
            qr_top_matrix.template triangularView<Eigen::Upper>();
        auto r_matrix = EigenDynamicMatrix(r_matrix_view);
        memcpy(r_matrix_ptr, r_matrix.data(), r_matrix.size() * sizeof(T));
      } else {
        auto r_matrix_view =
            qr.matrixQR().template triangularView<Eigen::Upper>();
        auto r_matrix = EigenDynamicMatrix(r_matrix_view);
        memcpy(r_matrix_ptr, r_matrix.data(), r_matrix.size() * sizeof(T));
      }

      if (compute_q) {
        T* q_matrix_ptr = q_data + i * q_stride;
        if (reduced_mode) {
          auto q_matrix =
              qr.householderQ() * EigenDynamicMatrix::Identity(m, min_mn);
          q_matrix.transposeInPlace();
          memcpy(q_matrix_ptr, q_matrix.data(), q_matrix.size() * sizeof(T));
        } else {
          auto q_matrix =
              qr.householderQ() * EigenDynamicMatrix::Identity(m, m);
          q_matrix.transposeInPlace();
          memcpy(q_matrix_ptr, q_matrix.data(), q_matrix.size() * sizeof(T));
        }
      }
    }
  }
};

template <typename DeviceContext, typename T>
class QrGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    PADDLE_ENFORCE(
        false,
        "QR doesn't have the backward kernel now and will be supported soon.");
  }
};

}  // namespace operators
}  // namespace paddle
