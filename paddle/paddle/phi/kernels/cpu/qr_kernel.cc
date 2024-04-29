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

#include "paddle/phi/kernels/qr_kernel.h"

#include <Eigen/Dense>

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/complex_functors.h"
#include "paddle/phi/kernels/funcs/parse_qr_mode.h"

namespace phi {

template <typename T, typename Context>
void QrKernel(const Context& ctx,
              const DenseTensor& x,
              const std::string& mode,
              DenseTensor* q,
              DenseTensor* r) {
  bool compute_q = false;
  bool reduced_mode = false;
  std::tie(compute_q, reduced_mode) = phi::funcs::ParseQrMode(mode);
  auto numel = x.numel();
  PADDLE_ENFORCE_GT(
      numel, 0, errors::PreconditionNotMet("The input of QR is empty."));
  auto x_dims = x.dims();
  int x_rank = x_dims.size();
  int m = static_cast<int>(x_dims[x_rank - 2]);
  int n = static_cast<int>(x_dims[x_rank - 1]);
  int min_mn = std::min(m, n);
  int k = reduced_mode ? min_mn : m;
  int batch_size = static_cast<int>(numel / (m * n));
  int x_stride = m * n;
  int q_stride = m * k;
  int r_stride = k * n;
  auto* x_data = x.data<phi::dtype::Real<T>>();
  T* q_data = nullptr;
  if (compute_q) {
    q_data = ctx.template Alloc<phi::dtype::Real<T>>(
        q, batch_size * m * k * sizeof(phi::dtype::Real<T>));
  }
  auto* r_data = ctx.template Alloc<phi::dtype::Real<T>>(
      r, batch_size * k * n * sizeof(phi::dtype::Real<T>));

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
        auto q_matrix = qr.householderQ() * EigenDynamicMatrix::Identity(m, m);
        q_matrix.transposeInPlace();
        memcpy(q_matrix_ptr, q_matrix.data(), q_matrix.size() * sizeof(T));
      }
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(qr, CPU, ALL_LAYOUT, phi::QrKernel, float, double) {}
