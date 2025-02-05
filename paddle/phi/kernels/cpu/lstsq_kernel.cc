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

#include <algorithm>
#include <cmath>
#include <complex>

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/complex_functors.h"
#include "paddle/phi/kernels/funcs/lapack/lapack_function.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/impl/lstsq_kernel_impl.h"
#include "paddle/phi/kernels/lstsq_kernel.h"
#include "paddle/phi/kernels/transpose_kernel.h"

namespace phi {

enum class LapackDriverType : int { Gels, Gelsd, Gelsy, Gelss };

template <typename T, typename Context>
void LstsqKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 const DenseTensor& y,
                 const Scalar& rcond_scaler,
                 const std::string& driver_string,
                 DenseTensor* solution,
                 DenseTensor* residuals,
                 DenseTensor* rank,
                 DenseTensor* singular_values) {
  using ValueType = phi::dtype::Real<T>;

  static auto driver_type = std::unordered_map<std::string, LapackDriverType>(
      {{"gels", LapackDriverType::Gels},
       {"gelsy", LapackDriverType::Gelsy},
       {"gelsd", LapackDriverType::Gelsd},
       {"gelss", LapackDriverType::Gelss}});
  auto driver = driver_type[driver_string];
  T rcond = rcond_scaler.to<T>();

  auto x_dims = x.dims();
  auto y_dims = y.dims();
  int dim_size = x_dims.size();
  int x_stride = phi::GetMatrixStride(x_dims);
  int y_stride = phi::GetMatrixStride(y_dims);
  int batch_count = phi::GetBatchCount(x_dims);
  auto solution_dim = solution->dims();
  int ori_solu_stride = phi::GetMatrixStride(solution_dim);
  int max_solu_stride = std::max(y_stride, ori_solu_stride);
  int min_solu_stride = std::min(y_stride, ori_solu_stride);

  // lapack is a column-major storage, transpose make the input to
  // have a continuous memory layout
  int info = 0;
  int m = static_cast<int>(x_dims[dim_size - 2]);
  int n = static_cast<int>(x_dims[dim_size - 1]);
  int nrhs = static_cast<int>(y_dims[dim_size - 1]);
  int lda = std::max<int>(m, 1);
  int ldb = std::max<int>(1, std::max(m, n));

  DenseTensor* new_x = new DenseTensor();
  new_x->Resize(common::make_ddim({batch_count, m, n}));
  dev_ctx.template Alloc<T>(new_x);
  phi::Copy<Context>(dev_ctx, x, dev_ctx.GetPlace(), true, new_x);

  solution->Resize(common::make_ddim({batch_count, std::max(m, n), nrhs}));
  dev_ctx.template Alloc<T>(solution);

  if (m >= n) {
    phi::Copy<Context>(dev_ctx, y, dev_ctx.GetPlace(), true, solution);
  } else {
    auto* solu_data = solution->data<T>();
    auto* y_data = y.data<T>();
    for (auto i = 0; i < batch_count; i++) {
      for (auto j = 0; j < min_solu_stride; j++) {
        solu_data[i * max_solu_stride + j] = y_data[i * y_stride + j];
      }
    }
  }

  DenseTensor input_x_trans = phi::TransposeLast2Dim<T>(dev_ctx, *new_x);
  DenseTensor input_y_trans = phi::TransposeLast2Dim<T>(dev_ctx, *solution);
  phi::Copy<Context>(dev_ctx, input_x_trans, dev_ctx.GetPlace(), true, new_x);
  phi::Copy<Context>(
      dev_ctx, input_y_trans, dev_ctx.GetPlace(), true, solution);

  auto* x_vector = new_x->data<T>();
  auto* y_vector = solution->data<T>();

  // "gels" divers does not need to compute rank
  int rank_32 = 0;
  int* rank_data = nullptr;
  int* rank_working_ptr = nullptr;
  if (driver != LapackDriverType::Gels) {
    rank_data = dev_ctx.template Alloc<int>(rank);
    rank_working_ptr = rank_data;
  }

  // "gelsd" and "gelss" divers need to compute singular values
  ValueType* s_data = nullptr;
  ValueType* s_working_ptr = nullptr;
  int s_stride = 0;
  if (driver == LapackDriverType::Gelsd || driver == LapackDriverType::Gelss) {
    s_data = dev_ctx.template Alloc<T>(singular_values);
    s_working_ptr = s_data;
    auto s_dims = singular_values->dims();
    s_stride = static_cast<int>(s_dims[s_dims.size() - 1]);
  }

  // "jpvt" is only used for "gelsy" driver
  DenseTensor* jpvt = new DenseTensor();
  int* jpvt_data = nullptr;
  if (driver == LapackDriverType::Gelsy) {
    jpvt->Resize(common::make_ddim({std::max<int>(1, n)}));
    jpvt_data = dev_ctx.template Alloc<int>(jpvt);
  }

  // run once the driver, first to get the optimal workspace size
  int lwork = -1;
  T wkopt = 0.0;
  ValueType rwkopt;
  int iwkopt = 0;

  if (driver == LapackDriverType::Gels) {
    phi::funcs::lapackGels(
        'N', m, n, nrhs, x_vector, lda, y_vector, ldb, &wkopt, lwork, &info);
  } else if (driver == LapackDriverType::Gelsd) {
    phi::funcs::lapackGelsd(m,
                            n,
                            nrhs,
                            x_vector,
                            lda,
                            y_vector,
                            ldb,
                            s_working_ptr,
                            static_cast<ValueType>(rcond),
                            &rank_32,
                            &wkopt,
                            lwork,
                            &rwkopt,
                            &iwkopt,
                            &info);
  } else if (driver == LapackDriverType::Gelsy) {
    phi::funcs::lapackGelsy(m,
                            n,
                            nrhs,
                            x_vector,
                            lda,
                            y_vector,
                            ldb,
                            jpvt_data,
                            static_cast<ValueType>(rcond),
                            &rank_32,
                            &wkopt,
                            lwork,
                            &rwkopt,
                            &info);
  } else if (driver == LapackDriverType::Gelss) {
    phi::funcs::lapackGelss(m,
                            n,
                            nrhs,
                            x_vector,
                            lda,
                            y_vector,
                            ldb,
                            s_working_ptr,
                            static_cast<ValueType>(rcond),
                            &rank_32,
                            &wkopt,
                            lwork,
                            &rwkopt,
                            &info);
  }

  lwork = std::max<int>(1, static_cast<int>(phi::dtype::Real<T>(wkopt)));
  DenseTensor* work = new DenseTensor();
  work->Resize(common::make_ddim({lwork}));
  T* work_data = dev_ctx.template Alloc<T>(work);

  // "rwork" only used for complex inputs and "gelsy/gelsd/gelss" drivers
  DenseTensor* rwork = new DenseTensor();
  ValueType* rwork_data = nullptr;
  if (IsComplexDtype(x.dtype()) && driver != LapackDriverType::Gels) {
    int rwork_len = 0;
    if (driver == LapackDriverType::Gelsy) {
      rwork_len = std::max<int>(1, 2 * n);
    } else if (driver == LapackDriverType::Gelss) {
      rwork_len = std::max<int>(1, 5 * std::min(m, n));
    } else if (driver == LapackDriverType::Gelsd) {
      rwork_len = std::max<int>(1, rwkopt);
    }
    rwork->Resize(common::make_ddim({rwork_len}));
    rwork_data = dev_ctx.template Alloc<ValueType>(rwork);
  }

  // "iwork" workspace array is relevant only for "gelsd" driver
  DenseTensor* iwork = new DenseTensor();
  int* iwork_data = nullptr;
  if (driver == LapackDriverType::Gelsd) {
    iwork->Resize(common::make_ddim({std::max<int>(1, iwkopt)}));
    iwork_data = dev_ctx.template Alloc<int>(iwork);
  }

  for (auto i = 0; i < batch_count; ++i) {
    auto* x_input = &x_vector[i * x_stride];
    auto* y_input = &y_vector[i * max_solu_stride];
    rank_working_ptr = rank_working_ptr ? &rank_data[i] : nullptr;
    s_working_ptr = s_working_ptr ? &s_data[i * s_stride] : nullptr;

    if (driver == LapackDriverType::Gels) {
      phi::funcs::lapackGels(
          'N', m, n, nrhs, x_input, lda, y_input, ldb, work_data, lwork, &info);
    } else if (driver == LapackDriverType::Gelsd) {
      phi::funcs::lapackGelsd(m,
                              n,
                              nrhs,
                              x_input,
                              lda,
                              y_input,
                              ldb,
                              s_working_ptr,
                              static_cast<ValueType>(rcond),
                              &rank_32,
                              work_data,
                              lwork,
                              rwork_data,
                              iwork_data,
                              &info);
    } else if (driver == LapackDriverType::Gelsy) {
      phi::funcs::lapackGelsy(m,
                              n,
                              nrhs,
                              x_input,
                              lda,
                              y_input,
                              ldb,
                              jpvt_data,
                              static_cast<ValueType>(rcond),
                              &rank_32,
                              work_data,
                              lwork,
                              rwork_data,
                              &info);
    } else if (driver == LapackDriverType::Gelss) {
      phi::funcs::lapackGelss(m,
                              n,
                              nrhs,
                              x_input,
                              lda,
                              y_input,
                              ldb,
                              s_working_ptr,
                              static_cast<ValueType>(rcond),
                              &rank_32,
                              work_data,
                              lwork,
                              rwork_data,
                              &info);
    }

    PADDLE_ENFORCE_EQ(
        info,
        0,
        common::errors::PreconditionNotMet(
            "For batch [%d]: Lapack info is not zero but [%d]", i, info));

    if (rank_working_ptr) *rank_working_ptr = static_cast<int>(rank_32);
  }

  DenseTensor tmp_s = phi::TransposeLast2Dim<T>(dev_ctx, *solution);
  phi::Copy<Context>(dev_ctx, tmp_s, dev_ctx.GetPlace(), true, solution);

  if (m > n) {
    auto* solu_data = solution->data<T>();
    for (auto i = 1; i < batch_count; i++) {
      for (auto j = 0; j < min_solu_stride; j++) {
        solu_data[i * min_solu_stride + j] = solu_data[i * max_solu_stride + j];
      }
    }
  }

  if (batch_count > 1) {
    solution->Resize(solution_dim);
  } else {
    solution->Resize(common::make_ddim({n, nrhs}));
  }

  GetResidualsTensor<Context, T>(dev_ctx, x, y, solution, residuals);
}

}  // namespace phi

PD_REGISTER_KERNEL(lstsq, CPU, ALL_LAYOUT, phi::LstsqKernel, float, double) {
  kernel->OutputAt(2).SetDataType(phi::DataType::INT32);
}
