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

#include <math.h>
#include <algorithm>
#include <complex>
#include "paddle/fluid/operators/eig_op.h"
#include "paddle/fluid/operators/math/complex_functors.h"
#include "paddle/fluid/operators/math/eigen_values_vectors.h"
#include "paddle/fluid/operators/math/lapack_function.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/matrix_solve.h"
#include "paddle/fluid/operators/svd_helper.h"
#include "paddle/fluid/operators/transpose_op.h"
#include "paddle/fluid/operators/triangular_solve_op.h"
#include "paddle/fluid/platform/dynload/cublas.h"
#include "paddle/fluid/platform/for_range.h"

#define EPSILON 1e-6

namespace paddle {
namespace operators {

using paddle::framework::Tensor;
enum class LapackDriverType : int64_t { Gels, Gelsd, Gelsy, Gelss };

template <typename DeviceContext, typename T>
class LstsqCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    using ValueType = math::Real<T>;

    auto* x = context.Input<Tensor>("X");
    auto* y = context.Input<Tensor>("Y");
    auto rcond = context.Attr<float>("rcond");
    auto driver_string = context.Attr<std::string>("driver");

    static auto driver_type = std::unordered_map<std::string, LapackDriverType>(
        {{"gels", LapackDriverType::Gels},
         {"gelsy", LapackDriverType::Gelsy},
         {"gelsd", LapackDriverType::Gelsd},
         {"gelss", LapackDriverType::Gelss}});
    auto driver = driver_type[driver_string];
    auto* solution = context.Output<Tensor>("Solution");
    auto* residuals = context.Output<Tensor>("Residuals");
    auto* rank = context.Output<Tensor>("Rank");
    auto* singular_values = context.Output<Tensor>("SingularValues");

    auto dito =
        math::DeviceIndependenceTensorOperations<DeviceContext, T>(context);

    // lapack is a column-major storge, transpose make the input to
    // have a continuous memory layout
    Tensor input_x_trans = dito.Transpose(*x);
    Tensor input_y_trans = dito.Transpose(*y);
    auto* x_vector = input_x_trans.data<T>();
    auto* y_vector = input_y_trans.data<T>();

    auto x_dims = x->dims();
    auto y_dims = y->dims();
    int dim_size = x_dims.size();
    int64_t x_stride = MatrixStride(*x);
    int64_t y_stride = MatrixStride(*y);

    int info = 0;
    int m = x_dims[dim_size - 2];
    int n = x_dims[dim_size - 1];
    int nrhs = y_dims[dim_size - 1];
    int lda = std::max<int64_t>(m, 1);
    int ldb = std::max<int64_t>(1, std::max(m, n));

    // "gels" divers does not need to compute rank
    int rank_32 = 0;
    int64_t* rank_data = nullptr;
    int64_t* rank_working_ptr = nullptr;
    if (driver != LapackDriverType::Gels) {
      rank_data = rank->mutable_data<int64_t>(context.GetPlace());
      rank_working_ptr = rank_data;
    }

    // "gelsd" and "gelss" divers need to compute singular values
    ValueType* s_data = nullptr;
    ValueType* s_working_ptr = nullptr;
    int64_t s_stride = 0;
    if (driver == LapackDriverType::Gelsd ||
        driver == LapackDriverType::Gelss) {
      s_data = singular_values->mutable_data<ValueType>(context.GetPlace());
      s_working_ptr = s_data;
      auto s_dims = singular_values->dims();
      s_stride = s_dims[s_dims.size() - 1];
    }

    // "jpvt" is only used for "gelsy" driver
    Tensor jpvt;
    int* jpvt_data = nullptr;
    if (driver == LapackDriverType::Gelsy) {
      jpvt.Resize(framework::make_ddim({std::max<int64_t>(1, n)}));
      jpvt_data = jpvt.mutable_data<int>(context.GetPlace());
    }

    // run once the driver, first to get the optimal workspace size
    int lwork = -1;
    T wkopt;
    ValueType rwkopt;
    int iwkopt = 0;

    if (driver == LapackDriverType::Gels) {
      math::lapackGels('N', m, n, nrhs, x_vector, lda, y_vector, ldb, &wkopt,
                       lwork, &info);
    } else if (driver == LapackDriverType::Gelsd) {
      math::lapackGelsd(m, n, nrhs, x_vector, lda, y_vector, ldb, s_working_ptr,
                        static_cast<ValueType>(rcond), &rank_32, &wkopt, lwork,
                        &rwkopt, &iwkopt, &info);
    } else if (driver == LapackDriverType::Gelsy) {
      math::lapackGelsy(m, n, nrhs, x_vector, lda, y_vector, ldb, jpvt_data,
                        static_cast<ValueType>(rcond), &rank_32, &wkopt, lwork,
                        &rwkopt, &info);
    } else if (driver == LapackDriverType::Gelss) {
      math::lapackGelss(m, n, nrhs, x_vector, lda, y_vector, ldb, s_working_ptr,
                        static_cast<ValueType>(rcond), &rank_32, &wkopt, lwork,
                        &rwkopt, &info);
    }

    lwork = std::max<int>(1, static_cast<int>(math::Real<T>(wkopt)));
    Tensor work;
    work.Resize(framework::make_ddim({lwork}));
    T* work_data = work.mutable_data<T>(context.GetPlace());
    VLOG(0) << "work_data : " << work_data;
    VLOG(0) << "residuals : " << residuals;

    T* solution_data = solution->mutable_data<T>(context.GetPlace());

    // "rwork" only used for complex inputs and "gelsy/gelsd/gelss" drivers
    Tensor rwork;
    ValueType* rwork_data = nullptr;
    if (framework::IsComplexType(x->type()) &&
        driver != LapackDriverType::Gels) {
      int64_t rwork_len = 0;
      if (driver == LapackDriverType::Gelsy) {
        rwork_len = std::max<int64_t>(1, 2 * n);
      } else if (driver == LapackDriverType::Gelss) {
        rwork_len = std::max<int64_t>(1, 5 * std::min(m, n));
      } else if (driver == LapackDriverType::Gelsd) {
        rwork_len = std::max<int64_t>(1, rwkopt);
      }
      rwork.Resize(framework::make_ddim({rwork_len}));
      rwork_data = rwork.mutable_data<ValueType>(context.GetPlace());
    }

    // "iwork" workspace array is relavant only for "gelsd" driver
    Tensor iwork;
    int* iwork_data = nullptr;
    if (driver == LapackDriverType::Gelsd) {
      iwork.Resize(framework::make_ddim({std::max<int>(1, iwkopt)}));
      iwork_data = iwork.mutable_data<int>(context.GetPlace());
    }

    int batch_count = BatchCount(*x);
    int solution_stride = MatrixStride(*solution);

    VLOG(0) << "solution_stride : " << solution_stride;

    for (auto i = 0; i < batch_count; ++i) {
      auto* x_input = &x_vector[i * x_stride];
      auto* y_input = &y_vector[i * y_stride];
      auto* current_solution = &solution_data[i * solution_stride];
      rank_working_ptr = rank_working_ptr ? &rank_data[i] : nullptr;
      s_working_ptr = s_working_ptr ? &s_data[i * s_stride] : nullptr;

      if (driver == LapackDriverType::Gels) {
        math::lapackGels('N', m, n, nrhs, x_input, lda, y_input, ldb,
                         current_solution, lwork, &info);
      } else if (driver == LapackDriverType::Gelsd) {
        math::lapackGelsd(m, n, nrhs, x_input, lda, y_input, ldb, s_working_ptr,
                          static_cast<ValueType>(rcond), &rank_32,
                          current_solution, lwork, rwork_data, iwork_data,
                          &info);
      } else if (driver == LapackDriverType::Gelsy) {
        math::lapackGelsy(m, n, nrhs, x_input, lda, y_input, ldb, jpvt_data,
                          static_cast<ValueType>(rcond), &rank_32,
                          current_solution, lwork, rwork_data, &info);
      } else if (driver == LapackDriverType::Gelss) {
        math::lapackGelss(m, n, nrhs, x_input, lda, y_input, ldb, s_working_ptr,
                          static_cast<ValueType>(rcond), &rank_32,
                          current_solution, lwork, rwork_data, &info);
      }

      PADDLE_ENFORCE_EQ(
          info, 0,
          platform::errors::PreconditionNotMet(
              "current info is not 0, computation failed. "
              "= 0:  successful exit."
              "< 0:  if INFO = -i, the i-th argument had an illegal value."
              "> 0:  if INFO = i, the QR algorithm failed to compute all the "
              "eigenvalues, and no eigenvectors have been computed; "
              "elements i+1:N of WR and WI contain eigenvalues which "
              "have converged."));

      if (rank_working_ptr) {
        *rank_working_ptr = static_cast<int64_t>(rank_32);
      }
    }
  }
};

template <typename T>
void BatchedOrmqr(const platform::CUDADeviceContext& dev_ctx, bool left,
                  bool transpose, int batch_size, int m, int n, int k, T* a,
                  int a_stride, T* tau, int tau_stride, T* other,
                  int other_stride);

}  // namespace operators
}  // namespace paddle
