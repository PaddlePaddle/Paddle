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
#include "paddle/fluid/operators/math/eigen_values_vectors.h"
#include "paddle/fluid/operators/math/matrix_solve.h"
#include "paddle/fluid/operators/svd_helper.h"
#include "paddle/fluid/operators/transpose_op.h"
#include "paddle/fluid/platform/for_range.h"
#include "paddle/phi/kernels/funcs/complex_functors.h"
#include "paddle/phi/kernels/funcs/lapack/lapack_function.h"
#include "paddle/phi/kernels/funcs/math_function.h"

#define EPSILON 1e-6

namespace paddle {
namespace operators {

using paddle::framework::Tensor;
enum class LapackDriverType : int { Gels, Gelsd, Gelsy, Gelss };

using DDim = framework::DDim;
static DDim UDDim(const DDim& x_dim) {
  auto x_vec = vectorize(x_dim);
  return phi::make_ddim(x_vec);
}

template <typename DeviceContext, typename T>
class LstsqCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    using ValueType = phi::dtype::Real<T>;

    const Tensor& x = *context.Input<Tensor>("X");
    auto y = context.Input<Tensor>("Y");
    auto rcond = context.Attr<float>("rcond");
    auto driver_string = context.Attr<std::string>("driver");

    static auto driver_type = std::unordered_map<std::string, LapackDriverType>(
        {{"gels", LapackDriverType::Gels},
         {"gelsy", LapackDriverType::Gelsy},
         {"gelsd", LapackDriverType::Gelsd},
         {"gelss", LapackDriverType::Gelss}});
    auto driver = driver_type[driver_string];

    auto solution = context.Output<Tensor>("Solution");
    auto* rank = context.Output<Tensor>("Rank");
    auto* singular_values = context.Output<Tensor>("SingularValues");

    auto dito =
        math::DeviceIndependenceTensorOperations<DeviceContext, T>(context);

    auto x_dims = x.dims();
    auto y_dims = y->dims();
    int dim_size = x_dims.size();
    int x_stride = MatrixStride(x);
    int y_stride = MatrixStride(*y);
    int batch_count = BatchCount(x);
    auto solution_dim = solution->dims();
    int ori_solu_stride = MatrixStride(*solution);
    int max_solu_stride = std::max(y_stride, ori_solu_stride);
    int min_solu_stride = std::min(y_stride, ori_solu_stride);

    // lapack is a column-major storge, transpose make the input to
    // have a continuous memory layout
    int info = 0;
    int m = x_dims[dim_size - 2];
    int n = x_dims[dim_size - 1];
    int nrhs = y_dims[dim_size - 1];
    int lda = std::max<int>(m, 1);
    int ldb = std::max<int>(1, std::max(m, n));

    Tensor new_x;
    new_x.mutable_data<T>(context.GetPlace(),
                          size_t(batch_count * m * n * sizeof(T)));
    framework::TensorCopy(x, context.GetPlace(), &new_x);

    solution->mutable_data<T>(
        context.GetPlace(),
        size_t(batch_count * std::max(m, n) * nrhs * sizeof(T)));

    if (m >= n) {
      const Tensor& new_y = *context.Input<Tensor>("Y");
      framework::TensorCopy(new_y, context.GetPlace(), solution);
    } else {
      auto* solu_data = solution->data<T>();
      auto* y_data = y->data<T>();
      for (auto i = 0; i < batch_count; i++) {
        for (auto j = 0; j < min_solu_stride; j++) {
          solu_data[i * max_solu_stride + j] = y_data[i * y_stride + j];
        }
      }
    }

    Tensor input_x_trans = dito.Transpose(new_x);
    Tensor input_y_trans = dito.Transpose(*solution);
    framework::TensorCopy(input_x_trans, new_x.place(), &new_x);
    framework::TensorCopy(input_y_trans, solution->place(), solution);

    auto* x_vector = new_x.data<T>();
    auto* y_vector = solution->data<T>();

    // "gels" divers does not need to compute rank
    int rank_32 = 0;
    int* rank_data = nullptr;
    int* rank_working_ptr = nullptr;
    if (driver != LapackDriverType::Gels) {
      rank_data = rank->mutable_data<int>(context.GetPlace());
      rank_working_ptr = rank_data;
    }

    // "gelsd" and "gelss" divers need to compute singular values
    ValueType* s_data = nullptr;
    ValueType* s_working_ptr = nullptr;
    int s_stride = 0;
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
      jpvt.Resize(phi::make_ddim({std::max<int>(1, n)}));
      jpvt_data = jpvt.mutable_data<int>(context.GetPlace());
    }

    // run once the driver, first to get the optimal workspace size
    int lwork = -1;
    T wkopt;
    ValueType rwkopt;
    int iwkopt = 0;

    if (driver == LapackDriverType::Gels) {
      phi::funcs::lapackGels('N', m, n, nrhs, x_vector, lda, y_vector, ldb,
                             &wkopt, lwork, &info);
    } else if (driver == LapackDriverType::Gelsd) {
      phi::funcs::lapackGelsd(m, n, nrhs, x_vector, lda, y_vector, ldb,
                              s_working_ptr, static_cast<ValueType>(rcond),
                              &rank_32, &wkopt, lwork, &rwkopt, &iwkopt, &info);
    } else if (driver == LapackDriverType::Gelsy) {
      phi::funcs::lapackGelsy(m, n, nrhs, x_vector, lda, y_vector, ldb,
                              jpvt_data, static_cast<ValueType>(rcond),
                              &rank_32, &wkopt, lwork, &rwkopt, &info);
    } else if (driver == LapackDriverType::Gelss) {
      phi::funcs::lapackGelss(m, n, nrhs, x_vector, lda, y_vector, ldb,
                              s_working_ptr, static_cast<ValueType>(rcond),
                              &rank_32, &wkopt, lwork, &rwkopt, &info);
    }

    lwork = std::max<int>(1, static_cast<int>(phi::dtype::Real<T>(wkopt)));
    Tensor work;
    work.Resize(phi::make_ddim({lwork}));
    T* work_data = work.mutable_data<T>(context.GetPlace());

    // "rwork" only used for complex inputs and "gelsy/gelsd/gelss" drivers
    Tensor rwork;
    ValueType* rwork_data = nullptr;
    if (framework::IsComplexType(framework::TransToProtoVarType(x.dtype())) &&
        driver != LapackDriverType::Gels) {
      int rwork_len = 0;
      if (driver == LapackDriverType::Gelsy) {
        rwork_len = std::max<int>(1, 2 * n);
      } else if (driver == LapackDriverType::Gelss) {
        rwork_len = std::max<int>(1, 5 * std::min(m, n));
      } else if (driver == LapackDriverType::Gelsd) {
        rwork_len = std::max<int>(1, rwkopt);
      }
      rwork.Resize(phi::make_ddim({rwork_len}));
      rwork_data = rwork.mutable_data<ValueType>(context.GetPlace());
    }

    // "iwork" workspace array is relavant only for "gelsd" driver
    Tensor iwork;
    int* iwork_data = nullptr;
    if (driver == LapackDriverType::Gelsd) {
      iwork.Resize(phi::make_ddim({std::max<int>(1, iwkopt)}));
      iwork_data = iwork.mutable_data<int>(context.GetPlace());
    }

    for (auto i = 0; i < batch_count; ++i) {
      auto* x_input = &x_vector[i * x_stride];
      auto* y_input = &y_vector[i * max_solu_stride];
      rank_working_ptr = rank_working_ptr ? &rank_data[i] : nullptr;
      s_working_ptr = s_working_ptr ? &s_data[i * s_stride] : nullptr;

      if (driver == LapackDriverType::Gels) {
        phi::funcs::lapackGels('N', m, n, nrhs, x_input, lda, y_input, ldb,
                               work_data, lwork, &info);
      } else if (driver == LapackDriverType::Gelsd) {
        phi::funcs::lapackGelsd(m, n, nrhs, x_input, lda, y_input, ldb,
                                s_working_ptr, static_cast<ValueType>(rcond),
                                &rank_32, work_data, lwork, rwork_data,
                                iwork_data, &info);
      } else if (driver == LapackDriverType::Gelsy) {
        phi::funcs::lapackGelsy(m, n, nrhs, x_input, lda, y_input, ldb,
                                jpvt_data, static_cast<ValueType>(rcond),
                                &rank_32, work_data, lwork, rwork_data, &info);
      } else if (driver == LapackDriverType::Gelss) {
        phi::funcs::lapackGelss(m, n, nrhs, x_input, lda, y_input, ldb,
                                s_working_ptr, static_cast<ValueType>(rcond),
                                &rank_32, work_data, lwork, rwork_data, &info);
      }

      PADDLE_ENFORCE_EQ(
          info, 0,
          platform::errors::PreconditionNotMet(
              "For batch [%d]: Lapack info is not zero but [%d]", i, info));

      if (rank_working_ptr) *rank_working_ptr = static_cast<int>(rank_32);
    }

    Tensor tmp_s = dito.Transpose(*solution);
    framework::TensorCopy(tmp_s, solution->place(), solution);

    if (m > n) {
      auto* solu_data = solution->data<T>();
      for (auto i = 1; i < batch_count; i++) {
        for (auto j = 0; j < min_solu_stride; j++) {
          solu_data[i * min_solu_stride + j] =
              solu_data[i * max_solu_stride + j];
        }
      }
    }

    solution->Resize(UDDim(solution_dim));
  }
};

template <typename DeviceContext, typename T>
void BatchedOrmqr(const DeviceContext& dev_ctx, bool left, bool transpose,
                  int batch_size, int m, int n, int k, T* a, int a_stride,
                  T* tau, int tau_stride, T* other, int other_stride);

}  // namespace operators
}  // namespace paddle
