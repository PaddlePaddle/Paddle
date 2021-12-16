/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/fluid/platform/dynload/cusparse.h"
#include "paddle/pten/core/dense_tensor.h"

namespace pten {

template <typename T>
inline void sparsennz(const cusparseHandle_t& handle,
                      const cusparseMatDescr_t& descr,
                      const int rows,
                      const int cols,
                      const T* data,
                      int* nnz,
                      int* nnzPerRowColumn);

template <>
inline void sparsennz<float>(const cusparseHandle_t& handle,
                             const cusparseMatDescr_t& descr,
                             const int rows,
                             const int cols,
                             const float* data,
                             int* nnz,
                             int* nnzPerRowColumn) {
  paddle::platform::dynload::cusparseSnnz(handle,
                                          CUSPARSE_DIRECTION_ROW,
                                          rows,
                                          cols,
                                          descr,
                                          data,
                                          rows,
                                          nnzPerRowColumn,
                                          nnz);
}

template <>
inline void sparsennz<double>(const cusparseHandle_t& handle,
                              const cusparseMatDescr_t& descr,
                              const int rows,
                              const int cols,
                              const double* data,
                              int* nnz,
                              int* nnzPerRowColumn) {
  paddle::platform::dynload::cusparseDnnz(handle,
                                          CUSPARSE_DIRECTION_ROW,
                                          rows,
                                          cols,
                                          descr,
                                          data,
                                          rows,
                                          nnzPerRowColumn,
                                          nnz);
}

template <typename T>
inline void get_non_zero_num(const paddle::platform::CUDADeviceContext& ctx,
                             const DenseTensor& dense,
                             const int64_t sparse_dim,
                             int* nnz,
                             int* nnzPerRowColumn) {
  const auto& dims = dense.dims();
  PADDLE_ENFORCE_GE(
      dims.size(),
      sparse_dim,
      paddle::platform::errors::InvalidArgument(
          "sparse_dim(%d) should be less than or equal to dense.dim(%d)",
          sparse_dim,
          dims.size()));

  auto dims_2d = flatten_to_2d(dims, sparse_dim);
  const int rows = dims_2d[0];
  const int cols = dims_2d[1];

  const T* data = dense.data<T>();
  cusparseHandle_t handle = ctx.cusparse_handle();
  cusparseMatDescr_t descr = 0;
  paddle::platform::dynload::cusparseCreateMatDescr(&descr);
  paddle::platform::dynload::cusparseSetMatType(descr,
                                                CUSPARSE_MATRIX_TYPE_GENERAL);
  paddle::platform::dynload::cusparseSetMatIndexBase(descr,
                                                     CUSPARSE_INDEX_BASE_ZERO);
  sparsennz<T>(handle, descr, rows, cols, data, nnz, nnzPerRowColumn);
}

}  // namespace pten
