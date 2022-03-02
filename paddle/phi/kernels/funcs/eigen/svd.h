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

#pragma once

#include <Eigen/SVD>

#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"

namespace phi {
namespace funcs {

template <typename T>
void EigenSvd(
    const T* X, T* U, T* VH, T* S, int rows, int cols, int full = false) {
  auto flag = Eigen::DecompositionOptions::ComputeThinU |
              Eigen::DecompositionOptions::ComputeThinV;
  if (full) {
    flag = Eigen::DecompositionOptions::ComputeFullU |
           Eigen::DecompositionOptions::ComputeFullV;
  }
  Eigen::BDCSVD<
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      svd(2, 2, flag);
  /*NOTE(xiongkun03) Eigen::Matrix API need non-const pointer.*/
  T* input = const_cast<T*>(X);
  auto m = Eigen::Map<
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
      input, rows, cols);
  svd.compute(m);
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> V_trans =
      svd.matrixV().transpose();
  memcpy(U, svd.matrixU().data(), svd.matrixU().size() * sizeof(T));
  memcpy(VH, V_trans.data(), V_trans.size() * sizeof(T));
  memcpy(
      S, svd.singularValues().data(), svd.singularValues().size() * sizeof(T));
}

template <typename T>
void BatchSvd(const T* X,
              T* U,
              T* VH,
              T* S,
              int rows,
              int cols,
              int batches,
              int full = false) {
  int stride = rows * cols;
  int k = std::min(rows, cols);
  int stride_u = full ? rows * rows : k * rows;
  int stride_v = full ? cols * cols : k * cols;
  for (int i = 0; i < batches; ++i) {
    EigenSvd<T>(X + i * stride,
                U + i * stride_u,
                VH + i * stride_v,
                S + i * k,
                rows,
                cols,
                full);
  }
  return;
}

}  // namespace funcs
}  // namespace phi
