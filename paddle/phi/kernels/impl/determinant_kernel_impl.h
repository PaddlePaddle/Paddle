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

#include <Eigen/Dense>
#include <Eigen/LU>
#include <algorithm>
#include <cmath>
#include <vector>

#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/determinant_kernel.h"

namespace phi {
namespace detail {
template <typename T>
class EigenMatrix {};

template <>
class EigenMatrix<float> {
 public:
  using MatrixType = Eigen::MatrixXf;
};

template <>
class EigenMatrix<double> {
 public:
  using MatrixType = Eigen::MatrixXd;
};

inline int64_t GetBatchCount(const DDim dims) {
  int64_t batch_count = 1;
  auto dim_size = dims.size();
  PADDLE_ENFORCE_GE(
      dim_size,
      2,
      phi::errors::InvalidArgument(
          "the input matrix dimension size should greater than 2."));

  // Cumulative multiplying each dimension until the last 2 to get the batch
  // count,
  // for example a tensor with shape [3,3,3,3], the batch count of matrices is
  // 9.
  for (int64_t i = 0; i < dims.size() - 2; i++) {
    batch_count *= dims[i];
  }

  return batch_count;
}
}  // namespace detail

template <typename T, typename Context>
struct DeterminantFunctor {
  void operator()(const Context& dev_ctx,
                  const DenseTensor& input,
                  int64_t rank,
                  int64_t batch_count,
                  DenseTensor* output) {
    std::vector<T> input_vec;
    std::vector<T> output_vec;
    phi::TensorToVector(input, dev_ctx, &input_vec);
    for (int64_t i = 0; i < batch_count; ++i) {  // maybe can be parallel
      auto begin_iter = input_vec.begin() + i * rank * rank;
      auto end_iter = input_vec.begin() + (i + 1) * rank * rank;
      std::vector<T> sub_vec(begin_iter,
                             end_iter);  // get every square matrix data
      typename detail::EigenMatrix<T>::MatrixType matrix(rank, rank);
      for (int64_t i = 0; i < rank; ++i) {
        for (int64_t j = 0; j < rank; ++j) {
          matrix(i, j) = sub_vec[rank * i + j];
        }
      }
      output_vec.push_back(matrix.determinant());
    }
    phi::TensorFromVector(output_vec, dev_ctx, output);
  }
};

template <typename T, typename Context>
void DeterminantKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       DenseTensor* out) {
  auto x_dim = x.dims();
  auto batch_count = detail::GetBatchCount(x.dims());
  auto rank = x_dim[x_dim.size() - 1];  // square matrix rank
  auto origin_out_dim = out->dims();
  DeterminantFunctor<T, Context>()(dev_ctx, x, rank, batch_count, out);
  out->Resize(origin_out_dim);
}

}  // namespace phi
