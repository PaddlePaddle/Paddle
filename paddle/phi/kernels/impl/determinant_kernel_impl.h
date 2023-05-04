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

#include "glog/logging.h"
#include "paddle/phi/common/amp_type_traits.h"

#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/determinant_kernel.h"

namespace phi {
namespace detail {
template <typename T>
class EigenMatrix {};

template <>
class EigenMatrix<phi::dtype::float16> {
 public:
  using MatrixType =
      Eigen::Matrix<phi::dtype::float16, Eigen::Dynamic, Eigen::Dynamic>;
};

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
    using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
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
      output_vec.push_back(
          static_cast<T>(matrix.template cast<MPType>().determinant()));
    }
    phi::TensorFromVector(output_vec, dev_ctx, output);
  }
};

template <typename T, typename Context>
void DeterminantKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       DenseTensor* out) {
  auto input_dim = vectorize(x.dims());
  auto input_dim_size = input_dim.size();

  auto batch_count = detail::GetBatchCount(x.dims());
  VLOG(10) << "input dim:" << x.dims();
  PADDLE_ENFORCE_GE(
      input_dim_size,
      2,
      phi::errors::InvalidArgument(
          "the input matrix dimension size should greater than 2."));
  PADDLE_ENFORCE_EQ(input_dim[input_dim_size - 1],
                    input_dim[input_dim_size - 2],
                    phi::errors::InvalidArgument(
                        "the input matrix should be square matrix."));
  auto rank = input_dim[input_dim_size - 1];  // square matrix length
  DeterminantFunctor<T, Context>()(dev_ctx, x, rank, batch_count, out);
  auto output_dims = phi::slice_ddim(x.dims(), 0, input_dim_size - 2);
  if (input_dim_size > 2) {
    out->Resize(output_dims);
  } else {
    // when input is a two-dimension matrix, The det value is a number.
    out->Resize(phi::make_ddim({}));
  }
  VLOG(10) << "output dim:" << out->dims();
}

}  // namespace phi
