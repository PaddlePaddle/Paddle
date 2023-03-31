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

#include <algorithm>
#include <cmath>
#include <vector>

#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/impl/determinant_kernel_impl.h"
#include "paddle/phi/kernels/slogdeterminant_kernel.h"

namespace phi {

template <typename T>
T sign(T val) {
  return static_cast<T>(T(0) < val) - (val < T(0));
}

template <typename T, typename Context>
struct SlogDeterminantFunctor {
  void operator()(const Context& dev_ctx,
                  const DenseTensor& input,
                  int64_t rank,
                  int64_t batch_count,
                  DenseTensor* output) {
    std::vector<T> input_vec;
    std::vector<T> sign_vec;
    std::vector<T> log_vec;
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
      VLOG(2) << "det value: " << matrix.determinant();
      VLOG(2) << "matrix val: " << matrix;
      auto det_val = matrix.determinant();
      sign_vec.push_back(sign(det_val));
      det_val >= 0
          ? log_vec.push_back(std::log(det_val))
          : log_vec.push_back(std::log(std::abs(
                det_val)));  // for computing log value of a negative value.
    }
    // merge sign_vec and log_vec as final output_vec
    output_vec.insert(output_vec.end(), sign_vec.begin(), sign_vec.end());
    output_vec.insert(output_vec.end(), log_vec.begin(), log_vec.end());
    phi::TensorFromVector(output_vec, dev_ctx, output);
  }
};

template <typename T, typename Context>
void SlogDeterminantKernel(const Context& dev_ctx,
                           const DenseTensor& x,
                           DenseTensor* out) {
  auto x_dim = x.dims();
  auto batch_count = detail::GetBatchCount(x.dims());
  auto rank = x_dim[x_dim.size() - 1];  // square matrix rank
  auto origin_out_dim = out->dims();
  SlogDeterminantFunctor<T, Context>()(dev_ctx, x, rank, batch_count, out);
  out->Resize(origin_out_dim);
}

}  // namespace phi
