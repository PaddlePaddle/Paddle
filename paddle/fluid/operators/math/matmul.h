/* Copyright (c) 2017 PaddlePaddle Authors. All Rights Reserved.

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
#include <algorithm>
#include <vector>
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {
namespace math {

// Implements the logic of numpy matmul:
// https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html
//
// but allowing also for a, b to be transposed
//
// Both a & b can be 1- to 3-dimensional. Higher rank tensors are not supported
// yet.
template <typename DeviceContext, typename T>
class MatMulFunctor {
 public:
  void operator()(const DeviceContext& context, const framework::Tensor& a,
                  bool trans_a, const framework::Tensor& b, bool trans_b,
                  T alpha, framework::Tensor* out, T beta) {
    auto dim_a = a.dims();
    auto dim_b = b.dims();

    PADDLE_ENFORCE(a.place() == b.place() && b.place() == out->place(),
                   "Tensors must all be in the same place.");
    PADDLE_ENFORCE_GE(dim_a.size(), 1,
                      "Input tensor a must be at least 1-dimensional.");
    PADDLE_ENFORCE_GE(dim_b.size(), 1,
                      "Input tensor b must be at least 1-dimensional.");

    std::vector<int64_t> out_dim;
    int64_t batch_count = 1;
    if (dim_a.size() > 3) {
      PADDLE_ENFORCE(dim_b.size() == dim_a.size(),
                     "The dimensions of X and Y must be the same, and both of "
                     "them should be %d-dimensional.",
                     dim_b.size());
      // The first rank-2 dimensions are accumulated on the batch_count, and the
      // last two dimensions are used for matrix multiplication.
      for (int j = 0; j < dim_a.size() - 2; ++j) {
        PADDLE_ENFORCE_EQ(dim_b[j], dim_a[j],
                          "The %d-th dimension of X and Y must be the same.",
                          j);
        out_dim.push_back(dim_a[j]);
        batch_count *= dim_a[j];
      }
    }

    int M = 0, N = 0, kA = 0, kB = 0, batchCountA = 0, batchCountB = 0,
        strideA = 0, strideB = 0;

    switch (dim_a.size()) {
      case 1:
        // similar to np.matmul:
        // prepend dimension 1 (no transpose) or append dimension 1 (transpose)
        M = trans_a ? dim_a[0] : 1;
        kA = trans_a ? 1 : dim_a[0];
        break;
      case 2:
        M = trans_a ? dim_a[1] : dim_a[0];
        kA = trans_a ? dim_a[0] : dim_a[1];
        break;
      case 3:
        batchCountA = dim_a[0];
        M = trans_a ? dim_a[2] : dim_a[1];
        kA = trans_a ? dim_a[1] : dim_a[2];
        strideA = M * kA;
        break;
      default:
        batchCountA = batch_count;
        size_t mat_s = dim_a.size() - 2;
        M = trans_a ? dim_a[mat_s + 1] : dim_a[mat_s];
        kA = trans_a ? dim_a[mat_s] : dim_a[mat_s + 1];
        strideA = M * kA;
    }

    switch (dim_b.size()) {
      case 1:
        // similar to np.matmul:
        // append dimension 1 (no transpose) or prepend dimension 1 (transpose)
        kB = trans_b ? 1 : dim_b[0];
        N = trans_b ? dim_b[0] : 1;
        break;
      case 2:
        kB = trans_b ? dim_b[1] : dim_b[0];
        N = trans_b ? dim_b[0] : dim_b[1];
        break;
      case 3:
        batchCountB = dim_b[0];
        kB = trans_b ? dim_b[2] : dim_b[1];
        N = trans_b ? dim_b[1] : dim_b[2];
        strideB = kB * N;
        break;
      default:
        batchCountB = batch_count;
        size_t mat_s = dim_b.size() - 2;
        kB = trans_b ? dim_b[mat_s + 1] : dim_b[mat_s];
        N = trans_b ? dim_b[mat_s] : dim_b[mat_s + 1];
        strideB = kB * N;
    }

    PADDLE_ENFORCE_EQ(
        kA, kB,
        "First matrix's width must be equal with second matrix's height.");
    if (batchCountA && batchCountB) {
      PADDLE_ENFORCE_EQ(
          batchCountA, batchCountB,
          "When input tensors a and b are both batched, they must have the "
          "same batch dimension.");
    }
    int batchCount = std::max(batchCountA, batchCountB);

    CBLAS_TRANSPOSE transA = (trans_a == false) ? CblasNoTrans : CblasTrans;
    CBLAS_TRANSPOSE transB = (trans_b == false) ? CblasNoTrans : CblasTrans;

    if (!batchCount) {
      // regular matrix multiplication
      gemm<DeviceContext, T>(context, transA, transB, M, N, kA, alpha,
                             a.data<T>(), b.data<T>(), beta, out->data<T>());
    } else {
      // batched matrix multiplication
      batched_gemm<DeviceContext, T>(
          context, transA, transB, M, N, kA, alpha, a.data<T>(), b.data<T>(),
          beta, out->data<T>(), batchCount, strideA, strideB);
    }
  }
};

}  // namespace math
}  // namespace operators
}  // namespace paddle
