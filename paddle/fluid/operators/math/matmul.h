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
  struct MatDim {
    int height_;
    int width_;
    int batch_size_{0};
    int stride_{0};
  };

  static MatDim GetMatDim(bool trans, const framework::DDim& dim) {
    MatDim res;
    auto dim_vec = framework::vectorize(dim);
    PADDLE_ENFORCE(!dim_vec.empty(),
                   "Input tensor must be at least 1-dimensional");
    if (dim_vec.size() == 1) {
      res.height_ = trans ? dim_vec[0] : 1;
      res.width_ = trans ? 1 : dim_vec[0];
    } else if (dim_vec.size() == 2) {
      res.height_ = trans ? dim_vec[1] : dim_vec[0];
      res.width_ = trans ? dim_vec[0] : dim_vec[1];
    } else {
      res.height_ =
          trans ? dim_vec[dim_vec.size() - 1] : dim_vec[dim_vec.size() - 2];
      res.width_ =
          trans ? dim_vec[dim_vec.size() - 2] : dim_vec[dim_vec.size() - 1];
      res.batch_size_ =
          std::accumulate(dim_vec.begin(), dim_vec.end() - 2, 1,
                          [](int64_t elem, int a) -> int { return elem * a; });
      res.stride_ = res.height_ * res.width_;
    }
    return res;
  }

  static CBLAS_TRANSPOSE GetTranspose(bool trans) {
    return trans ? CblasTrans : CblasNoTrans;
  }

  void operator()(const DeviceContext& context, const framework::Tensor& a,
                  bool trans_a, const framework::Tensor& b, bool trans_b,
                  T alpha, framework::Tensor* out, T beta) {
    PADDLE_ENFORCE(a.place() == b.place() && b.place() == out->place(),
                   "Tensors must all be in the same place.");
    auto dim_a = GetMatDim(trans_a, a.dims());
    auto dim_b = GetMatDim(trans_b, b.dims());

    PADDLE_ENFORCE_EQ(dim_a.width_, dim_b.height_);

    if (dim_a.batch_size_ != 0 && dim_b.batch_size_ != 0) {
      PADDLE_ENFORCE_EQ(dim_a.batch_size_, dim_b.batch_size_);
    }

    if (dim_a.batch_size_ == 0 && dim_b.batch_size_ == 0) {
      PADDLE_ENFORCE_EQ(out->numel(), dim_a.height_ * dim_b.width_);
      gemm<DeviceContext, T>(context, GetTranspose(trans_a),
                             GetTranspose(trans_b), dim_a.height_, dim_b.width_,
                             dim_a.width_, alpha, a.data<T>(), b.data<T>(),
                             beta, out->data<T>());
    } else {
      int batch_size =
          dim_a.batch_size_ == 0 ? dim_b.batch_size_ : dim_a.batch_size_;
      PADDLE_ENFORCE_EQ(out->numel(),
                        batch_size * dim_a.height_ * dim_b.width_);
      batched_gemm<DeviceContext, T>(
          context, GetTranspose(trans_a), GetTranspose(trans_b), dim_a.height_,
          dim_b.width_, dim_a.width_, alpha, a.data<T>(), b.data<T>(), beta,
          out->data<T>(), batch_size, dim_a.stride_, dim_b.stride_);
    }
  }
};

}  // namespace math
}  // namespace operators
}  // namespace paddle
