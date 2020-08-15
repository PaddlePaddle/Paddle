/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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

#include <cublas.h>
#include <algorithm>
#include <functional>
#include <vector>
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/blas.h"

namespace paddle {
namespace operators {

using framework::Tensor;
void ComputeBroadcastBinaryOpDims(const int A_ndim, const std::int64_t* A_dims,
                                  const int B_ndim, const std::int64_t* B_dims,
                                  std::int64_t* A_broadcast_dims,
                                  std::int64_t* B_broadcast_dims,
                                  std::int64_t* C_broadcast_dims) {
  const int ndim = std::max(A_ndim, B_ndim);
  std::fill(A_broadcast_dims, A_broadcast_dims + ndim - A_ndim, 1);
  std::fill(B_broadcast_dims, B_broadcast_dims + ndim - B_ndim, 1);
  std::copy(A_dims, A_dims + A_ndim, A_broadcast_dims + ndim - A_ndim);
  std::copy(B_dims, B_dims + B_ndim, B_broadcast_dims + ndim - B_ndim);
  for (int i = 0; i < ndim; ++i) {
    PADDLE_ENFORCE_EQ(A_broadcast_dims[i] == B_broadcast_dims[i] ||
                          A_broadcast_dims[i] <= 1 || B_broadcast_dims[i] <= 1,
                      true, platform::errors::InvalidArgument(
                                "Input(X) and Input(Y) has error dim."));
    if (A_broadcast_dims[i] == 0 || B_broadcast_dims[i] == 0) {
      C_broadcast_dims[i] = 0;
    } else {
      C_broadcast_dims[i] = std::max(A_broadcast_dims[i], B_broadcast_dims[i]);
    }
  }
}

int64_t GetIndexFromDims(const int n, const int64_t* dims,
                         const int64_t* index) {
  int64_t sum = 0;
  for (int i = 0; i < n; ++i) {
    if (dims[i] > 1) {
      sum = sum * dims[i] + index[i];
    }
  }
  return sum;
}

void IncreaseIndexInDims(const int ndim, const int64_t* dims, int64_t* index) {
  for (int i = ndim - 1; i >= 0; --i) {
    ++index[i];
    if (index[i] >= dims[i]) {
      index[i] -= dims[i];
    } else {
      break;
    }
  }
}

template <typename DeviceContext, typename T>
void MatMulFunction(const Tensor* X, const Tensor* Y, Tensor* Out, bool trans_x,
                    bool trans_y,
                    const paddle::framework::ExecutionContext& ctx) {
  // get dims
  const std::vector<std::int64_t> x_dims = vectorize(X->dims());
  const std::vector<std::int64_t> y_dims = vectorize(Y->dims());
  const int x_ndim = x_dims.size();
  const int y_ndim = y_dims.size();

  // get data ptr
  const T* x_data = X->data<T>();
  const T* y_data = Y->data<T>();

  if (x_ndim == 1 && y_ndim == 1) {
    VLOG(0) << "MatMul's case 1";
    Out->Resize({1});
    Out->mutable_data<T>(ctx.GetPlace());
    auto out_eigen = framework::EigenScalar<T>::From(*Out);
    auto x_eigen = framework::EigenVector<T>::Flatten(*X);
    auto y_eigen = framework::EigenVector<T>::Flatten(*Y);

    auto& dev = *ctx.template device_context<DeviceContext>().eigen_device();
    out_eigen.device(dev) = (x_eigen * y_eigen).sum();
    return;
  }

  auto& dev_ctx = ctx.template device_context<DeviceContext>();
  auto blas = math::GetBlas<DeviceContext, T>(dev_ctx);

  if (x_ndim == 1) {
    const int N = X->numel();
    if (trans_y) {
      PADDLE_ENFORCE_EQ(
          y_dims[y_ndim - 1], N,
          platform::errors::InvalidArgument("Input(Y) has error dim."));
    } else {
      PADDLE_ENFORCE_EQ(
          y_dims[y_ndim - 2], N,
          platform::errors::InvalidArgument("Input(Y) has error dim."));
    }
    std::vector<std::int64_t> out_dims(y_ndim - 1);
    if (trans_y) {
      std::copy_n(y_dims.cbegin(), y_ndim - 1, out_dims.begin());
    } else {
      std::copy_n(y_dims.cbegin(), y_ndim - 2, out_dims.begin());
      out_dims.back() = y_dims.back();
    }
    Out->Resize(framework::make_ddim(out_dims));
    Out->mutable_data<T>(ctx.GetPlace());
    if (trans_y) {
      VLOG(0) << "haha";
    }
    if (trans_y) {
      const int M = Y->numel() / N;
      VLOG(0) << "MatMul's case 2";
      blas.GEMV(false, M, N, 1., y_data, x_data, 0., Out->data<T>());
    } else {
      const int M = y_dims[y_ndim - 1];
      const int batch_size = Y->numel() / (M * N);
      if (batch_size == 1) {
        VLOG(0) << "MatMul's case 3";
        blas.GEMV(true, N, M, 1., y_data, x_data, 0., Out->data<T>());
      } else {
        VLOG(0) << "MatMul's case 4";
        blas.BatchedGEMM(CblasTrans, CblasNoTrans, M, 1, N, 1.0f, y_data,
                         x_data, 0, Out->data<T>(), batch_size, M * N, 0);
      }
    }
    return;
  }

  if (y_ndim == 1) {
    const int N = Y->numel();
    if (trans_x) {
      PADDLE_ENFORCE_EQ(
          x_dims[x_ndim - 2], N,
          platform::errors::InvalidArgument("Input(X) has error dim."));
    } else {
      PADDLE_ENFORCE_EQ(
          x_dims[x_ndim - 1], N,
          platform::errors::InvalidArgument("Input(X) has error dim."));
    }
    std::vector<std::int64_t> out_dims(x_ndim - 1);
    if (trans_x) {
      std::copy_n(x_dims.cbegin(), x_ndim - 2, out_dims.begin());
      out_dims.back() = x_dims.back();
    } else {
      std::copy_n(x_dims.cbegin(), x_ndim - 1, out_dims.begin());
    }
    Out->Resize(framework::make_ddim(out_dims));
    Out->mutable_data<T>(ctx.GetPlace());

    if (trans_x) {
      const int M = x_dims[x_ndim - 1];
      const int batch_size = X->numel() / (M * N);
      if (batch_size == 1) {
        VLOG(0) << "MatMul's case 5";
        blas.GEMV(true, N, M, 1.0f, x_data, y_data, 0.0f, Out->data<T>());
      } else {
        VLOG(0) << "MatMul's case 6";
        blas.BatchedGEMM(CblasTrans, CblasNoTrans, M, 1, N, 1.0f, x_data,
                         y_data, 0, Out->data<T>(), batch_size, M * N, 0);
      }
    } else {
      const int M = X->numel() / N;
      VLOG(0) << "MatMul's case 7";
      blas.GEMV(false, M, N, 1.0f, x_data, y_data, 0.0f, Out->data<T>());
    }
    return;
  }

  const int M = trans_x ? x_dims[x_ndim - 1] : x_dims[x_ndim - 2];
  const int K = trans_x ? x_dims[x_ndim - 2] : x_dims[x_ndim - 1];
  if (trans_y) {
    PADDLE_ENFORCE_EQ(y_dims[y_ndim - 1], K, platform::errors::InvalidArgument(
                                                 "Input(X) has error dim."));
  } else {
    PADDLE_ENFORCE_EQ(y_dims[y_ndim - 2], K, platform::errors::InvalidArgument(
                                                 "Input(X) has error dim."));
  }
  const int N = trans_y ? y_dims[y_ndim - 2] : y_dims[y_ndim - 1];
  const int ndim = std::max(x_ndim, y_ndim);
  std::vector<std::int64_t> x_broadcast_dims(ndim);
  std::vector<std::int64_t> y_broadcast_dims(ndim);
  std::vector<std::int64_t> out_broadcast_dims(ndim);

  ComputeBroadcastBinaryOpDims(x_ndim - 2, x_dims.data(), y_ndim - 2,
                               y_dims.data(), x_broadcast_dims.data(),
                               y_broadcast_dims.data(),
                               out_broadcast_dims.data());

  out_broadcast_dims[ndim - 2] = M;
  out_broadcast_dims[ndim - 1] = N;

  Out->Resize(framework::make_ddim(out_broadcast_dims));
  Out->mutable_data<T>(ctx.GetPlace());

  const int batch_dim = ndim - 2;
  // broadcast message
  const bool is_broadcast_dims = !std::equal(
      x_broadcast_dims.cbegin(), x_broadcast_dims.cbegin() + batch_dim,
      y_broadcast_dims.cbegin());

  const std::int64_t x_batch_size = std::accumulate(
      x_broadcast_dims.cbegin(), x_broadcast_dims.cbegin() + batch_dim, 1LL,
      std::multiplies<std::int64_t>());
  const std::int64_t y_batch_size = std::accumulate(
      y_broadcast_dims.cbegin(), y_broadcast_dims.cbegin() + batch_dim, 1LL,
      std::multiplies<std::int64_t>());
  const std::int64_t out_batch_size = std::accumulate(
      out_broadcast_dims.cbegin(), out_broadcast_dims.cbegin() + batch_dim, 1LL,
      std::multiplies<std::int64_t>());
  if (out_batch_size == 0) {
    return;
  }

  if (x_batch_size == 1 && y_batch_size == 1) {
    VLOG(0) << "MatMul's case 8";
    blas.GEMM(trans_x ? CblasTrans : CblasNoTrans,
              trans_y ? CblasTrans : CblasNoTrans, M, N, K, 1.0f, x_data,
              y_data, 0.0f, Out->data<T>());
  } else if (x_batch_size == 1) {
    if (M == 1 && trans_y) {
      VLOG(0) << "MatMul's case 9";
      blas.GEMV(false, y_batch_size * N, K, 1.0f, y_data, x_data, 0.0f,
                Out->data<T>());
    } else {
      VLOG(0) << "MatMul's case 10";
      blas.BatchedGEMM(trans_x ? CblasTrans : CblasNoTrans,
                       trans_y ? CblasTrans : CblasNoTrans, M, N, K, 1.0f,
                       x_data, y_data, 0, Out->data<T>(), out_batch_size, 0,
                       K * N);
    }
  } else if (y_batch_size == 1) {
    if (!trans_x) {
      VLOG(0) << "MatMul's case 11";
      blas.GEMM(CblasNoTrans, trans_y ? CblasTrans : CblasNoTrans,
                x_batch_size * M, N, K, 1.0f, x_data, y_data, 0.0f,
                Out->data<T>());
    } else {
      VLOG(0) << "MatMul's case 12";
      blas.BatchedGEMM(CblasTrans, trans_y ? CblasTrans : CblasNoTrans, M, N, K,
                       1.0f, x_data, y_data, 0, Out->data<T>(), out_batch_size,
                       M * K, 0);
    }
  } else if (!is_broadcast_dims) {
    VLOG(0) << "MatMul's case 13";
    blas.BatchedGEMM(trans_x ? CblasTrans : CblasNoTrans,
                     trans_y ? CblasTrans : CblasNoTrans, M, N, K, 1.0f, x_data,
                     y_data, 0, Out->data<T>(), out_batch_size, M * K, K * N);
  } else {
    std::vector<const T*> x_ptr(out_batch_size);
    std::vector<const T*> y_ptr(out_batch_size);
    std::vector<T*> out_ptr(out_batch_size);
    std::vector<std::int64_t> index(batch_dim);
    for (std::int64_t i = 0; i < out_batch_size; ++i) {
      const std::int64_t x_index =
          GetIndexFromDims(batch_dim, x_broadcast_dims.data(), index.data());
      const std::int64_t y_index =
          GetIndexFromDims(batch_dim, y_broadcast_dims.data(), index.data());

      x_ptr[i] = x_data + x_index * M * K;
      y_ptr[i] = y_data + y_index * K * N;
      out_ptr[i] = Out->data<T>() + i * M * N;
      IncreaseIndexInDims(batch_dim, out_broadcast_dims.data(), index.data());
    }
    VLOG(0) << "MatMul's case 14";
    blas.BatchedGEMM(trans_x ? CblasTrans : CblasNoTrans,
                     trans_y ? CblasTrans : CblasNoTrans, M, N, K, 1.0f,
                     x_ptr.data(), y_ptr.data(), 0.0f, out_ptr.data(),
                     out_batch_size);
  }
}

template <typename DeviceContext, typename T>
class MatMulV2Kernel : public framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    auto* X = ctx.Input<Tensor>("X");
    auto* Y = ctx.Input<Tensor>("Y");
    auto* Out = ctx.Output<Tensor>("Out");
    bool trans_x = ctx.Attr<bool>("trans_x");
    bool trans_y = ctx.Attr<bool>("trans_y");

    MatMulFunction<DeviceContext, T>(X, Y, Out, trans_x, trans_y, ctx);
  }
};

}  // namespace operators
}  // namespace paddle
