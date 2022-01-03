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

#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/complex_functors.h"

#include "paddle/pten/core/dense_tensor.h"

namespace pten {

static void GetBroadcastFromDims(const int x_ndim,
                                 const std::int64_t* x_dims,
                                 const int y_ndim,
                                 const std::int64_t* y_dims,
                                 std::int64_t* x_bd_dims,
                                 std::int64_t* y_bd_dims,
                                 std::int64_t* out_bd_dims) {
  const int ndim = (std::max)(x_ndim, y_ndim);
  std::fill(x_bd_dims, x_bd_dims + ndim - x_ndim, 1);
  std::fill(y_bd_dims, y_bd_dims + ndim - y_ndim, 1);
  std::copy(x_dims, x_dims + x_ndim, x_bd_dims + ndim - x_ndim);
  std::copy(y_dims, y_dims + y_ndim, y_bd_dims + ndim - y_ndim);

  for (int i = 0; i < ndim; ++i) {
    PADDLE_ENFORCE_EQ(
        x_bd_dims[i] == y_bd_dims[i] || x_bd_dims[i] <= 1 || y_bd_dims[i] <= 1,
        true,
        paddle::platform::errors::InvalidArgument(
            "Input(X) and Input(Y) has error dim."
            "X_broadcast's shape[%s] must be equal to Y_broadcast's shape[%s],"
            "or X_broadcast's shape[%s] <= 1, or Y_broadcast's shape[%s] <= 1,"
            "But received X_broadcast's shape[%s] = [%s]"
            "received Y_broadcast's shape[%s] = [%s]",
            i,
            i,
            i,
            i,
            i,
            x_bd_dims[i],
            i,
            y_bd_dims[i]));
    if (x_bd_dims[i] == 0 || y_bd_dims[i] == 0) {
      out_bd_dims[i] = 0;
    } else {
      out_bd_dims[i] = (std::max)(x_bd_dims[i], y_bd_dims[i]);
    }
  }
}

static int64_t GetIndexMessage(const int n,
                               const int64_t* dims,
                               const int64_t* index) {
  int64_t sum = 0;
  for (int i = 0; i < n; ++i) {
    if (dims[i] > 1) {
      sum = sum * dims[i] + index[i];
    }
  }
  return sum;
}

static void IndexIncreaseFromDims(const int ndim,
                                  const int64_t* dims,
                                  int64_t* index) {
  for (int i = ndim - 1; i >= 0; --i) {
    ++index[i];
    if (index[i] >= dims[i]) {
      index[i] -= dims[i];
    } else {
      break;
    }
  }
}

template <typename Context, typename T>
void MatMulFunction(const Context& context,
                    const DenseTensor& X,
                    const DenseTensor& Y,
                    const std::vector<std::int64_t>& x_dims,
                    const std::vector<std::int64_t>& y_dims,
                    DenseTensor* Out,
                    bool trans_x,
                    bool trans_y,
                    bool flag = false) {
  const int x_ndim = x_dims.size();
  const int y_ndim = y_dims.size();

  // Get data ptr
  const T* x_data = X.data<T>();
  const T* y_data = Y.data<T>();

  auto blas = paddle::operators::math::GetBlas<Context, T>(context);

  if (x_ndim == 1 && y_ndim == 1) {
    const int M = X.numel();
    const int N = Y.numel();
    PADDLE_ENFORCE_EQ(
        M,
        N,
        paddle::platform::errors::InvalidArgument(
            "X's numbers must be equal to Y's numbers,"
            "when X/Y's dims =1. But received X has [%d] elements,"
            "received Y has [%d] elements",
            M,
            N));
    VLOG(3) << "MatMul's case 1";
    blas.GEMM(CblasNoTrans,
              CblasTrans,
              1,
              1,
              M,
              static_cast<T>(1),
              y_data,
              x_data,
              static_cast<T>(flag),
              Out->mutable_data<T>());
    return;
  }

  if (x_ndim == 1) {
    const int N = X.numel();
    if (trans_y) {
      PADDLE_ENFORCE_EQ(y_dims[y_ndim - 1],
                        N,
                        paddle::platform::errors::InvalidArgument(
                            "Input(Y) has error dim."
                            "Y'dims[%d] must be equal to %d"
                            "But received Y'dims[%d] is %d",
                            y_ndim - 1,
                            N,
                            y_ndim - 1,
                            y_dims[y_ndim - 1]));
    } else {
      PADDLE_ENFORCE_EQ(y_dims[y_ndim - 2],
                        N,
                        paddle::platform::errors::InvalidArgument(
                            "Input(Y) has error dim."
                            "Y'dims[%d] must be equal to %d"
                            "But received Y'dims[%d] is %d",
                            y_ndim - 2,
                            N,
                            y_ndim - 2,
                            y_dims[y_ndim - 2]));
    }
    std::vector<std::int64_t> out_dims(y_ndim - 1);
    if (trans_y) {
      std::copy_n(y_dims.cbegin(), y_ndim - 1, out_dims.begin());
    } else {
      std::copy_n(y_dims.cbegin(), y_ndim - 2, out_dims.begin());
      out_dims.back() = y_dims.back();
    }
    Out->Resize(paddle::framework::make_ddim(out_dims));
    Out->mutable_data<T>();
    if (trans_y) {
      const int M = Y.numel() / N;
      VLOG(3) << "MatMul's case 2";
      blas.GEMV(false,
                M,
                N,
                static_cast<T>(1),
                y_data,
                x_data,
                static_cast<T>(flag),
                Out->mutable_data<T>());
    } else {
      const int M = y_dims[y_ndim - 1];
      const int batch_size = Y.numel() / (M * N);
      if (batch_size == 1) {
        VLOG(3) << "MatMul's case 3";
        blas.GEMV(true,
                  N,
                  M,
                  static_cast<T>(1),
                  y_data,
                  x_data,
                  static_cast<T>(flag),
                  Out->mutable_data<T>());
      } else {
        VLOG(3) << "MatMul's case 4";
        blas.BatchedGEMM(CblasTrans,
                         CblasNoTrans,
                         M,
                         1,
                         N,
                         static_cast<T>(1),
                         y_data,
                         x_data,
                         static_cast<T>(flag),
                         Out->mutable_data<T>(),
                         batch_size,
                         M * N,
                         0);
      }
    }
    return;
  }

  if (y_ndim == 1) {
    const int N = Y.numel();
    if (trans_x) {
      PADDLE_ENFORCE_EQ(x_dims[x_ndim - 2],
                        N,
                        paddle::platform::errors::InvalidArgument(
                            "Input(X) has error dim."
                            "X'dims[%d] must be equal to %d"
                            "But received X'dims[%d] is %d",
                            x_ndim - 2,
                            N,
                            x_ndim - 2,
                            x_dims[x_ndim - 2]));
    } else {
      PADDLE_ENFORCE_EQ(x_dims[x_ndim - 1],
                        N,
                        paddle::platform::errors::InvalidArgument(
                            "Input(X) has error dim."
                            "X'dims[%d] must be equal to %d"
                            "But received X'dims[%d] is %d",
                            x_ndim - 1,
                            N,
                            x_ndim - 1,
                            x_dims[x_ndim - 1]));
    }
    std::vector<std::int64_t> out_dims(x_ndim - 1);
    if (trans_x) {
      std::copy_n(x_dims.cbegin(), x_ndim - 2, out_dims.begin());
      out_dims.back() = x_dims.back();
    } else {
      std::copy_n(x_dims.cbegin(), x_ndim - 1, out_dims.begin());
    }
    Out->Resize(paddle::framework::make_ddim(out_dims));
    Out->mutable_data<T>();

    if (trans_x) {
      const int M = x_dims[x_ndim - 1];
      const int batch_size = X.numel() / (M * N);
      if (batch_size == 1) {
        VLOG(3) << "MatMul's case 5";
        blas.GEMV(true,
                  N,
                  M,
                  static_cast<T>(1),
                  x_data,
                  y_data,
                  static_cast<T>(flag),
                  Out->mutable_data<T>());
      } else {
        VLOG(3) << "MatMul's case 6";
        blas.BatchedGEMM(CblasTrans,
                         CblasNoTrans,
                         M,
                         1,
                         N,
                         static_cast<T>(1),
                         x_data,
                         y_data,
                         static_cast<T>(flag),
                         Out->mutable_data<T>(),
                         batch_size,
                         M * N,
                         0);
      }
    } else {
      const int M = X.numel() / N;
      VLOG(3) << "MatMul's case 7";
      blas.GEMV(false,
                M,
                N,
                static_cast<T>(1),
                x_data,
                y_data,
                static_cast<T>(flag),
                Out->mutable_data<T>());
    }
    return;
  }

  const int M = trans_x ? x_dims[x_ndim - 1] : x_dims[x_ndim - 2];
  const int K = trans_x ? x_dims[x_ndim - 2] : x_dims[x_ndim - 1];
  if (trans_y) {
    PADDLE_ENFORCE_EQ(y_dims[y_ndim - 1],
                      K,
                      paddle::platform::errors::InvalidArgument(
                          "Input(Y) has error dim."
                          "Y'dims[%d] must be equal to %d"
                          "But received Y'dims[%d] is %d",
                          y_ndim - 1,
                          K,
                          y_ndim - 1,
                          y_dims[y_ndim - 1]));
  } else {
    PADDLE_ENFORCE_EQ(y_dims[y_ndim - 2],
                      K,
                      paddle::platform::errors::InvalidArgument(
                          "Input(Y) has error dim."
                          "Y'dims[%d] must be equal to %d"
                          "But received Y'dims[%d] is %d",
                          y_ndim - 2,
                          K,
                          y_ndim - 2,
                          y_dims[y_ndim - 2]));
  }
  const int N = trans_y ? y_dims[y_ndim - 2] : y_dims[y_ndim - 1];
  const int ndim = (std::max)(x_ndim, y_ndim);
  std::vector<std::int64_t> x_broadcast_dims(ndim);
  std::vector<std::int64_t> y_broadcast_dims(ndim);
  std::vector<std::int64_t> out_broadcast_dims(ndim);

  GetBroadcastFromDims(x_ndim - 2,
                       x_dims.data(),
                       y_ndim - 2,
                       y_dims.data(),
                       x_broadcast_dims.data(),
                       y_broadcast_dims.data(),
                       out_broadcast_dims.data());
  out_broadcast_dims[ndim - 2] = M;
  out_broadcast_dims[ndim - 1] = N;

  Out->Resize(paddle::framework::make_ddim(out_broadcast_dims));
  Out->mutable_data<T>();

  const int batch_dim = ndim - 2;
  // broadcast message
  const bool is_broadcast_dims =
      !std::equal(x_broadcast_dims.cbegin(),
                  x_broadcast_dims.cbegin() + batch_dim,
                  y_broadcast_dims.cbegin());

  const std::int64_t x_batch_size =
      std::accumulate(x_broadcast_dims.cbegin(),
                      x_broadcast_dims.cbegin() + batch_dim,
                      1LL,
                      std::multiplies<std::int64_t>());
  const std::int64_t y_batch_size =
      std::accumulate(y_broadcast_dims.cbegin(),
                      y_broadcast_dims.cbegin() + batch_dim,
                      1LL,
                      std::multiplies<std::int64_t>());
  const std::int64_t out_batch_size =
      std::accumulate(out_broadcast_dims.cbegin(),
                      out_broadcast_dims.cbegin() + batch_dim,
                      1LL,
                      std::multiplies<std::int64_t>());
  if (out_batch_size == 0) return;
  if (x_batch_size == 1 && y_batch_size == 1) {
    VLOG(3) << "MatMul's case 8";
    blas.GEMM(trans_x ? CblasTrans : CblasNoTrans,
              trans_y ? CblasTrans : CblasNoTrans,
              M,
              N,
              K,
              static_cast<T>(1),
              x_data,
              y_data,
              static_cast<T>(flag),
              Out->mutable_data<T>());
  } else if (x_batch_size == 1) {
    if (M == 1 && trans_y) {
      VLOG(3) << "MatMul's case 9";
      blas.GEMV(false,
                y_batch_size * N,
                K,
                static_cast<T>(1),
                y_data,
                x_data,
                static_cast<T>(flag),
                Out->mutable_data<T>());
    } else {
      VLOG(3) << "MatMul's case 10";
      blas.BatchedGEMM(trans_x ? CblasTrans : CblasNoTrans,
                       trans_y ? CblasTrans : CblasNoTrans,
                       M,
                       N,
                       K,
                       static_cast<T>(1),
                       x_data,
                       y_data,
                       static_cast<T>(flag),
                       Out->mutable_data<T>(),
                       out_batch_size,
                       0,
                       K * N);
    }
  } else if (y_batch_size == 1) {
    if (!trans_x) {
      VLOG(3) << "MatMul's case 11";
      blas.GEMM(CblasNoTrans,
                trans_y ? CblasTrans : CblasNoTrans,
                x_batch_size * M,
                N,
                K,
                static_cast<T>(1),
                x_data,
                y_data,
                static_cast<T>(flag),
                Out->mutable_data<T>());
    } else {
      VLOG(3) << "MatMul's case 12";
      blas.BatchedGEMM(CblasTrans,
                       trans_y ? CblasTrans : CblasNoTrans,
                       M,
                       N,
                       K,
                       static_cast<T>(1),
                       x_data,
                       y_data,
                       static_cast<T>(flag),
                       Out->mutable_data<T>(),
                       out_batch_size,
                       M * K,
                       0);
    }
  } else if (!is_broadcast_dims) {
    VLOG(3) << "MatMul's case 13";
    blas.BatchedGEMM(trans_x ? CblasTrans : CblasNoTrans,
                     trans_y ? CblasTrans : CblasNoTrans,
                     M,
                     N,
                     K,
                     static_cast<T>(1),
                     x_data,
                     y_data,
                     static_cast<T>(flag),
                     Out->mutable_data<T>(),
                     out_batch_size,
                     M * K,
                     K * N);
  } else {
    // in the case, can't use stridedgemm
    std::vector<const T*> x_ptr(out_batch_size);
    std::vector<const T*> y_ptr(out_batch_size);
    std::vector<T*> out_ptr(out_batch_size);
    std::vector<std::int64_t> index(batch_dim, 0);
    for (std::int64_t i = 0; i < out_batch_size; ++i) {
      // using the index to get offset
      const std::int64_t x_index =
          GetIndexMessage(batch_dim, x_broadcast_dims.data(), index.data());
      const std::int64_t y_index =
          GetIndexMessage(batch_dim, y_broadcast_dims.data(), index.data());

      x_ptr[i] = x_data + x_index * M * K;
      y_ptr[i] = y_data + y_index * K * N;
      out_ptr[i] = Out->mutable_data<T>() + i * M * N;
      IndexIncreaseFromDims(batch_dim, out_broadcast_dims.data(), index.data());
    }
    VLOG(3) << "MatMul's case 14";
    blas.BatchedGEMM(trans_x ? CblasTrans : CblasNoTrans,
                     trans_y ? CblasTrans : CblasNoTrans,
                     M,
                     N,
                     K,
                     static_cast<T>(1),
                     x_ptr.data(),
                     y_ptr.data(),
                     static_cast<T>(flag),
                     out_ptr.data(),
                     out_batch_size);
  }
}

template <typename Context, typename T>
void MatMulFunction(const Context& context,
                    const DenseTensor& X,
                    const DenseTensor& Y,
                    DenseTensor* Out,
                    bool trans_x,
                    bool trans_y,
                    bool flag = false) {
  const std::vector<std::int64_t> x_dims = vectorize(X.dims());
  const std::vector<std::int64_t> y_dims = vectorize(Y.dims());
  MatMulFunction<Context, T>(
      context, X, Y, x_dims, y_dims, Out, trans_x, trans_y, flag);
}

template <typename T, typename Context>
void MatmulKernel(const Context& context,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  bool transpose_x,
                  bool transpose_y,
                  DenseTensor* out) {
  PADDLE_ENFORCE_NE(paddle::framework::product(x.dims()),
                    0,
                    paddle::platform::errors::InvalidArgument(
                        "The Input(X) dims size must not be equal 0,"
                        " but reviced dims size is 0. "));
  PADDLE_ENFORCE_NE(paddle::framework::product(y.dims()),
                    0,
                    paddle::platform::errors::InvalidArgument(
                        "The Input(Y) dims size must not be equal 0,"
                        " but reviced dims size is 0. "));
  MatMulFunction<Context, T>(context, x, y, out, transpose_x, transpose_y);
}

}  // namespace pten
