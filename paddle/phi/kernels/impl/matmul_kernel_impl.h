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

#include "glog/logging.h"

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/autotune/cache_base.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/blas/blaslt_impl.cu.h"
#include "paddle/phi/kernels/funcs/complex_functors.h"
#if defined(PADDLE_WITH_CUDA) && CUDA_VERSION >= 11060
#include "paddle/phi/kernels/autotune/auto_tune_base.h"
#endif

namespace phi {

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
        phi::errors::InvalidArgument(
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

// The general implementation with blas.
template <typename Context, typename T>
void MatMulFunctionImplWithBlas(
    const Context& dev_ctx,
    const DenseTensor& X,
    const DenseTensor& Y,
    const std::vector<std::int64_t>& x_dims,
    const std::vector<std::int64_t>& y_dims,
    DenseTensor* Out,
    bool trans_x,
    bool trans_y,
    bool flag = false,
    phi::funcs::MatmulPlanner* matmul_planner UNUSED = nullptr) {
  const int x_ndim = x_dims.size();
  const int y_ndim = y_dims.size();

  // Get data ptr
  const T* x_data = X.data<T>();
  const T* y_data = Y.data<T>();

  auto blas = phi::funcs::GetBlas<Context, T>(dev_ctx);

  if (x_ndim == 1 && y_ndim == 1) {
    const int M = X.numel();
    const int N = Y.numel();
    PADDLE_ENFORCE_EQ(
        M,
        N,
        phi::errors::InvalidArgument(
            "X's numbers must be equal to Y's numbers,"
            "when X/Y's dims =1. But received X has [%d] elements,"
            "received Y has [%d] elements",
            M,
            N));
    VLOG(3) << "MatMul's case 1";
    Out->Resize(phi::make_ddim({}));
    dev_ctx.template Alloc<T>(Out);
    blas.GEMM(CblasNoTrans,
              CblasTrans,
              1,
              1,
              M,
              static_cast<T>(1),
              y_data,
              x_data,
              static_cast<T>(flag),
              dev_ctx.template Alloc<T>(Out));
    return;
  }

  if (x_ndim == 1) {
    const int N = X.numel();
    if (trans_y) {
      PADDLE_ENFORCE_EQ(
          y_dims[y_ndim - 1],
          N,
          phi::errors::InvalidArgument("Input(Y) has error dim."
                                       "Y'dims[%d] must be equal to %d"
                                       "But received Y'dims[%d] is %d",
                                       y_ndim - 1,
                                       N,
                                       y_ndim - 1,
                                       y_dims[y_ndim - 1]));
    } else {
      PADDLE_ENFORCE_EQ(
          y_dims[y_ndim - 2],
          N,
          phi::errors::InvalidArgument("Input(Y) has error dim."
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
    Out->ResizeAndAllocate(phi::make_ddim(out_dims));
    dev_ctx.template Alloc<T>(Out);
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
                dev_ctx.template Alloc<T>(Out));
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
                  dev_ctx.template Alloc<T>(Out));
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
                         dev_ctx.template Alloc<T>(Out),
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
      PADDLE_ENFORCE_EQ(
          x_dims[x_ndim - 2],
          N,
          phi::errors::InvalidArgument("Input(X) has error dim."
                                       "X'dims[%d] must be equal to %d"
                                       "But received X'dims[%d] is %d",
                                       x_ndim - 2,
                                       N,
                                       x_ndim - 2,
                                       x_dims[x_ndim - 2]));
    } else {
      PADDLE_ENFORCE_EQ(
          x_dims[x_ndim - 1],
          N,
          phi::errors::InvalidArgument("Input(X) has error dim."
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
    Out->ResizeAndAllocate(phi::make_ddim(out_dims));
    dev_ctx.template Alloc<T>(Out);

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
                  dev_ctx.template Alloc<T>(Out));
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
                         dev_ctx.template Alloc<T>(Out),
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
                dev_ctx.template Alloc<T>(Out));
    }
    return;
  }

  const int M = trans_x ? x_dims[x_ndim - 1] : x_dims[x_ndim - 2];
  const int K = trans_x ? x_dims[x_ndim - 2] : x_dims[x_ndim - 1];
  if (trans_y) {
    PADDLE_ENFORCE_EQ(
        y_dims[y_ndim - 1],
        K,
        phi::errors::InvalidArgument("Input(Y) has error dim."
                                     "Y'dims[%d] must be equal to %d"
                                     "But received Y'dims[%d] is %d",
                                     y_ndim - 1,
                                     K,
                                     y_ndim - 1,
                                     y_dims[y_ndim - 1]));
  } else {
    PADDLE_ENFORCE_EQ(
        y_dims[y_ndim - 2],
        K,
        phi::errors::InvalidArgument("Input(Y) has error dim."
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

  Out->ResizeAndAllocate(phi::make_ddim(out_broadcast_dims));
  dev_ctx.template Alloc<T>(Out);

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
              dev_ctx.template Alloc<T>(Out));
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
                dev_ctx.template Alloc<T>(Out));
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
                       dev_ctx.template Alloc<T>(Out),
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
                dev_ctx.template Alloc<T>(Out));
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
                       dev_ctx.template Alloc<T>(Out),
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
                     dev_ctx.template Alloc<T>(Out),
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
      out_ptr[i] = dev_ctx.template Alloc<T>(Out) + i * M * N;
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

#if defined(PADDLE_WITH_CUDA) && CUDA_VERSION >= 11060
// This is almost a copy from MatMulFunctionImplWithBlas,
// compare cublas with cublasLt kernels when Matmul autotune is on
template <typename Context, typename T>
void MatMulFunctionImplWithCublasLt(
    const Context& dev_ctx,
    const DenseTensor& X,
    const DenseTensor& Y,
    const std::vector<std::int64_t>& x_dims,
    const std::vector<std::int64_t>& y_dims,
    DenseTensor* Out,
    bool trans_x,
    bool trans_y,
    bool flag = false,
    phi::funcs::MatmulPlanner* matmul_planner = nullptr) {
  const int x_ndim = x_dims.size();
  const int y_ndim = y_dims.size();
  const T* x_data = X.data<T>();
  const T* y_data = Y.data<T>();
  using blaslt = phi::funcs::MatmulWithCublasLt<T>;

  if (x_ndim == 1 && y_ndim == 1) {
    const int M = X.numel();
    const int N = Y.numel();
    PADDLE_ENFORCE_EQ(
        M,
        N,
        phi::errors::InvalidArgument(
            "X's numbers must be equal to Y's numbers,"
            "when X/Y's dims =1. But received X has [%d] elements,"
            "received Y has [%d] elements",
            M,
            N));

    // MatMul's case 0  =>  vector * vector
    Out->Resize(phi::make_ddim({}));
    dev_ctx.template Alloc<T>(Out);
    VLOG(3) << "MatMul with blaslt case 1";
    blaslt::Run(dev_ctx,
                y_data,
                x_data,
                dev_ctx.template Alloc<T>(Out),
                1,
                1,
                M,
                false,
                true,
                matmul_planner);
    return;
  }

  if (x_ndim == 1) {
    const int N = X.numel();
    if (trans_y) {
      PADDLE_ENFORCE_EQ(
          y_dims[y_ndim - 1],
          N,
          phi::errors::InvalidArgument("Input(Y) has error dim."
                                       "Y'dims[%d] must be equal to %d"
                                       "But received Y'dims[%d] is %d",
                                       y_ndim - 1,
                                       N,
                                       y_ndim - 1,
                                       y_dims[y_ndim - 1]));
    } else {
      PADDLE_ENFORCE_EQ(
          y_dims[y_ndim - 2],
          N,
          phi::errors::InvalidArgument("Input(Y) has error dim."
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
    Out->ResizeAndAllocate(phi::make_ddim(out_dims));
    dev_ctx.template Alloc<T>(Out);
    if (trans_y) {
      const int M = Y.numel() / N;
      VLOG(3) << "MatMul with blaslt 2";
      blaslt::Run(dev_ctx,
                  y_data,
                  x_data,
                  dev_ctx.template Alloc<T>(Out),
                  M,
                  1,
                  N,
                  false,
                  false,
                  matmul_planner);
    } else {
      const int M = y_dims[y_ndim - 1];
      const int batch_size = Y.numel() / (M * N);
      if (batch_size == 1) {
        VLOG(3) << "MatMul with blaslt 3";
        blaslt::Run(dev_ctx,
                    y_data,
                    x_data,
                    dev_ctx.template Alloc<T>(Out),
                    M,
                    1,
                    N,
                    true,
                    false,
                    matmul_planner);
      } else {
        VLOG(3) << "MatMul with blaslt 4";
        blaslt::RunWithBatch(dev_ctx,
                             y_data,
                             x_data,
                             dev_ctx.template Alloc<T>(Out),
                             M,
                             1,
                             N,
                             true,
                             false,
                             batch_size,
                             M * N,
                             0,
                             M,
                             matmul_planner);
      }
    }
    return;
  }

  if (y_ndim == 1) {
    const int N = Y.numel();
    if (trans_x) {
      PADDLE_ENFORCE_EQ(
          x_dims[x_ndim - 2],
          N,
          phi::errors::InvalidArgument("Input(X) has error dim."
                                       "X'dims[%d] must be equal to %d"
                                       "But received X'dims[%d] is %d",
                                       x_ndim - 2,
                                       N,
                                       x_ndim - 2,
                                       x_dims[x_ndim - 2]));
    } else {
      PADDLE_ENFORCE_EQ(
          x_dims[x_ndim - 1],
          N,
          phi::errors::InvalidArgument("Input(X) has error dim."
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
    Out->ResizeAndAllocate(phi::make_ddim(out_dims));
    dev_ctx.template Alloc<T>(Out);

    if (trans_x) {
      const int M = x_dims[x_ndim - 1];
      const int batch_size = X.numel() / (M * N);
      if (batch_size == 1) {
        VLOG(3) << "MatMul with blaslt 5";
        blaslt::Run(dev_ctx,
                    x_data,
                    y_data,
                    dev_ctx.template Alloc<T>(Out),
                    M,
                    1,
                    N,
                    true,
                    false,
                    matmul_planner);
      } else {
        VLOG(3) << "MatMul with blaslt 6";
        blaslt::RunWithBatch(dev_ctx,
                             x_data,
                             y_data,
                             dev_ctx.template Alloc<T>(Out),
                             M,
                             1,
                             N,
                             true,
                             false,
                             batch_size,
                             M * N,
                             0,
                             M,
                             matmul_planner);
      }
    } else {
      const int M = X.numel() / N;
      VLOG(3) << "MatMul with blaslt 7";
      blaslt::Run(dev_ctx,
                  x_data,
                  y_data,
                  dev_ctx.template Alloc<T>(Out),
                  M,
                  1,
                  N,
                  false,
                  false,
                  matmul_planner);
    }
    return;
  }

  const int M = trans_x ? x_dims[x_ndim - 1] : x_dims[x_ndim - 2];
  const int K = trans_x ? x_dims[x_ndim - 2] : x_dims[x_ndim - 1];
  if (trans_y) {
    PADDLE_ENFORCE_EQ(
        y_dims[y_ndim - 1],
        K,
        phi::errors::InvalidArgument("Input(Y) has error dim."
                                     "Y'dims[%d] must be equal to %d"
                                     "But received Y'dims[%d] is %d",
                                     y_ndim - 1,
                                     K,
                                     y_ndim - 1,
                                     y_dims[y_ndim - 1]));
  } else {
    PADDLE_ENFORCE_EQ(
        y_dims[y_ndim - 2],
        K,
        phi::errors::InvalidArgument("Input(Y) has error dim."
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

  Out->ResizeAndAllocate(phi::make_ddim(out_broadcast_dims));
  dev_ctx.template Alloc<T>(Out);

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
    VLOG(3) << "MatMul with blaslt 8";
    blaslt::Run(dev_ctx,
                x_data,
                y_data,
                dev_ctx.template Alloc<T>(Out),
                M,
                N,
                K,
                trans_x,
                trans_y,
                matmul_planner);
  } else if (x_batch_size == 1) {
    if (M == 1 && trans_y) {
      VLOG(3) << "MatMul with blaslt 9";
      blaslt::Run(dev_ctx,
                  y_data,
                  x_data,
                  dev_ctx.template Alloc<T>(Out),
                  y_batch_size * N,
                  1,
                  K,
                  false,
                  false,
                  matmul_planner);
    } else {
      VLOG(3) << "MatMul with blaslt 10";
      blaslt::RunWithBatch(dev_ctx,
                           x_data,
                           y_data,
                           dev_ctx.template Alloc<T>(Out),
                           M,
                           N,
                           K,
                           trans_x,
                           trans_y,
                           out_batch_size,
                           0,
                           K * N,
                           M * N,
                           matmul_planner);
    }
  } else if (y_batch_size == 1) {
    if (!trans_x) {
      VLOG(3) << "MatMul with blaslt 11";
      blaslt::Run(dev_ctx,
                  x_data,
                  y_data,
                  dev_ctx.template Alloc<T>(Out),
                  x_batch_size * M,
                  N,
                  K,
                  false,
                  trans_y,
                  matmul_planner);
    } else {
      VLOG(3) << "MatMul with blaslt 12";
      blaslt::RunWithBatch(dev_ctx,
                           x_data,
                           y_data,
                           dev_ctx.template Alloc<T>(Out),
                           M,
                           N,
                           K,
                           true,
                           trans_y,
                           out_batch_size,
                           M * K,
                           0,
                           M * N,
                           matmul_planner);
    }
  } else if (!is_broadcast_dims) {
    VLOG(3) << "MatMul with blaslt 13";
    blaslt::RunWithBatch(dev_ctx,
                         x_data,
                         y_data,
                         dev_ctx.template Alloc<T>(Out),
                         M,
                         N,
                         K,
                         trans_x,
                         trans_y,
                         out_batch_size,
                         M * K,
                         K * N,
                         M * N,
                         matmul_planner);
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
      out_ptr[i] = dev_ctx.template Alloc<T>(Out) + i * M * N;
      IndexIncreaseFromDims(batch_dim, out_broadcast_dims.data(), index.data());
    }
    VLOG(3) << "MatMul with blaslt 14";
    blaslt::RunWithBatch(dev_ctx,
                         x_ptr.data(),
                         y_ptr.data(),
                         out_ptr.data(),
                         M,
                         N,
                         K,
                         trans_x,
                         trans_y,
                         out_batch_size,
                         matmul_planner);
  }
}
#endif

template <typename Context, typename T>
struct MatMulDispatcher {
  void operator()(const Context& ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  const std::vector<std::int64_t>& x_dims,
                  const std::vector<std::int64_t>& y_dims,
                  DenseTensor* out,
                  bool trans_x,
                  bool trans_y,
                  bool flag = false) {
    MatMulFunctionImplWithBlas<Context, T>(
        ctx, x, y, x_dims, y_dims, out, trans_x, trans_y, flag);
  }
};

#ifdef PADDLE_WITH_CUDA
template <typename T>
struct MatMulDispatcher<phi::GPUContext, T> {
  void operator()(const phi::GPUContext& ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  const std::vector<std::int64_t>& x_dims,
                  const std::vector<std::int64_t>& y_dims,
                  DenseTensor* out,
                  bool trans_x,
                  bool trans_y,
                  bool flag = false) {
#if CUDA_VERSION >= 11060
    auto* tuner = phi::autotune::MakeMatmulTuner<T>(
        MatMulFunctionImplWithBlas<phi::GPUContext, T>);
    tuner->AddCallBack(MatMulFunctionImplWithCublasLt<phi::GPUContext, T>);
    phi::funcs::MatmulPlanner matmul_planner(x_dims,
                                             y_dims,
                                             trans_x,
                                             trans_y,
                                             phi::CppTypeToDataType<T>::Type(),
                                             funcs::MatmulFusedType::kMatmul,
                                             /* bias_data */ nullptr,
                                             /* reserve_data */ nullptr,
                                             /* use_addto */ flag,
                                             /* no_exchange */ true);
    tuner->Run(ctx,
               matmul_planner.GetKey(),
               ctx,
               x,
               y,
               x_dims,
               y_dims,
               out,
               trans_x,
               trans_y,
               flag,
               &matmul_planner);
#else
    MatMulFunctionImplWithBlas<phi::GPUContext, T>(
        ctx, x, y, x_dims, y_dims, out, trans_x, trans_y, flag);
#endif
  }
};
#endif  // PADDLE_WITH_CUDA

template <typename Context, typename T>
void MatMulFunction(const Context& ctx,
                    const DenseTensor& x,
                    const DenseTensor& y,
                    const std::vector<std::int64_t>& x_dims,
                    const std::vector<std::int64_t>& y_dims,
                    DenseTensor* out,
                    bool trans_x,
                    bool trans_y,
                    bool flag = false) {
  MatMulDispatcher<Context, T>()(
      ctx, x, y, x_dims, y_dims, out, trans_x, trans_y, flag);
}

template <typename T, typename Context>
void MatmulKernel(const Context& ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  bool transpose_x,
                  bool transpose_y,
                  DenseTensor* out) {
  PADDLE_ENFORCE_NE(
      phi::product(x.dims()),
      0,
      phi::errors::InvalidArgument("The Input(X) dims size must not be equal 0,"
                                   " but reviced dims size is 0. "));
  PADDLE_ENFORCE_NE(
      phi::product(y.dims()),
      0,
      phi::errors::InvalidArgument("The Input(Y) dims size must not be equal 0,"
                                   " but reviced dims size is 0. "));
  const std::vector<std::int64_t> x_dims = vectorize(x.dims());
  const std::vector<std::int64_t> y_dims = vectorize(y.dims());
  MatMulFunction<Context, T>(
      ctx, x, y, x_dims, y_dims, out, transpose_x, transpose_y);
}

template <typename T, typename Context>
void MatmulWithFlattenKernel(const Context& dev_ctx,
                             const DenseTensor& x,
                             const DenseTensor& y,
                             int x_num_col_dims,
                             int y_num_col_dims,
                             DenseTensor* out) {
  const DenseTensor x_matrix =
      x.dims().size() > 2 ? phi::ReshapeToMatrix(x, x_num_col_dims) : x;
  const DenseTensor y_matrix =
      y.dims().size() > 2 ? phi::ReshapeToMatrix(y, y_num_col_dims) : y;

  dev_ctx.template Alloc<T>(out);
  auto z_dim = out->dims();
  if (z_dim.size() != 2) {
    out->Resize({x_matrix.dims()[0], y_matrix.dims()[1]});
  }

  auto blas = phi::funcs::GetBlas<Context, T>(dev_ctx);

  blas.MatMul(x_matrix, y_matrix, out);
  if (z_dim.size() != 2) {
    out->Resize(z_dim);
  }
}

}  // namespace phi
