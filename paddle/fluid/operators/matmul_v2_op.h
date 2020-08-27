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

#include <algorithm>
#include <functional>
#include <vector>
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/dot_op.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/reduce_ops/reduce_sum_op.h"

#ifdef __NVCC__
#include "paddle/fluid/operators/reduce_ops/cub_reduce.h"
#endif

namespace paddle {
namespace operators {

using framework::Tensor;

template <typename T>
struct IdentityFunctor {
  HOSTDEVICE explicit inline IdentityFunctor() {}

  HOSTDEVICE inline T operator()(const T& x) const { return x; }
};

template <typename DeviceContext, typename T>
void ReduceSumForMatmulGrad(const Tensor* input, Tensor* output,
                            const std::vector<int>& reduce_dims,
                            const paddle::framework::ExecutionContext& ctx) {
  if (reduce_dims.empty()) {
    // FIXME maybe reduce this copy operation
    framework::TensorCopySync(*input, ctx.GetPlace(), output);
    return;
  }
#ifdef __NVCC__
  auto stream = ctx.cuda_device_context().stream();
  TensorReduce<T, T, cub::Sum, IdentityFunctor<T>>(
      *input, output, reduce_dims, static_cast<T>(0), cub::Sum(),
      IdentityFunctor<T>(), stream);
#else
  ReduceKernelFunctor<DeviceContext, T, ops::SumFunctor>(
      input, output, reduce_dims, true, false, ctx)
      .template apply<T>();
#endif
}

static void GetBroadcastFromDims(const int x_ndim, const std::int64_t* x_dims,
                                 const int y_ndim, const std::int64_t* y_dims,
                                 std::int64_t* x_bd_dims,
                                 std::int64_t* y_bd_dims,
                                 std::int64_t* out_bd_dims) {
  const int ndim = std::max(x_ndim, y_ndim);
  std::fill(x_bd_dims, x_bd_dims + ndim - x_ndim, 1);
  std::fill(y_bd_dims, y_bd_dims + ndim - y_ndim, 1);
  std::copy(x_dims, x_dims + x_ndim, x_bd_dims + ndim - x_ndim);
  std::copy(y_dims, y_dims + y_ndim, y_bd_dims + ndim - y_ndim);

  for (int i = 0; i < ndim; ++i) {
    PADDLE_ENFORCE_EQ(
        x_bd_dims[i] == y_bd_dims[i] || x_bd_dims[i] <= 1 || y_bd_dims[i] <= 1,
        true, platform::errors::InvalidArgument(
                  "Input(X) and Input(Y) has error dim."));
    if (x_bd_dims[i] == 0 || y_bd_dims[i] == 0) {
      out_bd_dims[i] = 0;
    } else {
      out_bd_dims[i] = std::max(x_bd_dims[i], y_bd_dims[i]);
    }
  }
}

static int64_t GetIndexMessage(const int n, const int64_t* dims,
                               const int64_t* index) {
  int64_t sum = 0;
  for (int i = 0; i < n; ++i) {
    if (dims[i] > 1) {
      sum = sum * dims[i] + index[i];
    }
  }
  return sum;
}

static void IndexIncreaseFromDims(const int ndim, const int64_t* dims,
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

template <typename DeviceContext, typename T>
void MatMulFunction(const Tensor* X, const Tensor* Y,
                    const std::vector<std::int64_t>& x_dims,
                    const std::vector<std::int64_t>& y_dims, Tensor* Out,
                    bool trans_x, bool trans_y,
                    const paddle::framework::ExecutionContext& ctx) {
  const int x_ndim = x_dims.size();
  const int y_ndim = y_dims.size();

  // get data ptr
  const T* x_data = X->data<T>();
  const T* y_data = Y->data<T>();

  if (x_ndim == 1 && y_ndim == 1) {
    PADDLE_ENFORCE_EQ(X->numel(), Y->numel(),
                      platform::errors::InvalidArgument(
                          "X's numbers is not equal to Y's numbers,"
                          "when X/Y's dims =1"));
    VLOG(3) << "MatMul's case 1";
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
      const int M = Y->numel() / N;
      VLOG(3) << "MatMul's case 2";
      blas.GEMV(false, M, N, 1., y_data, x_data, 0., Out->data<T>());
    } else {
      const int M = y_dims[y_ndim - 1];
      const int batch_size = Y->numel() / (M * N);
      if (batch_size == 1) {
        VLOG(3) << "MatMul's case 3";
        blas.GEMV(true, N, M, 1., y_data, x_data, 0., Out->data<T>());
      } else {
        VLOG(3) << "MatMul's case 4";
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
        VLOG(3) << "MatMul's case 5";
        blas.GEMV(true, N, M, 1.0f, x_data, y_data, 0.0f, Out->data<T>());
      } else {
        VLOG(3) << "MatMul's case 6";
        blas.BatchedGEMM(CblasTrans, CblasNoTrans, M, 1, N, 1.0f, x_data,
                         y_data, 0, Out->data<T>(), batch_size, M * N, 0);
      }
    } else {
      const int M = X->numel() / N;
      VLOG(3) << "MatMul's case 7";
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

  GetBroadcastFromDims(x_ndim - 2, x_dims.data(), y_ndim - 2, y_dims.data(),
                       x_broadcast_dims.data(), y_broadcast_dims.data(),
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
  if (out_batch_size == 0) return;
  if (x_batch_size == 1 && y_batch_size == 1) {
    VLOG(3) << "MatMul's case 8";
    blas.GEMM(trans_x ? CblasTrans : CblasNoTrans,
              trans_y ? CblasTrans : CblasNoTrans, M, N, K, 1.0f, x_data,
              y_data, 0.0f, Out->data<T>());
  } else if (x_batch_size == 1) {
    if (M == 1 && trans_y) {
      VLOG(3) << "MatMul's case 9";
      blas.GEMV(false, y_batch_size * N, K, 1.0f, y_data, x_data, 0.0f,
                Out->data<T>());
    } else {
      VLOG(3) << "MatMul's case 10";
      blas.BatchedGEMM(trans_x ? CblasTrans : CblasNoTrans,
                       trans_y ? CblasTrans : CblasNoTrans, M, N, K, 1.0f,
                       x_data, y_data, 0, Out->data<T>(), out_batch_size, 0,
                       K * N);
    }
  } else if (y_batch_size == 1) {
    if (!trans_x) {
      VLOG(3) << "MatMul's case 11";
      blas.GEMM(CblasNoTrans, trans_y ? CblasTrans : CblasNoTrans,
                x_batch_size * M, N, K, 1.0f, x_data, y_data, 0.0f,
                Out->data<T>());
    } else {
      VLOG(3) << "MatMul's case 12";
      blas.BatchedGEMM(CblasTrans, trans_y ? CblasTrans : CblasNoTrans, M, N, K,
                       1.0f, x_data, y_data, 0, Out->data<T>(), out_batch_size,
                       M * K, 0);
    }
  } else if (!is_broadcast_dims) {
    VLOG(3) << "MatMul's case 13";
    blas.BatchedGEMM(trans_x ? CblasTrans : CblasNoTrans,
                     trans_y ? CblasTrans : CblasNoTrans, M, N, K, 1.0f, x_data,
                     y_data, 0, Out->data<T>(), out_batch_size, M * K, K * N);
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
      out_ptr[i] = Out->data<T>() + i * M * N;
      IndexIncreaseFromDims(batch_dim, out_broadcast_dims.data(), index.data());
    }
    VLOG(3) << "MatMul's case 14";
    blas.BatchedGEMM(trans_x ? CblasTrans : CblasNoTrans,
                     trans_y ? CblasTrans : CblasNoTrans, M, N, K, 1.0f,
                     x_ptr.data(), y_ptr.data(), 0.0f, out_ptr.data(),
                     out_batch_size);
  }
}

template <typename DeviceContext, typename T>
void MatMulFunction(const Tensor* X, const Tensor* Y, Tensor* Out, bool trans_x,
                    bool trans_y,
                    const paddle::framework::ExecutionContext& ctx) {
  const std::vector<std::int64_t> x_dims = vectorize(X->dims());
  const std::vector<std::int64_t> y_dims = vectorize(Y->dims());
  MatMulFunction<DeviceContext, T>(X, Y, x_dims, y_dims, Out, trans_x, trans_y,
                                   ctx);
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

template <typename DeviceContext, typename T>
class MatMulV2GradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* X = ctx.Input<Tensor>("X");
    auto* Y = ctx.Input<Tensor>("Y");
    auto* dOut = ctx.Input<Tensor>(framework::GradVarName("Out"));
    bool trans_x = ctx.Attr<bool>("trans_x");
    bool trans_y = ctx.Attr<bool>("trans_y");

    // get dims
    std::vector<std::int64_t> x_dims = vectorize(X->dims());
    std::vector<std::int64_t> y_dims = vectorize(Y->dims());
    std::vector<std::int64_t> dout_dims = vectorize(dOut->dims());

    int x_ndim = x_dims.size();
    int y_ndim = y_dims.size();
    int ndim = dout_dims.size();

    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));

    // x's or y's dim = 1
    if (x_ndim == 1 && y_ndim == 1) {
      if (dx) dx->mutable_data<T>(ctx.GetPlace());
      if (dy) dy->mutable_data<T>(ctx.GetPlace());
      if (dOut->numel() == 1) {
        DotGradFunction<DeviceContext, T>(X, Y, dOut, dx, dy, ctx);
        return;
      }
    }
    // It is very tricky. For this broadcast, currently using the reduce sum to
    // get gradient.
    if (x_ndim == 1) {
      x_dims.insert(x_dims.begin() + 0, 1);
      x_ndim += 1;
      if (trans_x)
        dout_dims.push_back(1);
      else
        dout_dims.insert(dout_dims.begin() + ndim - 1, 1);
      ndim += 1;
    }

    if (y_ndim == 1) {
      y_dims.push_back(1);
      y_ndim += 1;
      if (trans_y)
        dout_dims.insert(dout_dims.begin() + ndim - 1, 1);
      else
        dout_dims.push_back(1);
      ndim += 1;
    }

    // the normal case
    Tensor dx_help, dy_help;
    if (trans_x) {
      if (trans_y) {
        // X'Y': dA = Y'G', dB = G'X'
        if (dx)
          MatMulFunction<DeviceContext, T>(Y, dOut, y_dims, dout_dims, &dx_help,
                                           true, true, ctx);
        if (dy)
          MatMulFunction<DeviceContext, T>(dOut, X, dout_dims, x_dims, &dy_help,
                                           true, true, ctx);
      } else {
        // X'Y: dX = YG', dY = XG
        if (dx)
          MatMulFunction<DeviceContext, T>(Y, dOut, y_dims, dout_dims, &dx_help,
                                           false, true, ctx);
        if (dy)
          MatMulFunction<DeviceContext, T>(X, dOut, x_dims, dout_dims, &dy_help,
                                           false, false, ctx);
      }
    } else {
      if (trans_y) {
        // XY': dX = GY, dY = G'X
        if (dx)
          MatMulFunction<DeviceContext, T>(dOut, Y, dout_dims, y_dims, &dx_help,
                                           false, false, ctx);
        if (dy)
          MatMulFunction<DeviceContext, T>(dOut, X, dout_dims, x_dims, &dy_help,
                                           true, false, ctx);
      } else {
        // XY: dX = GY', dY = X'G
        if (dx)
          MatMulFunction<DeviceContext, T>(dOut, Y, dout_dims, y_dims, &dx_help,
                                           false, true, ctx);
        if (dy)
          MatMulFunction<DeviceContext, T>(X, dOut, x_dims, dout_dims, &dy_help,
                                           true, false, ctx);
      }
    }
    // get help dims
    const std::vector<std::int64_t> dx_help_dims = vectorize(dx_help.dims());
    const std::vector<std::int64_t> dy_help_dims = vectorize(dy_help.dims());

    std::vector<std::int64_t> dx_broadcast_dims(ndim);
    std::vector<std::int64_t> dy_broadcast_dims(ndim);

    std::fill(dx_broadcast_dims.data(),
              dx_broadcast_dims.data() + ndim - x_ndim, 1);
    std::fill(dy_broadcast_dims.data(),
              dy_broadcast_dims.data() + ndim - y_ndim, 1);
    std::copy(x_dims.data(), x_dims.data() + x_ndim,
              dx_broadcast_dims.data() + ndim - x_ndim);
    std::copy(y_dims.data(), y_dims.data() + y_ndim,
              dy_broadcast_dims.data() + ndim - y_ndim);

    std::vector<int> dx_reduce_dims;
    std::vector<int> dy_reduce_dims;
    for (int idx = 0; idx <= ndim - 3; idx++) {
      if (dx_help_dims[idx] != 1 && dx_broadcast_dims[idx] == 1) {
        dx_reduce_dims.push_back(idx);
      }
      if (dy_help_dims[idx] != 1 && dy_broadcast_dims[idx] == 1) {
        dy_reduce_dims.push_back(idx);
      }
    }
    // reduce sum to get grad by ReduceSum
    if (dx) {
      dx->Resize(dx_help.dims());
      ReduceSumForMatmulGrad<DeviceContext, T>(&dx_help, dx, dx_reduce_dims,
                                               ctx);
      dx->Resize(X->dims());
    }
    if (dy) {
      dy->Resize(dy_help.dims());
      ReduceSumForMatmulGrad<DeviceContext, T>(&dy_help, dy, dy_reduce_dims,
                                               ctx);
      dy->Resize(Y->dims());
    }
  }
};

}  // namespace operators
}  // namespace paddle
