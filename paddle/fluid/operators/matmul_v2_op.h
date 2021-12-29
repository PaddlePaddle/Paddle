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
#include <utility>
#include <vector>
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/dot_op.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/complex_functors.h"
#include "paddle/fluid/operators/reduce_ops/reduce_sum_op.h"

// only can include the headers in paddle/pten/api dirs
#include "paddle/pten/api/lib/utils/tensor_utils.h"
#include "paddle/pten/include/core.h"
#include "paddle/pten/kernels/matmul_kernel.h"

#if defined(__NVCC__) || defined(__HIPCC__)
#include "paddle/fluid/operators/reduce_ops/reduce_op.cu.h"
#endif

namespace paddle {
namespace operators {

using framework::Tensor;

template <typename DeviceContext, typename T>
void ReduceSumForMatmulGrad(const Tensor* input, Tensor* output,
                            const std::vector<int>& reduce_dims,
                            const paddle::framework::ExecutionContext& ctx) {
#if defined(__NVCC__) || defined(__HIPCC__)
  auto stream = ctx.cuda_device_context().stream();
  TensorReduceFunctorImpl<T, T, kps::AddFunctor, kps::IdentityFunctor<T>>(
      *input, output, kps::IdentityFunctor<T>(), reduce_dims, stream);
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
  const int ndim = (std::max)(x_ndim, y_ndim);
  std::fill(x_bd_dims, x_bd_dims + ndim - x_ndim, 1);
  std::fill(y_bd_dims, y_bd_dims + ndim - y_ndim, 1);
  std::copy(x_dims, x_dims + x_ndim, x_bd_dims + ndim - x_ndim);
  std::copy(y_dims, y_dims + y_ndim, y_bd_dims + ndim - y_ndim);

  for (int i = 0; i < ndim; ++i) {
    PADDLE_ENFORCE_EQ(
        x_bd_dims[i] == y_bd_dims[i] || x_bd_dims[i] <= 1 || y_bd_dims[i] <= 1,
        true,
        platform::errors::InvalidArgument(
            "Input(X) and Input(Y) has error dim."
            "X_broadcast's shape[%s] must be equal to Y_broadcast's shape[%s],"
            "or X_broadcast's shape[%s] <= 1, or Y_broadcast's shape[%s] <= 1,"
            "But received X_broadcast's shape[%s] = [%s]"
            "received Y_broadcast's shape[%s] = [%s]",
            i, i, i, i, i, x_bd_dims[i], i, y_bd_dims[i]));
    if (x_bd_dims[i] == 0 || y_bd_dims[i] == 0) {
      out_bd_dims[i] = 0;
    } else {
      out_bd_dims[i] = (std::max)(x_bd_dims[i], y_bd_dims[i]);
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
                    const paddle::framework::ExecutionContext& ctx,
                    bool flag = false) {
  const int x_ndim = x_dims.size();
  const int y_ndim = y_dims.size();

  // Get data ptr
  const T* x_data = X->data<T>();
  const T* y_data = Y->data<T>();

  if (x_ndim == 1 && y_ndim == 1) {
    PADDLE_ENFORCE_EQ(
        X->numel(), Y->numel(),
        platform::errors::InvalidArgument(
            "X's numbers must be equal to Y's numbers,"
            "when X/Y's dims =1. But received X has [%d] elements,"
            "received Y has [%d] elements",
            X->numel(), Y->numel()));
    VLOG(3) << "MatMul's case 1";
    Out->Resize({1});
    Out->mutable_data<T>(ctx.GetPlace());
    auto out_eigen = framework::EigenScalar<T>::From(*Out);
    auto x_eigen = framework::EigenVector<T>::Flatten(*X);
    auto y_eigen = framework::EigenVector<T>::Flatten(*Y);

    auto& dev = *ctx.template device_context<DeviceContext>().eigen_device();
    if (flag) {
      out_eigen.device(dev) = (x_eigen * y_eigen).sum() + out_eigen;
    } else {
      out_eigen.device(dev) = (x_eigen * y_eigen).sum();
    }
    return;
  }

  auto& dev_ctx = ctx.template device_context<DeviceContext>();
  auto blas = math::GetBlas<DeviceContext, T>(dev_ctx);

  if (x_ndim == 1) {
    const int N = X->numel();
    if (trans_y) {
      PADDLE_ENFORCE_EQ(y_dims[y_ndim - 1], N,
                        platform::errors::InvalidArgument(
                            "Input(Y) has error dim."
                            "Y'dims[%d] must be equal to %d"
                            "But received Y'dims[%d] is %d",
                            y_ndim - 1, N, y_ndim - 1, y_dims[y_ndim - 1]));
    } else {
      PADDLE_ENFORCE_EQ(y_dims[y_ndim - 2], N,
                        platform::errors::InvalidArgument(
                            "Input(Y) has error dim."
                            "Y'dims[%d] must be equal to %d"
                            "But received Y'dims[%d] is %d",
                            y_ndim - 2, N, y_ndim - 2, y_dims[y_ndim - 2]));
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
      blas.GEMV(false, M, N, static_cast<T>(1), y_data, x_data,
                static_cast<T>(flag), Out->data<T>());
    } else {
      const int M = y_dims[y_ndim - 1];
      const int batch_size = Y->numel() / (M * N);
      if (batch_size == 1) {
        VLOG(3) << "MatMul's case 3";
        blas.GEMV(true, N, M, static_cast<T>(1), y_data, x_data,
                  static_cast<T>(flag), Out->data<T>());
      } else {
        VLOG(3) << "MatMul's case 4";
        blas.BatchedGEMM(CblasTrans, CblasNoTrans, M, 1, N, static_cast<T>(1),
                         y_data, x_data, static_cast<T>(flag), Out->data<T>(),
                         batch_size, M * N, 0);
      }
    }
    return;
  }

  if (y_ndim == 1) {
    const int N = Y->numel();
    if (trans_x) {
      PADDLE_ENFORCE_EQ(x_dims[x_ndim - 2], N,
                        platform::errors::InvalidArgument(
                            "Input(X) has error dim."
                            "X'dims[%d] must be equal to %d"
                            "But received X'dims[%d] is %d",
                            x_ndim - 2, N, x_ndim - 2, x_dims[x_ndim - 2]));
    } else {
      PADDLE_ENFORCE_EQ(x_dims[x_ndim - 1], N,
                        platform::errors::InvalidArgument(
                            "Input(X) has error dim."
                            "X'dims[%d] must be equal to %d"
                            "But received X'dims[%d] is %d",
                            x_ndim - 1, N, x_ndim - 1, x_dims[x_ndim - 1]));
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
        blas.GEMV(true, N, M, static_cast<T>(1), x_data, y_data,
                  static_cast<T>(flag), Out->data<T>());
      } else {
        VLOG(3) << "MatMul's case 6";
        blas.BatchedGEMM(CblasTrans, CblasNoTrans, M, 1, N, static_cast<T>(1),
                         x_data, y_data, static_cast<T>(flag), Out->data<T>(),
                         batch_size, M * N, 0);
      }
    } else {
      const int M = X->numel() / N;
      VLOG(3) << "MatMul's case 7";
      blas.GEMV(false, M, N, static_cast<T>(1), x_data, y_data,
                static_cast<T>(flag), Out->data<T>());
    }
    return;
  }

  const int M = trans_x ? x_dims[x_ndim - 1] : x_dims[x_ndim - 2];
  const int K = trans_x ? x_dims[x_ndim - 2] : x_dims[x_ndim - 1];
  if (trans_y) {
    PADDLE_ENFORCE_EQ(y_dims[y_ndim - 1], K,
                      platform::errors::InvalidArgument(
                          "Input(Y) has error dim."
                          "Y'dims[%d] must be equal to %d"
                          "But received Y'dims[%d] is %d",
                          y_ndim - 1, K, y_ndim - 1, y_dims[y_ndim - 1]));
  } else {
    PADDLE_ENFORCE_EQ(y_dims[y_ndim - 2], K,
                      platform::errors::InvalidArgument(
                          "Input(Y) has error dim."
                          "Y'dims[%d] must be equal to %d"
                          "But received Y'dims[%d] is %d",
                          y_ndim - 2, K, y_ndim - 2, y_dims[y_ndim - 2]));
  }
  const int N = trans_y ? y_dims[y_ndim - 2] : y_dims[y_ndim - 1];
  const int ndim = (std::max)(x_ndim, y_ndim);
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
              trans_y ? CblasTrans : CblasNoTrans, M, N, K, static_cast<T>(1),
              x_data, y_data, static_cast<T>(flag), Out->data<T>());
  } else if (x_batch_size == 1) {
    if (M == 1 && trans_y) {
      VLOG(3) << "MatMul's case 9";
      blas.GEMV(false, y_batch_size * N, K, static_cast<T>(1), y_data, x_data,
                static_cast<T>(flag), Out->data<T>());
    } else {
      VLOG(3) << "MatMul's case 10";
      blas.BatchedGEMM(trans_x ? CblasTrans : CblasNoTrans,
                       trans_y ? CblasTrans : CblasNoTrans, M, N, K,
                       static_cast<T>(1), x_data, y_data, static_cast<T>(flag),
                       Out->data<T>(), out_batch_size, 0, K * N);
    }
  } else if (y_batch_size == 1) {
    if (!trans_x) {
      VLOG(3) << "MatMul's case 11";
      blas.GEMM(CblasNoTrans, trans_y ? CblasTrans : CblasNoTrans,
                x_batch_size * M, N, K, static_cast<T>(1), x_data, y_data,
                static_cast<T>(flag), Out->data<T>());
    } else {
      VLOG(3) << "MatMul's case 12";
      blas.BatchedGEMM(CblasTrans, trans_y ? CblasTrans : CblasNoTrans, M, N, K,
                       static_cast<T>(1), x_data, y_data, static_cast<T>(flag),
                       Out->data<T>(), out_batch_size, M * K, 0);
    }
  } else if (!is_broadcast_dims) {
    VLOG(3) << "MatMul's case 13";
    blas.BatchedGEMM(trans_x ? CblasTrans : CblasNoTrans,
                     trans_y ? CblasTrans : CblasNoTrans, M, N, K,
                     static_cast<T>(1), x_data, y_data, static_cast<T>(flag),
                     Out->data<T>(), out_batch_size, M * K, K * N);
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
                     trans_y ? CblasTrans : CblasNoTrans, M, N, K,
                     static_cast<T>(1), x_ptr.data(), y_ptr.data(),
                     static_cast<T>(flag), out_ptr.data(), out_batch_size);
  }
}

template <typename DeviceContext, typename T>
void MatMulFunction(const Tensor* X, const Tensor* Y, Tensor* Out, bool trans_x,
                    bool trans_y,
                    const paddle::framework::ExecutionContext& ctx,
                    bool flag = false) {
  const std::vector<std::int64_t> x_dims = vectorize(X->dims());
  const std::vector<std::int64_t> y_dims = vectorize(Y->dims());
  MatMulFunction<DeviceContext, T>(X, Y, x_dims, y_dims, Out, trans_x, trans_y,
                                   ctx, flag);
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

    auto& dev_ctx = ctx.device_context<DeviceContext>();
    Out->mutable_data<T>(X->place());

    auto pt_x = paddle::experimental::MakePtenDenseTensor(*X);
    auto pt_y = paddle::experimental::MakePtenDenseTensor(*Y);
    auto pt_out = paddle::experimental::MakePtenDenseTensor(*Out);

    // call new kernel
    pten::Matmul<T>(dev_ctx, *pt_x.get(), *pt_y.get(), trans_x, trans_y,
                    pt_out.get());
  }
};

// Reshape a rank-3 tensor from P x M x N to (P * M) x N.
// Identity op if the tensor is not of rank 3.
static framework::Tensor FoldInitDims(const framework::Tensor& input) {
  auto output = input;
  auto in_dims = input.dims();
  if (in_dims.size() == 3) {
    output.Resize({in_dims[0] * in_dims[1], in_dims[2]});
  }
  return output;
}

// Reshape a rank-3 tensor from P x M x N to M x (P * N).
// (Warning: This requires transposing data and writes into new memory.)
// Identity op if the tensor is not of rank 3.
template <typename DeviceContext, typename T>
static framework::Tensor FoldHeadAndLastDims(const DeviceContext& context,
                                             const framework::Tensor& input) {
  auto in_dims = input.dims();
  if (in_dims.size() != 3) {
    return input;
  }
  framework::Tensor output;
  output.Resize({in_dims[1], in_dims[0], in_dims[2]});
  output.mutable_data<T>(context.GetPlace());
  std::vector<int> axis = {1, 0, 2};
  math::Transpose<DeviceContext, T, 3> trans;
  trans(context, input, &output, axis);
  output.Resize({in_dims[1], in_dims[0] * in_dims[2]});
  return output;
}

/**
 * Get row matrix shape from a vector shape. If the rank of x_dim > 1, the
 * original x_dim is returned.
 */
static framework::DDim RowMatrixFromVector(const framework::DDim& x_dim) {
  if (x_dim.size() > 1) {
    return x_dim;
  }
  return framework::make_ddim({1, x_dim[0]});
}

/**
 * Get column matrix shape from a vector shape. If the ran of y_dim > 1, the
 * original y_dim is returned.
 */
static framework::DDim ColumnMatrixFromVector(const framework::DDim& y_dim) {
  if (y_dim.size() > 1) {
    return y_dim;
  }
  return framework::make_ddim({y_dim[0], 1});
}

/**
 * Reshape a tensor to 3-D or 2-D tensor by matrix descriptor.
 *
 * The shape would be [BatchSize, H, W] or [H, W].
 * If transposed, `H,W` will be swapped.
 */
static void ReshapeTensorIntoMatrixSequence(
    framework::Tensor* x, const math::MatDescriptor& descriptor) {
  int64_t h, w;
  h = descriptor.height_;
  w = descriptor.width_;
  if (descriptor.trans_) {
    std::swap(w, h);
  }
  if (descriptor.batch_size_) {
    x->Resize({descriptor.batch_size_, h, w});
  } else {
    x->Resize({h, w});
  }
}

static void ReshapeXYOutIntoMatrixSequence(framework::Tensor* x,
                                           framework::Tensor* y,
                                           framework::Tensor* out, bool trans_x,
                                           bool trans_y) {
  auto x_dim = RowMatrixFromVector(x->dims());
  auto y_dim = ColumnMatrixFromVector(y->dims());
  auto mat_dim_x = math::CreateMatrixDescriptor(x_dim, 0, trans_x);
  auto mat_dim_y = math::CreateMatrixDescriptor(y_dim, 0, trans_y);
  if (mat_dim_x.batch_size_ == 0 && mat_dim_y.batch_size_ == 0) {
    out->Resize({mat_dim_x.height_, mat_dim_y.width_});
  } else {
    out->Resize({(std::max)(mat_dim_x.batch_size_, mat_dim_y.batch_size_),
                 mat_dim_x.height_, mat_dim_y.width_});
  }

  ReshapeTensorIntoMatrixSequence(x, mat_dim_x);
  ReshapeTensorIntoMatrixSequence(y, mat_dim_y);
}

template <typename DeviceContext, typename T>
struct ConjHelper {
  explicit ConjHelper(const framework::ExecutionContext& ctx) : ctx_(ctx) {}
  HOSTDEVICE void operator()(framework::Tensor& src, framework::Tensor& dst) {
    dst.Resize(src.dims());
    dst.set_layout(src.layout());
    dst.ShareDataWith(src);
    return;
  }

  const framework::ExecutionContext& ctx_;
};

template <typename DeviceContext>
struct ConjHelper<DeviceContext, paddle::platform::complex<float>> {
  explicit ConjHelper(const framework::ExecutionContext& ctx) : ctx_(ctx) {}

  HOSTDEVICE void operator()(framework::Tensor& src, framework::Tensor& dst) {
    dst.Resize(src.dims());
    auto* src_data = src.data<paddle::platform::complex<float>>();
    auto* dst_data = dst.mutable_data<paddle::platform::complex<float>>(
        ctx_.GetPlace(),
        size_t(src.numel() * sizeof(paddle::platform::complex<float>)));

    platform::ForRange<DeviceContext> for_range(
        ctx_.template device_context<DeviceContext>(), src.numel());
    math::ConjFunctor<paddle::platform::complex<float>> functor(
        src_data, src.numel(), dst_data);
    for_range(functor);
    return;
  }
  const framework::ExecutionContext& ctx_;
};

template <typename DeviceContext>
struct ConjHelper<DeviceContext, paddle::platform::complex<double>> {
  explicit ConjHelper(const framework::ExecutionContext& ctx) : ctx_(ctx) {}

  HOSTDEVICE void operator()(framework::Tensor& src, framework::Tensor& dst) {
    dst.Resize(src.dims());
    auto* src_data = src.data<paddle::platform::complex<double>>();
    auto* dst_data = dst.mutable_data<paddle::platform::complex<double>>(
        ctx_.GetPlace(),
        size_t(src.numel() * sizeof(paddle::platform::complex<double>)));

    platform::ForRange<DeviceContext> for_range(
        ctx_.template device_context<DeviceContext>(), src.numel());
    math::ConjFunctor<paddle::platform::complex<double>> functor(
        src_data, src.numel(), dst_data);
    for_range(functor);
    return;
  }
  const framework::ExecutionContext& ctx_;
};

template <typename DeviceContext, typename T, typename Enabel = void>
struct DotDoubleGradFunction {
  void operator()(const Tensor* tensor_x, const Tensor* tensor_y,
                  Tensor* tensor_dx, Tensor* tensor_dy,
                  const Tensor* tensor_dout, const Tensor* tensor_ddx,
                  const Tensor* tensor_ddy, Tensor* tensor_ddout,
                  const paddle::framework::ExecutionContext& ctx);
};

template <typename DeviceContext, typename T>
struct DotDoubleGradFunction<DeviceContext, T, math::EnableComplex<T>> {
  void operator()(const Tensor* tensor_x, const Tensor* tensor_y,
                  Tensor* tensor_dx, Tensor* tensor_dy,
                  const Tensor* tensor_dout, const Tensor* tensor_ddx,
                  const Tensor* tensor_ddy, Tensor* tensor_ddout,
                  const paddle::framework::ExecutionContext& ctx) {
#if defined(__NVCC__) || defined(__HIPCC__)
    if (1 == tensor_dout->dims().size()) {
      framework::Tensor tensor_dout_help;
      auto& dev_raw = ctx.template device_context<DeviceContext>();
      auto& dev = *dev_raw.eigen_device();
      if (tensor_dx || tensor_dy) {
        tensor_dout_help.Resize(tensor_dout->dims());
        tensor_dout_help.mutable_data<T>(ctx.GetPlace());
        paddle::platform::ForRange<DeviceContext> for_range(
            dev_raw, tensor_dout->numel());
        math::ConjFunctor<T> functor(tensor_dout->data<T>(),
                                     tensor_dout->numel(),
                                     tensor_dout_help.data<T>());
        for_range(functor);
      }
      if (tensor_dx) {
        auto ddy = framework::EigenVector<T>::Flatten(*tensor_ddy);
        Eigen::DSizes<int, 1> size(tensor_ddy->numel());
        auto dx = framework::EigenVector<T>::Flatten(*tensor_dx);
        auto dout = framework::EigenVector<T>::Flatten(tensor_dout_help);
        dx.device(dev) = ddy * dout.broadcast(size);
      }

      if (tensor_dy) {
        auto ddx = framework::EigenVector<T>::Flatten(*tensor_ddx);
        Eigen::DSizes<int, 1> size(tensor_ddx->numel());
        auto dy = framework::EigenVector<T>::Flatten(*tensor_dy);
        auto dout = framework::EigenVector<T>::Flatten(tensor_dout_help);
        dy.device(dev) = ddx * dout.broadcast(size);
      }

      if (tensor_ddout) {
        framework::Tensor tensor_x_help, tensor_y_help;
        tensor_x_help.Resize(tensor_x->dims());
        tensor_x_help.mutable_data<T>(ctx.GetPlace());
        tensor_y_help.Resize(tensor_y->dims());
        tensor_y_help.mutable_data<T>(ctx.GetPlace());

        auto& dev_raw = ctx.template device_context<DeviceContext>();
        auto& dev = *dev_raw.eigen_device();
        paddle::platform::ForRange<DeviceContext> for_range(dev_raw,
                                                            tensor_x->numel());
        math::ConjFunctor<T> functor_x(tensor_x->data<T>(), tensor_x->numel(),
                                       tensor_x_help.data<T>());
        for_range(functor_x);
        math::ConjFunctor<T> functor_y(tensor_y->data<T>(), tensor_y->numel(),
                                       tensor_y_help.data<T>());
        for_range(functor_y);
        auto x = framework::EigenVector<T>::Flatten(tensor_x_help);
        auto y = framework::EigenVector<T>::Flatten(tensor_y_help);
        auto ddx = framework::EigenVector<T>::Flatten(*tensor_ddx);
        auto ddy = framework::EigenVector<T>::Flatten(*tensor_ddy);
        auto ddout = framework::EigenVector<T>::Flatten(*tensor_ddout);
        ddout.device(dev) = (x * ddy + y * ddx).sum();
      }
    }
#else
    const auto* data_dout = tensor_dout->data<T>();

    if (tensor_dx) {
      auto* data_dx = tensor_dx->mutable_data<T>(ctx.GetPlace());
      const auto* data_ddy = tensor_ddy->data<T>();
      const framework::DDim& dim = tensor_dx->dims();
      size_t N = static_cast<size_t>(framework::product(dim));

      auto step = dim[dim.size() - 1];

      int s = -1;
      for (size_t i = 0; i < N; ++i) {
        if (0 == i % step) ++s;
        data_dx[i] = T(data_dout[s].real, -data_dout[s].imag) * data_ddy[i];
      }
    }

    if (tensor_dy) {
      auto* data_dy = tensor_dy->mutable_data<T>(ctx.GetPlace());
      const auto* data_ddx = tensor_ddx->data<T>();
      const framework::DDim& dim = tensor_dy->dims();
      size_t N = static_cast<size_t>(framework::product(dim));

      auto step = dim[dim.size() - 1];

      int s = -1;
      for (size_t i = 0; i < N; ++i) {
        if (0 == i % step) ++s;
        data_dy[i] = T(data_dout[s].real, -data_dout[s].imag) * data_ddx[i];
      }
    }

    if (tensor_ddout) {
      auto* data_ddout = tensor_ddout->mutable_data<T>(ctx.GetPlace());
      auto* data_x = tensor_x->data<T>();
      auto* data_y = tensor_y->data<T>();
      auto* data_ddx = tensor_ddx->data<T>();
      auto* data_ddy = tensor_ddy->data<T>();

      const framework::DDim& dim = tensor_dy->dims();
      size_t N = static_cast<size_t>(framework::product(dim));
      auto step = dim[dim.size() - 1];
      int s = -1;
      bool new_s = false;

      for (size_t i = 0; i < N; ++i) {
        if (0 == i % step) {
          ++s;
          new_s = true;
        }
        if (new_s) {
          data_ddout[s] = T(data_x[i].real, -data_x[i].imag) * data_ddy[i] +
                          T(data_y[i].real, -data_y[i].imag) * data_ddx[i];
        } else {
          data_ddout[s] += T(data_x[i].real, -data_x[i].imag) * data_ddy[i] +
                           T(data_y[i].real, -data_y[i].imag) * data_ddx[i];
        }
        new_s = false;
      }
    }
#endif
  }
};

template <typename DeviceContext, typename T>
struct DotDoubleGradFunction<DeviceContext, T, math::DisableComplex<T>> {
  void operator()(const Tensor* tensor_x, const Tensor* tensor_y,
                  Tensor* tensor_dx, Tensor* tensor_dy,
                  const Tensor* tensor_dout, const Tensor* tensor_ddx,
                  const Tensor* tensor_ddy, Tensor* tensor_ddout,
                  const paddle::framework::ExecutionContext& ctx) {
#if defined(__NVCC__) || defined(__HIPCC__)
    if (1 == tensor_dout->dims().size()) {
      auto& dev_raw = ctx.template device_context<DeviceContext>();
      auto& dev = *dev_raw.eigen_device();
      auto dout = framework::EigenVector<T>::Flatten(*tensor_dout);
      if (tensor_dx) {
        tensor_dx->mutable_data<T>(ctx.GetPlace());
        auto ddy = framework::EigenVector<T>::Flatten(*tensor_ddy);
        Eigen::DSizes<int, 1> size(tensor_ddy->numel());
        auto dx = framework::EigenVector<T>::Flatten(*tensor_dx);
        dx.device(dev) = ddy * dout.broadcast(size);
      }

      if (tensor_dy) {
        tensor_dy->mutable_data<T>(ctx.GetPlace());
        auto ddx = framework::EigenVector<T>::Flatten(*tensor_ddx);
        Eigen::DSizes<int, 1> size(tensor_ddx->numel());

        auto dy = framework::EigenVector<T>::Flatten(*tensor_dy);
        dy.device(dev) = ddx * dout.broadcast(size);
      }

      if (tensor_ddout) {
        tensor_ddout->mutable_data<T>(ctx.GetPlace());
        auto x = framework::EigenVector<T>::Flatten(*tensor_x);
        auto y = framework::EigenVector<T>::Flatten(*tensor_y);
        auto ddx = framework::EigenVector<T>::Flatten(*tensor_ddx);
        auto ddy = framework::EigenVector<T>::Flatten(*tensor_ddy);
        auto ddout = framework::EigenVector<T>::Flatten(*tensor_ddout);
        ddout.device(dev) = (x * ddy + y * ddx).sum();
      }
    }
#else
    const auto* data_dout = tensor_dout->data<T>();

    if (tensor_dx) {
      auto* data_dx = tensor_dx->mutable_data<T>(ctx.GetPlace());
      const auto* data_ddy = tensor_ddy->data<T>();
      const framework::DDim& dim = tensor_dx->dims();
      size_t N = static_cast<size_t>(framework::product(dim));

      auto step = dim[dim.size() - 1];

      int s = -1;
      for (size_t i = 0; i < N; ++i) {
        if (0 == i % step) ++s;
        data_dx[i] = data_dout[s] * data_ddy[i];
      }
    }

    if (tensor_dy) {
      auto* data_dy = tensor_dy->mutable_data<T>(ctx.GetPlace());
      const auto* data_ddx = tensor_ddx->data<T>();
      const framework::DDim& dim = tensor_dy->dims();
      size_t N = static_cast<size_t>(framework::product(dim));

      auto step = dim[dim.size() - 1];

      int s = -1;
      for (size_t i = 0; i < N; ++i) {
        if (0 == i % step) ++s;
        data_dy[i] = data_dout[s] * data_ddx[i];
      }
    }

    if (tensor_ddout) {
      auto* data_ddout = tensor_ddout->mutable_data<T>(ctx.GetPlace());
      auto* data_x = tensor_x->data<T>();
      auto* data_y = tensor_y->data<T>();
      auto* data_ddx = tensor_ddx->data<T>();
      auto* data_ddy = tensor_ddy->data<T>();

      const framework::DDim& dim = tensor_dy->dims();
      size_t N = static_cast<size_t>(framework::product(dim));
      auto step = dim[dim.size() - 1];
      int s = -1;
      bool new_s = false;

      for (size_t i = 0; i < N; ++i) {
        if (0 == i % step) {
          ++s;
          new_s = true;
        }
        if (new_s) {
          data_ddout[s] = data_x[i] * data_ddy[i] + data_y[i] * data_ddx[i];
        } else {
          data_ddout[s] += data_x[i] * data_ddy[i] + data_y[i] * data_ddx[i];
        }
        new_s = false;
      }
    }
#endif
  }
};

template <typename DeviceContext, typename T, typename Enabel = void>
struct DotTripleGradFunction {
  void operator()(const Tensor* in_tensor_x, const Tensor* in_tensor_y,
                  const Tensor* in_tensor_ddx, const Tensor* in_tensor_ddy,
                  const Tensor* in_tensor_d_dx, const Tensor* in_tensor_d_dy,
                  const Tensor* in_tensor_dout, const Tensor* in_tensor_d_ddout,
                  Tensor* out_tensor_d_x, Tensor* out_tensor_d_y,
                  Tensor* out_tensor_d_dout, Tensor* out_tensor_d_ddx,
                  Tensor* out_tensor_d_ddy,
                  const paddle::framework::ExecutionContext& ctx);
};

// TODO(wuweilong): enable this function when the unittests framewark for multi
// grad is ok (dtype: complex64 or complex128).
template <typename DeviceContext, typename T>
struct DotTripleGradFunction<DeviceContext, T, math::EnableComplex<T>> {
  void operator()(const Tensor* in_tensor_x, const Tensor* in_tensor_y,
                  const Tensor* in_tensor_ddx, const Tensor* in_tensor_ddy,
                  const Tensor* in_tensor_d_dx, const Tensor* in_tensor_d_dy,
                  const Tensor* in_tensor_dout, const Tensor* in_tensor_d_ddout,
                  Tensor* out_tensor_d_x, Tensor* out_tensor_d_y,
                  Tensor* out_tensor_d_dout, Tensor* out_tensor_d_ddx,
                  Tensor* out_tensor_d_ddy,
                  const paddle::framework::ExecutionContext& ctx) {
#if defined(__NVCC__) || defined(__HIPCC__)
    if (1 == in_tensor_d_ddout->dims().size()) {
      framework::Tensor in_tensor_d_ddout_help;
      auto& dev_raw = ctx.template device_context<DeviceContext>();
      auto& dev = *dev_raw.eigen_device();
      if (out_tensor_d_x || out_tensor_d_y) {
        in_tensor_d_ddout_help.Resize(in_tensor_d_ddout->dims());
        in_tensor_d_ddout_help.mutable_data<T>(ctx.GetPlace());
        paddle::platform::ForRange<DeviceContext> for_range(
            dev_raw, in_tensor_d_ddout->numel());
        math::ConjFunctor<T> functor(in_tensor_d_ddout->data<T>(),
                                     in_tensor_d_ddout->numel(),
                                     in_tensor_d_ddout_help.data<T>());
        for_range(functor);
      }
      if (out_tensor_d_x) {
        auto ddy = framework::EigenVector<T>::Flatten(*in_tensor_ddy);
        Eigen::DSizes<int, 1> size(in_tensor_ddy->numel());
        auto d_x = framework::EigenVector<T>::Flatten(*out_tensor_d_x);
        auto d_ddout =
            framework::EigenVector<T>::Flatten(in_tensor_d_ddout_help);
        d_x.device(dev) = ddy * d_ddout.broadcast(size);
      }

      if (out_tensor_d_y) {
        auto ddx = framework::EigenVector<T>::Flatten(*in_tensor_ddx);
        Eigen::DSizes<int, 1> size(in_tensor_ddx->numel());
        auto d_y = framework::EigenVector<T>::Flatten(*out_tensor_d_y);
        auto d_ddout =
            framework::EigenVector<T>::Flatten(in_tensor_d_ddout_help);
        d_y.device(dev) = ddx * d_ddout.broadcast(size);
      }

      if (out_tensor_d_dout) {
        framework::Tensor in_tensor_ddx_help, in_tensor_ddy_help;
        in_tensor_ddx_help.Resize(in_tensor_ddx->dims());
        in_tensor_ddx_help.mutable_data<T>(ctx.GetPlace());
        in_tensor_ddy_help.Resize(in_tensor_ddy->dims());
        in_tensor_ddy_help.mutable_data<T>(ctx.GetPlace());

        auto& dev_raw = ctx.template device_context<DeviceContext>();
        auto& dev = *dev_raw.eigen_device();
        paddle::platform::ForRange<DeviceContext> for_range(
            dev_raw, in_tensor_ddx->numel());
        math::ConjFunctor<T> functor_ddx(in_tensor_ddx->data<T>(),
                                         in_tensor_ddx->numel(),
                                         in_tensor_ddx_help.data<T>());
        for_range(functor_ddx);
        math::ConjFunctor<T> functor_ddy(in_tensor_ddy->data<T>(),
                                         in_tensor_ddy->numel(),
                                         in_tensor_ddy_help.data<T>());
        for_range(functor_ddy);
        auto ddx = framework::EigenVector<T>::Flatten(in_tensor_ddx_help);
        auto ddy = framework::EigenVector<T>::Flatten(in_tensor_ddy_help);
        auto d_dx = framework::EigenVector<T>::Flatten(*in_tensor_d_dx);
        auto d_dy = framework::EigenVector<T>::Flatten(*in_tensor_d_dy);
        auto d_dout = framework::EigenVector<T>::Flatten(*out_tensor_d_dout);
        d_dout.device(dev) = (ddx * d_dy + ddy * d_dx).sum();
      }
      if (out_tensor_d_ddx) {
        framework::Tensor in_tensor_dout_help, in_tensor_y_help;
        in_tensor_dout_help.Resize(in_tensor_dout->dims());
        in_tensor_dout_help.mutable_data<T>(ctx.GetPlace());
        in_tensor_y_help.Resize(in_tensor_y->dims());
        in_tensor_y_help.mutable_data<T>(ctx.GetPlace());

        auto& dev_raw = ctx.template device_context<DeviceContext>();
        auto& dev = *dev_raw.eigen_device();
        paddle::platform::ForRange<DeviceContext> for_range(
            dev_raw, in_tensor_dout->numel());
        math::ConjFunctor<T> functor_dout(in_tensor_dout->data<T>(),
                                          in_tensor_dout->numel(),
                                          in_tensor_dout_help.data<T>());
        for_range(functor_dout);
        math::ConjFunctor<T> functor_y(in_tensor_y->data<T>(),
                                       in_tensor_y->numel(),
                                       in_tensor_y_help.data<T>());
        for_range(functor_y);
        auto dout = framework::EigenVector<T>::Flatten(in_tensor_dout_help);
        auto y = framework::EigenVector<T>::Flatten(in_tensor_y_help);
        auto d_ddout = framework::EigenVector<T>::Flatten(*in_tensor_d_ddout);
        auto d_dy = framework::EigenVector<T>::Flatten(*in_tensor_d_dy);
        auto d_ddx = framework::EigenVector<T>::Flatten(*out_tensor_d_ddx);
        Eigen::DSizes<int, 1> size(in_tensor_y->numel());
        d_ddx.device(dev) =
            (dout.broadcast(size) * d_dy + y * d_ddout.broadcast(size));
      }
      if (out_tensor_d_ddy) {
        framework::Tensor in_tensor_dout_help, in_tensor_x_help;
        in_tensor_dout_help.Resize(in_tensor_dout->dims());
        in_tensor_dout_help.mutable_data<T>(ctx.GetPlace());
        in_tensor_x_help.Resize(in_tensor_x->dims());
        in_tensor_x_help.mutable_data<T>(ctx.GetPlace());

        auto& dev_raw = ctx.template device_context<DeviceContext>();
        auto& dev = *dev_raw.eigen_device();
        paddle::platform::ForRange<DeviceContext> for_range(
            dev_raw, in_tensor_dout->numel());
        math::ConjFunctor<T> functor_dout(in_tensor_dout->data<T>(),
                                          in_tensor_dout->numel(),
                                          in_tensor_dout_help.data<T>());
        for_range(functor_dout);
        math::ConjFunctor<T> functor_x(in_tensor_x->data<T>(),
                                       in_tensor_x->numel(),
                                       in_tensor_x_help.data<T>());
        for_range(functor_x);
        auto dout = framework::EigenVector<T>::Flatten(in_tensor_dout_help);
        auto x = framework::EigenVector<T>::Flatten(in_tensor_x_help);
        auto d_ddout = framework::EigenVector<T>::Flatten(*in_tensor_d_ddout);
        auto d_dx = framework::EigenVector<T>::Flatten(*in_tensor_d_dx);
        auto d_ddy = framework::EigenVector<T>::Flatten(*out_tensor_d_ddy);
        Eigen::DSizes<int, 1> size(in_tensor_x->numel());
        d_ddy.device(dev) =
            (dout.broadcast(size) * d_dx + x * d_ddout.broadcast(size));
      }
    }
#else
    const auto* data_d_ddout = in_tensor_d_ddout->data<T>();

    if (out_tensor_d_x) {
      auto* data_d_x = out_tensor_d_x->mutable_data<T>(ctx.GetPlace());
      const auto* data_ddy = in_tensor_ddy->data<T>();

      const framework::DDim& dim = out_tensor_d_x->dims();
      size_t N = static_cast<size_t>(framework::product(dim));
      auto step = dim[dim.size() - 1];
      int s = -1;

      for (size_t i = 0; i < N; ++i) {
        if (0 == i % step) ++s;
        data_d_x[i] = T(data_ddy[i].real, -data_ddy[i].imag) * data_d_ddout[s];
      }
    }

    if (out_tensor_d_y) {
      auto* data_d_y = out_tensor_d_y->mutable_data<T>(ctx.GetPlace());
      const auto* data_ddx = in_tensor_ddx->data<T>();

      const framework::DDim& dim = out_tensor_d_y->dims();
      size_t N = static_cast<size_t>(framework::product(dim));
      auto step = dim[dim.size() - 1];
      int s = -1;

      for (size_t i = 0; i < N; ++i) {
        if (0 == i % step) ++s;
        data_d_y[i] = T(data_ddx[i].real, -data_ddx[i].imag) * data_d_ddout[s];
      }
    }

    if (out_tensor_d_dout) {
      auto* data_d_dout = out_tensor_d_dout->mutable_data<T>(ctx.GetPlace());
      auto* data_ddx = in_tensor_ddx->data<T>();
      auto* data_ddy = in_tensor_ddy->data<T>();
      auto* data_d_dx = in_tensor_d_dx->data<T>();
      auto* data_d_dy = in_tensor_d_dy->data<T>();

      const framework::DDim& dim = out_tensor_d_dout->dims();
      size_t N = static_cast<size_t>(framework::product(dim));
      auto step = dim[dim.size() - 1];
      int s = -1;
      bool new_s = false;

      for (size_t i = 0; i < N; ++i) {
        if (0 == i % step) {
          ++s;
          new_s = true;
        }
        if (new_s) {
          data_d_dout[s] =
              T(data_ddy[i].real, -data_ddy[i].imag) * data_d_dx[i] +
              T(data_ddx[i].real, -data_ddx[i].imag) * data_d_dy[i];
        } else {
          data_d_dout[s] +=
              T(data_ddy[i].real, -data_ddy[i].imag) * data_d_dx[i] +
              T(data_ddx[i].real, -data_ddx[i].imag) * data_d_dy[i];
        }
        new_s = false;
      }
    }

    if (out_tensor_d_ddx) {
      auto* data_d_ddx = out_tensor_d_ddx->mutable_data<T>(ctx.GetPlace());
      auto* data_dout = in_tensor_dout->data<T>();
      auto* data_d_dy = in_tensor_d_dy->data<T>();
      auto* data_y = in_tensor_y->data<T>();
      auto* data_d_ddout = in_tensor_d_ddout->data<T>();

      const framework::DDim& dim = out_tensor_d_ddx->dims();
      size_t N = static_cast<size_t>(framework::product(dim));
      auto step = dim[dim.size() - 1];
      int s = -1;

      for (size_t i = 0; i < N; ++i) {
        if (0 == i % step) ++s;
        data_d_ddx[i] =
            T(data_dout[s].real, -data_dout[s].imag) * data_d_dy[i] +
            T(data_y[i].real, -data_y[i].imag) * data_d_ddout[s];
      }
    }

    if (out_tensor_d_ddy) {
      auto* data_d_ddy = out_tensor_d_ddy->mutable_data<T>(ctx.GetPlace());
      auto* data_dout = in_tensor_dout->data<T>();
      auto* data_d_dx = in_tensor_d_dx->data<T>();
      auto* data_x = in_tensor_x->data<T>();
      auto* data_d_ddout = in_tensor_d_ddout->data<T>();

      const framework::DDim& dim = out_tensor_d_ddy->dims();
      size_t N = static_cast<size_t>(framework::product(dim));
      auto step = dim[dim.size() - 1];
      int s = -1;

      for (size_t i = 0; i < N; ++i) {
        if (0 == i % step) ++s;
        data_d_ddy[i] =
            T(data_dout[s].real, -data_dout[s].imag) * data_d_dx[i] +
            T(data_x[i].real, -data_x[i].imag) * data_d_ddout[s];
      }
    }
#endif
  }
};

template <typename DeviceContext, typename T>
struct DotTripleGradFunction<DeviceContext, T, math::DisableComplex<T>> {
  void operator()(const Tensor* in_tensor_x, const Tensor* in_tensor_y,
                  const Tensor* in_tensor_ddx, const Tensor* in_tensor_ddy,
                  const Tensor* in_tensor_d_dx, const Tensor* in_tensor_d_dy,
                  const Tensor* in_tensor_dout, const Tensor* in_tensor_d_ddout,
                  Tensor* out_tensor_d_x, Tensor* out_tensor_d_y,
                  Tensor* out_tensor_d_dout, Tensor* out_tensor_d_ddx,
                  Tensor* out_tensor_d_ddy,
                  const paddle::framework::ExecutionContext& ctx) {
#if defined(__NVCC__) || defined(__HIPCC__)
    if (1 == in_tensor_d_ddout->dims().size()) {
      auto& dev_raw = ctx.template device_context<DeviceContext>();
      auto& dev = *dev_raw.eigen_device();
      auto d_ddout = framework::EigenVector<T>::Flatten(*in_tensor_d_ddout);
      if (out_tensor_d_x) {
        out_tensor_d_x->mutable_data<T>(ctx.GetPlace());
        auto ddy = framework::EigenVector<T>::Flatten(*in_tensor_ddy);
        Eigen::DSizes<int, 1> size(in_tensor_ddy->numel());
        auto d_x = framework::EigenVector<T>::Flatten(*out_tensor_d_x);
        d_x.device(dev) = ddy * d_ddout.broadcast(size);
      }

      if (out_tensor_d_y) {
        out_tensor_d_y->mutable_data<T>(ctx.GetPlace());
        auto ddx = framework::EigenVector<T>::Flatten(*in_tensor_ddx);
        Eigen::DSizes<int, 1> size(in_tensor_ddx->numel());

        auto d_y = framework::EigenVector<T>::Flatten(*out_tensor_d_y);
        d_y.device(dev) = ddx * d_ddout.broadcast(size);
      }

      if (out_tensor_d_dout) {
        out_tensor_d_dout->mutable_data<T>(ctx.GetPlace());
        auto ddx = framework::EigenVector<T>::Flatten(*in_tensor_ddx);
        auto ddy = framework::EigenVector<T>::Flatten(*in_tensor_ddy);
        auto d_dx = framework::EigenVector<T>::Flatten(*in_tensor_d_dx);
        auto d_dy = framework::EigenVector<T>::Flatten(*in_tensor_d_dy);
        auto d_dout = framework::EigenVector<T>::Flatten(*out_tensor_d_dout);
        d_dout.device(dev) = (ddx * d_dy + ddy * d_dx).sum();
      }

      if (out_tensor_d_ddx) {
        out_tensor_d_ddx->mutable_data<T>(ctx.GetPlace());
        auto dout = framework::EigenVector<T>::Flatten(*in_tensor_dout);
        auto y = framework::EigenVector<T>::Flatten(*in_tensor_y);
        auto d_ddout = framework::EigenVector<T>::Flatten(*in_tensor_d_ddout);
        auto d_dy = framework::EigenVector<T>::Flatten(*in_tensor_d_dy);
        auto d_ddx = framework::EigenVector<T>::Flatten(*out_tensor_d_ddx);
        Eigen::DSizes<int, 1> size(in_tensor_y->numel());
        d_ddx.device(dev) =
            (dout.broadcast(size) * d_dy + y * d_ddout.broadcast(size));
      }

      if (out_tensor_d_ddy) {
        out_tensor_d_ddy->mutable_data<T>(ctx.GetPlace());
        auto dout = framework::EigenVector<T>::Flatten(*in_tensor_dout);
        auto x = framework::EigenVector<T>::Flatten(*in_tensor_x);
        auto d_ddout = framework::EigenVector<T>::Flatten(*in_tensor_d_ddout);
        auto d_dx = framework::EigenVector<T>::Flatten(*in_tensor_d_dx);
        auto d_ddy = framework::EigenVector<T>::Flatten(*out_tensor_d_ddy);
        Eigen::DSizes<int, 1> size(in_tensor_x->numel());
        d_ddy.device(dev) =
            (dout.broadcast(size) * d_dx + x * d_ddout.broadcast(size));
      }
    }
#else
    const auto* data_d_ddout = in_tensor_d_ddout->data<T>();

    if (out_tensor_d_x) {
      auto* data_d_x = out_tensor_d_x->mutable_data<T>(ctx.GetPlace());
      const auto* data_ddy = in_tensor_ddy->data<T>();

      const framework::DDim& dim = out_tensor_d_x->dims();
      size_t N = static_cast<size_t>(framework::product(dim));
      auto step = dim[dim.size() - 1];
      int s = -1;

      for (size_t i = 0; i < N; ++i) {
        if (0 == i % step) ++s;
        data_d_x[i] = data_ddy[i] * data_d_ddout[s];
      }
    }

    if (out_tensor_d_y) {
      auto* data_d_y = out_tensor_d_y->mutable_data<T>(ctx.GetPlace());
      const auto* data_ddx = in_tensor_ddx->data<T>();

      const framework::DDim& dim = out_tensor_d_y->dims();
      size_t N = static_cast<size_t>(framework::product(dim));
      auto step = dim[dim.size() - 1];
      int s = -1;

      for (size_t i = 0; i < N; ++i) {
        if (0 == i % step) ++s;
        data_d_y[i] = data_ddx[i] * data_d_ddout[s];
      }
    }

    if (out_tensor_d_dout) {
      auto* data_d_dout = out_tensor_d_dout->mutable_data<T>(ctx.GetPlace());
      auto* data_ddx = in_tensor_ddx->data<T>();
      auto* data_ddy = in_tensor_ddy->data<T>();
      auto* data_d_dx = in_tensor_d_dx->data<T>();
      auto* data_d_dy = in_tensor_d_dy->data<T>();

      const framework::DDim& dim = in_tensor_ddx->dims();
      size_t N = static_cast<size_t>(framework::product(dim));
      auto step = dim[dim.size() - 1];
      int s = -1;
      bool new_s = false;
      for (size_t i = 0; i < N; ++i) {
        if (0 == i % step) {
          ++s;
          new_s = true;
        }
        if (new_s) {
          data_d_dout[s] =
              data_ddy[i] * data_d_dx[i] + data_ddx[i] * data_d_dy[i];
        } else {
          data_d_dout[s] +=
              data_ddy[i] * data_d_dx[i] + data_ddx[i] * data_d_dy[i];
        }
        new_s = false;
      }
    }

    if (out_tensor_d_ddx) {
      auto* data_d_ddx = out_tensor_d_ddx->mutable_data<T>(ctx.GetPlace());
      auto* data_dout = in_tensor_dout->data<T>();
      auto* data_d_dy = in_tensor_d_dy->data<T>();
      auto* data_y = in_tensor_y->data<T>();
      auto* data_d_ddout = in_tensor_d_ddout->data<T>();

      const framework::DDim& dim = out_tensor_d_ddx->dims();
      size_t N = static_cast<size_t>(framework::product(dim));
      auto step = dim[dim.size() - 1];
      int s = -1;

      for (size_t i = 0; i < N; ++i) {
        if (0 == i % step) ++s;
        data_d_ddx[i] =
            data_dout[s] * data_d_dy[i] + data_y[i] * data_d_ddout[s];
      }
    }

    if (out_tensor_d_ddy) {
      auto* data_d_ddy = out_tensor_d_ddy->mutable_data<T>(ctx.GetPlace());
      auto* data_dout = in_tensor_dout->data<T>();
      auto* data_d_dx = in_tensor_d_dx->data<T>();
      auto* data_x = in_tensor_x->data<T>();
      auto* data_d_ddout = in_tensor_d_ddout->data<T>();

      const framework::DDim& dim = out_tensor_d_ddy->dims();
      size_t N = static_cast<size_t>(framework::product(dim));
      auto step = dim[dim.size() - 1];
      int s = -1;

      for (size_t i = 0; i < N; ++i) {
        if (0 == i % step) ++s;
        data_d_ddy[i] =
            data_dout[s] * data_d_dx[i] + data_x[i] * data_d_ddout[s];
      }
    }
#endif
  }
};

template <typename DeviceContext, typename T>
class MatMulV2GradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    bool transpose_x = ctx.Attr<bool>("trans_x");
    bool transpose_y = ctx.Attr<bool>("trans_y");
    auto x = *ctx.Input<framework::Tensor>("X");
    auto y = *ctx.Input<framework::Tensor>("Y");
    auto dout = *ctx.Input<framework::Tensor>(framework::GradVarName("Out"));

    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));

    auto pt_x = paddle::experimental::MakePtenDenseTensor(*x);
    auto pt_y = paddle::experimental::MakePtenDenseTensor(*y);
    auto pt_dout = paddle::experimental::MakePtenDenseTensor(*dout);
    auto pt_dx = paddle::experimental::MakePtenDenseTensor(*dx);
    auto pt_dy = paddle::experimental::MakePtenDenseTensor(*dy);

    auto& dev_ctx = ctx.device_context<DeviceContext>();

    // call new kernel
    pten::MatmulGrad<T>(dev_ctx, *pt_x, *pt_y, *pt_dout, transpose_x,
                        transpose_y, pt_dx.get(), pt_dy.get());
  }
};

template <typename DeviceContext, typename T>
class MatMulV2DoubleGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto x = *context.Input<framework::Tensor>("X");
    auto y = *context.Input<framework::Tensor>("Y");
    auto dout = *context.Input<framework::Tensor>("DOut");
    auto* ddx = context.Input<framework::Tensor>("DDX");
    auto* ddy = context.Input<framework::Tensor>("DDY");

    auto* dx = context.Output<framework::Tensor>("DX");
    auto* dy = context.Output<framework::Tensor>("DY");
    auto* ddout = context.Output<framework::Tensor>("DDOut");

    bool transpose_x = context.Attr<bool>("trans_x");
    bool transpose_y = context.Attr<bool>("trans_y");

    auto& dev_ctx = ctx.device_context<DeviceContext>();

    auto pt_x = paddle::experimental::MakePtenDenseTensor(*x);
    auto pt_y = paddle::experimental::MakePtenDenseTensor(*y);
    auto pt_dout = paddle::experimental::MakePtenDenseTensor(*dout);
    auto pt_ddx = paddle::experimental::MakePtenDenseTensor(*ddx);
    auto pt_ddy = paddle::experimental::MakePtenDenseTensor(*ddy);
    auto pt_dx = paddle::experimental::MakePtenDenseTensor(*dx);
    auto pt_dy = paddle::experimental::MakePtenDenseTensor(*dy);
    auto pt_ddout = paddle::experimental::MakePtenDenseTensor(*ddout);

    // call new kernel
    pten::MatmulDoubleGrad<T>(dev_ctx, *pt_x.get(), *pt_y.get(), *pt_dout.get(),
                              *pt_ddx.get(), *pt_ddy.get(), transpose_x,
                              transpose_y, pt_dx.get(), pt_dy.get(),
                              pt_ddout.get());
  }
};

template <typename DeviceContext, typename T>
class MatMulV2TripleGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    // get input
    auto x = *context.Input<framework::Tensor>("X");
    auto y = *context.Input<framework::Tensor>("Y");
    auto dout = *context.Input<framework::Tensor>("DOut");
    auto ddx = *context.Input<framework::Tensor>("DDX");
    auto ddy = *context.Input<framework::Tensor>("DDY");

    auto* d_dx = context.Input<framework::Tensor>("D_DX");
    auto* d_dy = context.Input<framework::Tensor>("D_DY");
    auto* d_ddout = context.Input<framework::Tensor>("D_DDOut");

    // get output
    auto* out_d_x = context.Output<framework::Tensor>("D_X_out");
    auto* out_d_y = context.Output<framework::Tensor>("D_Y_out");
    auto* out_d_dout = context.Output<framework::Tensor>("D_DOut_out");

    auto* out_d_ddx = context.Output<framework::Tensor>("D_DDX_out");
    auto* out_d_ddy = context.Output<framework::Tensor>("D_DDY_out");

    bool transpose_x = context.Attr<bool>("trans_x");
    bool transpose_y = context.Attr<bool>("trans_y");

    auto pt_x = paddle::experimental::MakePtenDenseTensor(*x);
    auto pt_y = paddle::experimental::MakePtenDenseTensor(*y);
    auto pt_dout = paddle::experimental::MakePtenDenseTensor(*dout);
    auto pt_ddx = paddle::experimental::MakePtenDenseTensor(*ddx);
    auto pt_ddy = paddle::experimental::MakePtenDenseTensor(*ddy);
    auto pt_d_dx = paddle::experimental::MakePtenDenseTensor(*d_dx);
    auto pt_d_dy = paddle::experimental::MakePtenDenseTensor(*d_dy);
    auto pt_d_ddout = paddle::experimental::MakePtenDenseTensor(*d_ddout);

    auto pt_out_d_x = paddle::experimental::MakePtenDenseTensor(*out_d_x);
    auto pt_out_d_y = paddle::experimental::MakePtenDenseTensor(*out_d_y);
    auto pt_out_d_dout = paddle::experimental::MakePtenDenseTensor(*out_d_dout);
    auto pt_out_d_ddx = paddle::experimental::MakePtenDenseTensor(*out_d_ddx);
    auto pt_out_d_ddy = paddle::experimental::MakePtenDenseTensor(*out_d_ddy);

    auto& dev_ctx = ctx.device_context<DeviceContext>();
    // call new kernel
    pten::MatmulTripleGrad<T>(dev_ctx, *pt_x, *pt_y, *pt_dout, *pt_ddx, *pt_ddy,
                              *pt_d_dx, *pt_d_dy, *pt_d_ddout transpose_x,
                              transpose_y, pt_out_d_x.get(), pt_out_d_y.get(),
                              pt_out_d_dout.get(), pt_out_d_ddx.get(),
                              pt_out_d_ddy.get());
  }
};

}  // namespace operators
}  // namespace paddle
