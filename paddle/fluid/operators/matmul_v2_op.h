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
#include "paddle/pten/kernels/matmul_grad_kernel.h"
#include "paddle/pten/kernels/matmul_kernel.h"

#if defined(__NVCC__) || defined(__HIPCC__)
#include "paddle/fluid/operators/reduce_ops/reduce_op.cu.h"
#endif

namespace paddle {
namespace operators {

using framework::Tensor;

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

    // call new kernel
    pten::MatmulKernel<T>(dev_ctx, *X, *Y, trans_x, trans_y, Out);
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
class MatMulV2GradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    bool transpose_x = ctx.Attr<bool>("trans_x");
    bool transpose_y = ctx.Attr<bool>("trans_y");
    auto* x = ctx.Input<framework::Tensor>("X");
    auto* y = ctx.Input<framework::Tensor>("Y");
    auto* dout = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));

    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));

    if (dx) dx->mutable_data<T>(ctx.GetPlace());
    if (dy) dy->mutable_data<T>(ctx.GetPlace());

    auto& dev_ctx = ctx.device_context<DeviceContext>();

    // call new kernel
    pten::MatmulGradKernel<T>(dev_ctx, *x, *y, *dout, transpose_x, transpose_y,
                              dx, dy);
  }
};

template <typename DeviceContext, typename T>
class MatMulV2DoubleGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<framework::Tensor>("X");
    auto* y = context.Input<framework::Tensor>("Y");
    auto* dout = context.Input<framework::Tensor>("DOut");
    auto* ddx = context.Input<framework::Tensor>("DDX");
    auto* ddy = context.Input<framework::Tensor>("DDY");

    auto* dx = context.Output<framework::Tensor>("DX");
    auto* dy = context.Output<framework::Tensor>("DY");
    auto* ddout = context.Output<framework::Tensor>("DDOut");

    bool transpose_x = context.Attr<bool>("trans_x");
    bool transpose_y = context.Attr<bool>("trans_y");

    if (dx) dx->mutable_data<T>(context.GetPlace());
    if (dy) dy->mutable_data<T>(context.GetPlace());
    if (ddout) ddout->mutable_data<T>(context.GetPlace());

    auto& dev_ctx = context.device_context<DeviceContext>();

    // call new kernel
    pten::MatmulDoubleGradKernel<T>(dev_ctx, *x, *y, *dout, *ddx, *ddy,
                                    transpose_x, transpose_y, dx, dy, ddout);
  }
};

template <typename DeviceContext, typename T>
class MatMulV2TripleGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    // get input
    auto* x = context.Input<framework::Tensor>("X");
    auto* y = context.Input<framework::Tensor>("Y");
    auto* dout = context.Input<framework::Tensor>("DOut");
    auto* ddx = context.Input<framework::Tensor>("DDX");
    auto* ddy = context.Input<framework::Tensor>("DDY");

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

    if (out_d_x) out_d_x->mutable_data<T>(context.GetPlace());
    if (out_d_y) out_d_y->mutable_data<T>(context.GetPlace());
    if (out_d_dout) out_d_dout->mutable_data<T>(context.GetPlace());
    if (out_d_ddx) out_d_ddx->mutable_data<T>(context.GetPlace());
    if (out_d_ddy) out_d_ddy->mutable_data<T>(context.GetPlace());

    auto& dev_ctx = context.device_context<DeviceContext>();
    // call new kernel
    pten::MatmulTripleGradKernel<T>(
        dev_ctx, *x, *y, *dout, *ddx, *ddy, *d_dx, *d_dy, *d_ddout, transpose_x,
        transpose_y, out_d_x, out_d_y, out_d_dout, out_d_ddx, out_d_ddy);
  }
};

}  // namespace operators
}  // namespace paddle
