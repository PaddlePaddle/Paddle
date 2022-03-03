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

#ifdef PADDLE_WITH_XPU

#include <algorithm>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/xpu_api_wrapper.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"

namespace paddle {
namespace operators {
using framework::Tensor;

static framework::DDim RowMatrixFromVector(const framework::DDim &x_dim) {
  if (x_dim.size() > 1) {
    return x_dim;
  }
  return phi::make_ddim({1, x_dim[0]});
}

static framework::Tensor FoldInitDims(const framework::Tensor &input) {
  auto output = input;
  auto in_dims = input.dims();
  if (in_dims.size() == 3) {
    output.Resize({in_dims[0] * in_dims[1], in_dims[2]});
  }
  return output;
}
/**
 * Get column matrix shape from a vector shape. If the ran of y_dim > 1, the
 * original y_dim is returned.
 */
static framework::DDim ColumnMatrixFromVector(const framework::DDim &y_dim) {
  if (y_dim.size() > 1) {
    return y_dim;
  }
  return phi::make_ddim({y_dim[0], 1});
}

static void ReshapeTensorIntoMatrixSequence(
    framework::Tensor *x, const phi::funcs::MatDescriptor &descriptor) {
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
/**
 * Reshape the x,y,out tensor to 3-D or 2-D tensor by matrix descriptor
 * Out = matmul(x, y)
 *
 * This method will first calculate X,Y matrix sequence, and then calculate
 * the out shape.
 *
 * Assume X = [BatchSize, H1, W1], Y = [BatchSize, H2, W2]
 * The out = [BatchSize, H1, W2]
 *
 * If there is no batch size in `X` and `Y`, the out will be [H1, W2]
 * If any of `X` and `Y` has batch size BatchSize, the out will have the
 * BatchSize.
 */
static void ReshapeXYOutIntoMatrixSequence(framework::Tensor *x,
                                           framework::Tensor *y,
                                           framework::Tensor *out, bool trans_x,
                                           bool trans_y) {
  auto x_dim = RowMatrixFromVector(x->dims());
  auto y_dim = ColumnMatrixFromVector(y->dims());
  auto mat_dim_x = phi::funcs::CreateMatrixDescriptor(x_dim, 0, trans_x);
  auto mat_dim_y = phi::funcs::CreateMatrixDescriptor(y_dim, 0, trans_y);
  if (mat_dim_x.batch_size_ == 0 && mat_dim_y.batch_size_ == 0) {
    out->Resize({mat_dim_x.height_, mat_dim_y.width_});
  } else {
    out->Resize({std::max(mat_dim_x.batch_size_, mat_dim_y.batch_size_),
                 mat_dim_x.height_, mat_dim_y.width_});
  }

  ReshapeTensorIntoMatrixSequence(x, mat_dim_x);
  ReshapeTensorIntoMatrixSequence(y, mat_dim_y);
}

template <typename T, typename FCT>
static void MatMulXPUFunction(const Tensor *x, const Tensor *y, Tensor *out,
                              bool trans_x, bool trans_y,
                              const paddle::framework::ExecutionContext &ctx) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  const auto &x_dims = x->dims();
  const auto &y_dims = y->dims();
  auto &dev_ctx =
      ctx.template device_context<paddle::platform::XPUDeviceContext>();

  auto mat_dim_a = phi::funcs::CreateMatrixDescriptor(
      RowMatrixFromVector(x_dims), 0, trans_x);
  auto mat_dim_b = phi::funcs::CreateMatrixDescriptor(
      ColumnMatrixFromVector(y_dims), 0, trans_y);

  if (x_dims.size() == 3 && y_dims.size() <= 2) {
    // if transpose_X is true, the transpose cost much time
    if (!trans_x) {
      mat_dim_a.height_ *= mat_dim_a.batch_size_;
      mat_dim_a.batch_size_ = 0;
    } else {
      mat_dim_b.batch_size_ = mat_dim_a.batch_size_;
      mat_dim_b.height_ = mat_dim_b.height_ / mat_dim_b.batch_size_;
    }
  }

  if (mat_dim_a.width_ == mat_dim_b.height_) {
    if (mat_dim_a.batch_size_ == 0 && mat_dim_b.batch_size_ == 1) {
      mat_dim_a.batch_size_ = mat_dim_b.batch_size_ = 0;
    }
    if (mat_dim_a.batch_size_ == 1 && mat_dim_b.batch_size_ == 0) {
      mat_dim_a.batch_size_ = mat_dim_b.batch_size_ = 0;
    }
  }

  PADDLE_ENFORCE_EQ(mat_dim_a.width_, mat_dim_b.height_,
                    platform::errors::InvalidArgument(
                        "Shape mistake in matmul_op, the "
                        "first tensor width must be same as "
                        "second tensor height, but received "
                        "width:%d, height:%d x_dims = %s , y_dims = %s",
                        mat_dim_a.width_, mat_dim_b.height_,
                        x_dims.to_str().c_str(), y_dims.to_str().c_str()));
  PADDLE_ENFORCE_EQ(mat_dim_a.batch_size_, mat_dim_b.batch_size_,
                    platform::errors::InvalidArgument(
                        "Shape mistake in matmul_op, the two input"
                        "tensor batch_size must be same, but received first "
                        "tensor batch_size:%d, second "
                        "tensor batch_size:%d, x_dims = %s , y_dims = %s",
                        mat_dim_a.batch_size_, mat_dim_b.batch_size_,
                        x_dims.to_str().c_str(), y_dims.to_str().c_str()));

  float alpha = static_cast<T>(ctx.Attr<float>("alpha"));
  T *data_c = out->data<T>();
  int m = mat_dim_a.height_;
  int n = mat_dim_b.width_;
  int k = mat_dim_a.width_;
  int batch_size = mat_dim_a.batch_size_;
  int ldx = mat_dim_a.trans_ ? m : k;
  int ldy = mat_dim_b.trans_ ? k : n;
  int ldout = n;
  if (batch_size <= 1) {
    int r = 0;
    r = xpu_fc_wrapper<XPUType, FCT>(
        dev_ctx.x_context(), reinterpret_cast<const XPUType *>(x->data<T>()),
        reinterpret_cast<const XPUType *>(y->data<T>()),
        reinterpret_cast<XPUType *>(data_c), m, n, k, mat_dim_a.trans_,
        mat_dim_b.trans_, nullptr, nullptr, nullptr, ldx, ldy, ldout, alpha, 0,
        nullptr, xpu::Activation_t::LINEAR);
    PADDLE_ENFORCE_EQ(
        r, XPU_SUCCESS,
        platform::errors::External("XPU fc kernel return wrong value[%d %s]", r,
                                   XPUAPIErrorMsg[r]));
  } else {
    // batch matmul
    int r = xpu::fc_batched<XPUType, XPUType, XPUType, FCT>(
        dev_ctx.x_context(),                              // Context* ctx,
        batch_size,                                       // int batch_size,
        mat_dim_a.trans_,                                 // bool x_trans,
        mat_dim_b.trans_,                                 // bool w_trans,
        m,                                                // int m,
        n,                                                // int n,
        k,                                                // int k,
        alpha,                                            // float alpha,
        reinterpret_cast<const XPUType *>(x->data<T>()),  // const TX* x,
        mat_dim_a.stride_,                                // int stride_a,
        reinterpret_cast<const XPUType *>(y->data<T>()),  // const TW* w,
        mat_dim_b.stride_,                                // int stride_b,
        0.0,                                              // float beta,
        reinterpret_cast<XPUType *>(data_c),              // TY* y,
        m * n,                                            // int stride_c,
        nullptr,   // const float* x_maxptr,
        nullptr);  // const float* w_maxptr

    PADDLE_ENFORCE_EQ(r, XPU_SUCCESS,
                      platform::errors::External(
                          "XPU fc_batched kernel return wrong value[%d %s] "
                          "x_dims = %s , y_dims = %s",
                          r, XPUAPIErrorMsg[r], x_dims.to_str().c_str(),
                          y_dims.to_str().c_str()));
  }
}

template <typename DeviceContext, typename T>
class MatMulXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *x = context.Input<framework::Tensor>("X");
    auto *y = context.Input<framework::Tensor>("Y");
    auto *out = context.Output<framework::Tensor>("Out");
    out->mutable_data<T>(context.GetPlace());
    bool trans_x = context.Attr<bool>("transpose_X");
    bool trans_y = context.Attr<bool>("transpose_Y");
    if (std::is_same<paddle::platform::float16, T>::value) {
      MatMulXPUFunction<T, int16_t>(x, y, out, trans_x, trans_y, context);
    } else {
      if (std::getenv("XPU_PADDLE_FC_INT32") != nullptr) {
        MatMulXPUFunction<T, int32_t>(x, y, out, trans_x, trans_y, context);
      } else if (std::getenv("XPU_PADDLE_FC_LOCAL_INT16") != nullptr) {
        MatMulXPUFunction<T, float>(x, y, out, trans_x, trans_y, context);
      } else {
        MatMulXPUFunction<T, int16_t>(x, y, out, trans_x, trans_y, context);
      }
    }
  }
};

// Reshape a rank-3 tensor from P x M x N to M x (P * N).
// (Warning: This requires transposing data and writes into new memory.)
// Identity op if the tensor is not of rank 3.
template <typename DeviceContext, typename T>
static framework::Tensor XPUFoldHeadAndLastDims(
    const DeviceContext &context, const framework::Tensor &input) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  auto in_dims = input.dims();
  if (in_dims.size() != 3) {
    return input;
  }

  framework::Tensor output;
  output.Resize({in_dims[1], in_dims[0], in_dims[2]});
  output.mutable_data<T>(context.GetPlace());
  std::vector<int> in_shape_host = {static_cast<int>(in_dims[0]),
                                    static_cast<int>(in_dims[1]),
                                    static_cast<int>(in_dims[2])};
  std::vector<int> axis_host = {1, 0, 2};
  int r = xpu::transpose(
      context.x_context(), reinterpret_cast<const XPUType *>(input.data<T>()),
      reinterpret_cast<XPUType *>(output.data<T>()), in_shape_host, axis_host);
  PADDLE_ENFORCE_EQ(r, XPU_SUCCESS,
                    platform::errors::External(
                        "XPU transpose kernel return wrong value[%d %s]", r,
                        XPUAPIErrorMsg[r]));
  output.Resize({in_dims[1], in_dims[0] * in_dims[2]});

  return output;
}

// Using dimensional constraints on matrix multiplication, it is
// straight-forward to check the following table for when X and Y
// are both matrices.
//
// transpose_X | False    | True     | False    | True
// transpose_Y | False    | False    | True     | True
// -----------+----------+----------+----------+-----------
//        dX = | dOut Y^T | Y dOut^T | dOut Y   | Y^T dOut^T
//        dY = | X^T dOut | X dOut   | dOut^T X | dOut^T X^T
//
// When X is a vector of size K, we treat it instead as a matrix of shape
// (1, K). Similarly, when Y is a vector of size K, we treat it instead as
// a matrix of shape (K, 1).
//
// When X and Y are both 3-dimensional tensors, then the first dimension
// the batch dimension can be ignored and the exact same formulas apply
// as for two matrices.
//
// Finally, when, e.g., X is a 3-dimensional tensor but Y is a matrix, we end
// up with formulas like
//
//   dY_{ij} = \sum_{p, m} X_{pmi} dOut_{pmj}
//
// To handle this sort of scenario, we reshape X : P x M x K, dOut: P x M x N
// to X: (P * M) x K, dOut: (P * M) x N.
template <typename DeviceContext, typename T>
class MatMulGradXPUKernel : public framework::OpKernel<T> {
 public:
  void MatMul(const framework::ExecutionContext &context,
              const framework::Tensor &a, bool trans_a,
              const framework::Tensor &b, bool trans_b,
              framework::Tensor *out) const {
    out->mutable_data<T>(context.GetPlace());
    if (std::is_same<paddle::platform::float16, T>::value) {
      MatMulXPUFunction<T, int16_t>(&a, &b, out, trans_a, trans_b, context);
    } else {
      if (std::getenv("XPU_PADDLE_FC_INT32") != nullptr) {
        MatMulXPUFunction<T, int32_t>(&a, &b, out, trans_a, trans_b, context);
      } else if (std::getenv("XPU_PADDLE_FC_LOCAL_INT16") != nullptr) {
        MatMulXPUFunction<T, float>(&a, &b, out, trans_a, trans_b, context);
      } else {
        MatMulXPUFunction<T, int16_t>(&a, &b, out, trans_a, trans_b, context);
      }
    }
  }

  void CalcInputGrad(const framework::ExecutionContext &context,
                     const framework::Tensor &a, bool trans_a,
                     bool is_fold_init_dims_a, const framework::Tensor &b,
                     bool trans_b, bool is_fold_init_dims_b,
                     framework::Tensor *out) const {
    if (out == nullptr) return;
    bool need_combine = (a.dims().size() == 3 || b.dims().size() == 3) &&
                        out->dims().size() == 2;
    if (!need_combine) {
      MatMul(context, a, trans_a, b, trans_b, out);
    } else {
      auto &dev_ctx = context.template device_context<DeviceContext>();
      MatMul(
          context, is_fold_init_dims_a
                       ? FoldInitDims(a)
                       : XPUFoldHeadAndLastDims<DeviceContext, T>(dev_ctx, a),
          trans_a, is_fold_init_dims_b
                       ? FoldInitDims(b)
                       : XPUFoldHeadAndLastDims<DeviceContext, T>(dev_ctx, b),
          trans_b, out);
    }
  }

  void Compute(const framework::ExecutionContext &context) const override {
    auto x = *context.Input<framework::Tensor>("X");
    auto y = *context.Input<framework::Tensor>("Y");
    auto dout =
        *context.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto *dx = context.Output<framework::Tensor>(framework::GradVarName("X"));
    auto *dy = context.Output<framework::Tensor>(framework::GradVarName("Y"));
    bool transpose_x = context.Attr<bool>("transpose_X");
    bool transpose_y = context.Attr<bool>("transpose_Y");

    ReshapeXYOutIntoMatrixSequence(&x, &y, &dout, transpose_x, transpose_y);

    framework::DDim dx_dims;
    if (dx) {
      dx_dims = dx->dims();
      if (dx_dims != x.dims()) {
        dx->Resize(x.dims());
      }
    }

    framework::DDim dy_dims;
    if (dy) {
      dy_dims = dy->dims();
      if (dy_dims != y.dims()) {
        dy->Resize(y.dims());
      }
    }

    if (transpose_x && transpose_y) {
      CalcInputGrad(context, y, true, true, dout, true, false, dx);
      CalcInputGrad(context, dout, true, true, x, true, false, dy);
    } else if (transpose_x) {
      CalcInputGrad(context, y, false, false, dout, true, false, dx);
      CalcInputGrad(context, x, false, false, dout, false, true, dy);
    } else if (transpose_y) {
      CalcInputGrad(context, dout, false, false, y, false, true, dx);
      CalcInputGrad(context, dout, true, true, x, false, true, dy);
    } else {
      CalcInputGrad(context, dout, false, false, y, true, false, dx);
      CalcInputGrad(context, x, true, true, dout, false, true, dy);
    }

    if (dx) {
      if (dx_dims != x.dims()) {
        dx->Resize(dx_dims);
      }
    }

    if (dy) {
      if (dy_dims != y.dims()) {
        dy->Resize(dy_dims);
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_XPU_KERNEL(
    matmul, ops::MatMulXPUKernel<paddle::platform::XPUDeviceContext, float>,
    ops::MatMulXPUKernel<paddle::platform::XPUDeviceContext, plat::float16>);
REGISTER_OP_XPU_KERNEL(
    matmul_grad,
    ops::MatMulGradXPUKernel<paddle::platform::XPUDeviceContext, float>,
    ops::MatMulGradXPUKernel<paddle::platform::XPUDeviceContext,
                             plat::float16>);
#endif
