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

#include <algorithm>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

namespace paddle {
namespace operators {

/**
 * Printing shape information into a string is easy to use.
 */
inline static std::string DumpMatrixShape(
    const phi::funcs::MatDescriptor &desc) {
  std::stringstream buffer;
  buffer << "[" << desc.batch_size_ << ", " << desc.height_ << ", "
         << desc.width_ << "]";
  return buffer.str();
}

/**
 * Get row matrix shape from a vector shape. If the rank of x_dim > 1, the
 * original x_dim is returned.
 */
static framework::DDim RowMatrixFromVector(const framework::DDim &x_dim) {
  if (x_dim.size() > 1) {
    return x_dim;
  }
  return phi::make_ddim({1, x_dim[0]});
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

template <typename DeviceContext, typename T>
class MatMulKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto &x = GET_DATA_SAFELY(
        context.Input<phi::DenseTensor>("X"), "Input", "X", "MatMul");
    auto &y = GET_DATA_SAFELY(
        context.Input<phi::DenseTensor>("Y"), "Input", "Y", "MatMul");
    auto *out = context.Output<phi::DenseTensor>("Out");

    auto &dev_ctx = context.template device_context<DeviceContext>();
    dev_ctx.template Alloc<T>(out, out->numel() * sizeof(T));

    auto blas = phi::funcs::GetBlas<DeviceContext, T>(context);
    auto mat_dim_a = phi::funcs::CreateMatrixDescriptor(
        RowMatrixFromVector(x.dims()), 0, context.Attr<bool>("transpose_X"));
    auto mat_dim_b = phi::funcs::CreateMatrixDescriptor(
        ColumnMatrixFromVector(y.dims()), 0, context.Attr<bool>("transpose_Y"));
    auto scale = static_cast<T>(context.Attr<float>("alpha"));

    int head_number = 1;
#if defined(PADDLE_WITH_MKLML) && !defined(PADDLE_WITH_CUDA) && \
    !defined(PADDLE_WITH_HIP)
    head_number = context.Attr<int>("head_number");
#endif

    const auto &x_dims = x.dims();
    const auto &y_dims = y.dims();
    if (head_number <= 1 && x_dims.size() == 3 && y_dims.size() <= 2) {
      // the transpose_X must be false, if is true, the transpose cost much time
      if (!context.Attr<bool>("transpose_X")) {
        mat_dim_a.height_ *= mat_dim_a.batch_size_;
        mat_dim_a.batch_size_ = 0;
      }
    }
#if defined(PADDLE_WITH_MKLML) && !defined(PADDLE_WITH_CUDA) && \
    !defined(PADDLE_WITH_HIP)
    bool split_vertical_y = (mat_dim_a.width_ != mat_dim_b.height_);

    if (head_number > 1) {
      blas.MatMulWithHead(x,
                          mat_dim_a,
                          y,
                          mat_dim_b,
                          scale,
                          head_number,
                          out,
                          T(0),
                          split_vertical_y);
    } else {
      blas.MatMul(x, mat_dim_a, y, mat_dim_b, scale, out, T(0));
    }
#else
    blas.MatMul(x, mat_dim_a, y, mat_dim_b, scale, out, T(0));
#endif
  }
};

// Reshape a rank-3 tensor from P x M x N to (P * M) x N.
// Identity op if the tensor is not of rank 3.
static phi::DenseTensor FoldInitDims(const phi::DenseTensor &input) {
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
static phi::DenseTensor FoldHeadAndLastDims(const DeviceContext &context,
                                            const phi::DenseTensor &input) {
  auto in_dims = input.dims();
  if (in_dims.size() != 3) {
    return input;
  }
  phi::DenseTensor output;
  output.Resize({in_dims[1], in_dims[0], in_dims[2]});
  output.mutable_data<T>(context.GetPlace());
  std::vector<int> axis = {1, 0, 2};
  phi::funcs::Transpose<DeviceContext, T, 3> trans;
  trans(context, input, &output, axis);
  output.Resize({in_dims[1], in_dims[0] * in_dims[2]});

  return output;
}

/**
 * Reshape a tensor to 3-D or 2-D tensor by matrix descriptor.
 *
 * The shape would be [BatchSize, H, W] or [H, W].
 * If transposed, `H,W` will be swapped.
 */
static void ReshapeTensorIntoMatrixSequence(
    phi::DenseTensor *x, const phi::funcs::MatDescriptor &descriptor) {
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
static void ReshapeXYOutIntoMatrixSequence(phi::DenseTensor *x,
                                           phi::DenseTensor *y,
                                           phi::DenseTensor *out,
                                           bool trans_x,
                                           bool trans_y) {
  auto x_dim = RowMatrixFromVector(x->dims());
  auto y_dim = ColumnMatrixFromVector(y->dims());
  auto mat_dim_x = phi::funcs::CreateMatrixDescriptor(x_dim, 0, trans_x);
  auto mat_dim_y = phi::funcs::CreateMatrixDescriptor(y_dim, 0, trans_y);
  if (mat_dim_x.batch_size_ == 0 && mat_dim_y.batch_size_ == 0) {
    out->Resize({mat_dim_x.height_, mat_dim_y.width_});
  } else {
    out->Resize({std::max(mat_dim_x.batch_size_, mat_dim_y.batch_size_),
                 mat_dim_x.height_,
                 mat_dim_y.width_});
  }

  ReshapeTensorIntoMatrixSequence(x, mat_dim_x);
  ReshapeTensorIntoMatrixSequence(y, mat_dim_y);
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
class MatMulGradKernel : public framework::OpKernel<T> {
 public:
  void MatMul(const framework::ExecutionContext &context,
              const phi::DenseTensor &a,
              bool trans_a,
              const phi::DenseTensor &b,
              bool trans_b,
              phi::DenseTensor *out) const {
    out->mutable_data<T>(context.GetPlace());
    auto blas = phi::funcs::GetBlas<DeviceContext, T>(context);
    auto mat_dim_a = phi::funcs::CreateMatrixDescriptor(a.dims(), 0, trans_a);
    auto mat_dim_b = phi::funcs::CreateMatrixDescriptor(b.dims(), 0, trans_b);

    int head_number = 1;
#if defined(PADDLE_WITH_MKLML) && !defined(PADDLE_WITH_CUDA) && \
    !defined(PADDLE_WITH_HIP)
    if (context.HasAttr("head_number")) {
      head_number = context.Attr<int>("head_number");
    }
#endif

    if (head_number <= 1 && a.dims().size() == 3 && b.dims().size() <= 2) {
      // the transpose_X must be false, if is true, the transpose cost much time
      if (!trans_a) {
        mat_dim_a.height_ *= mat_dim_a.batch_size_;
        mat_dim_a.batch_size_ = 0;
      }
    }
    blas.MatMul(a,
                mat_dim_a,
                b,
                mat_dim_b,
                static_cast<T>(context.Attr<float>("alpha")),
                out,
                T(0));
  }

  void CalcInputGrad(const framework::ExecutionContext &context,
                     const phi::DenseTensor &a,
                     bool trans_a,
                     bool is_fold_init_dims_a,
                     const phi::DenseTensor &b,
                     bool trans_b,
                     bool is_fold_init_dims_b,
                     phi::DenseTensor *out) const {
    if (out == nullptr) return;
    bool need_combine = (a.dims().size() == 3 || b.dims().size() == 3) &&
                        out->dims().size() == 2;
    if (!need_combine) {
      MatMul(context, a, trans_a, b, trans_b, out);
    } else {
      auto &ctx = context.template device_context<DeviceContext>();
      MatMul(
          context,
          is_fold_init_dims_a ? FoldInitDims(a)
                              : FoldHeadAndLastDims<DeviceContext, T>(ctx, a),
          trans_a,
          is_fold_init_dims_b ? FoldInitDims(b)
                              : FoldHeadAndLastDims<DeviceContext, T>(ctx, b),
          trans_b,
          out);
    }
  }

  void Compute(const framework::ExecutionContext &context) const override {
    auto x = *context.Input<phi::DenseTensor>("X");
    auto y = *context.Input<phi::DenseTensor>("Y");
    auto dout = *context.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto *dx = context.Output<phi::DenseTensor>(framework::GradVarName("X"));
    auto *dy = context.Output<phi::DenseTensor>(framework::GradVarName("Y"));
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

framework::DDim GetDimForInput(const framework::InferShapeContext &ctx,
                               std::string input_name) {
  auto shape = ctx.Attrs().Get<std::vector<int>>("fused_reshape_" + input_name);
  auto axis =
      ctx.Attrs().Get<std::vector<int>>("fused_transpose_" + input_name);
  auto dim = ctx.GetInputDim(input_name);

  PADDLE_ENFORCE_GT(dim.size(),
                    0,
                    platform::errors::InvalidArgument(
                        "The Input(%s) has not been initialized properly. The "
                        "shape of Input(%s) = [%s].",
                        dim));

  if (!shape.empty() && !axis.empty()) {
    dim = dim.reshape(shape).transpose(axis);
  }
  return dim;
}

template <typename DeviceContext, typename T>
class MatMulDoubleGradKernel : public framework::OpKernel<T> {
 public:
  void MatMul(const framework::ExecutionContext &context,
              const phi::DenseTensor &a,
              bool trans_a,
              const phi::DenseTensor &b,
              bool trans_b,
              bool flag,
              phi::DenseTensor *out) const {
    out->mutable_data<T>(context.GetPlace());
    auto blas = phi::funcs::GetBlas<DeviceContext, T>(context);
    auto mat_dim_a = phi::funcs::CreateMatrixDescriptor(a.dims(), 0, trans_a);
    auto mat_dim_b = phi::funcs::CreateMatrixDescriptor(b.dims(), 0, trans_b);

    int head_number = 1;
#if defined(PADDLE_WITH_MKLML) && !defined(PADDLE_WITH_CUDA) && \
    !defined(PADDLE_WITH_HIP)
    head_number = context.Attr<int>("head_number");
#endif

    if (head_number <= 1 && a.dims().size() == 3 && b.dims().size() <= 2) {
      // the transpose_X must be false, if is true, the transpose cost much time
      if (!trans_a) {
        mat_dim_a.height_ *= mat_dim_a.batch_size_;
        mat_dim_a.batch_size_ = 0;
      }
    }
    blas.MatMul(a,
                mat_dim_a,
                b,
                mat_dim_b,
                static_cast<T>(context.Attr<float>("alpha")),
                out,
                static_cast<T>(flag));
  }

  void CalcInputGrad(const framework::ExecutionContext &context,
                     const phi::DenseTensor &a,
                     bool trans_a,
                     bool is_fold_init_dims_a,
                     const phi::DenseTensor &b,
                     bool trans_b,
                     bool is_fold_init_dims_b,
                     bool flag,
                     phi::DenseTensor *out) const {
    if (out == nullptr) return;
    bool need_combine = (a.dims().size() == 3 || b.dims().size() == 3) &&
                        out->dims().size() == 2;
    if (!need_combine) {
      MatMul(context, a, trans_a, b, trans_b, flag, out);
    } else {
      auto &ctx = context.template device_context<DeviceContext>();
      MatMul(
          context,
          is_fold_init_dims_a ? FoldInitDims(a)
                              : FoldHeadAndLastDims<DeviceContext, T>(ctx, a),
          trans_a,
          is_fold_init_dims_b ? FoldInitDims(b)
                              : FoldHeadAndLastDims<DeviceContext, T>(ctx, b),
          trans_b,
          flag,
          out);
    }
  }

  void Compute(const framework::ExecutionContext &context) const override {
    auto x = *context.Input<phi::DenseTensor>("X");
    auto y = *context.Input<phi::DenseTensor>("Y");
    auto dout = *context.Input<framework::LoDTensor>("DOut");
    auto *ddx = context.Input<framework::LoDTensor>("DDX");
    auto *ddy = context.Input<framework::LoDTensor>("DDY");

    auto *dx = context.Output<framework::LoDTensor>("DX");
    auto *dy = context.Output<framework::LoDTensor>("DY");
    auto *ddout = context.Output<framework::LoDTensor>("DDOut");

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

    framework::DDim ddout_dims;
    if (ddout) {
      ddout_dims = ddout->dims();
      if (ddout_dims != dout.dims()) {
        ddout->Resize(dout.dims());
      }
    }

    bool ddout_flag = false;
    if (ddx) {
      auto ddx_mat = *ddx;
      if (ddx_mat.dims() != x.dims()) {
        ddx_mat.Resize(x.dims());
      }
      if (dy) {
        if (transpose_x && transpose_y) {
          // dy = dout' * ddx'
          CalcInputGrad(
              context, dout, true, true, ddx_mat, true, false, false, dy);
        } else if (transpose_x) {
          // dy = ddx * dout
          CalcInputGrad(
              context, ddx_mat, false, false, dout, false, true, false, dy);
        } else if (transpose_y) {
          // dy = dout' * ddx
          CalcInputGrad(
              context, dout, true, true, ddx_mat, false, true, false, dy);
        } else {
          // dy = ddx' * dout
          CalcInputGrad(
              context, ddx_mat, true, true, dout, false, true, false, dy);
        }
      }

      if (ddout) {
        CalcInputGrad(context,
                      ddx_mat,
                      transpose_x,
                      true,
                      y,
                      transpose_y,
                      false,
                      ddout_flag,
                      ddout);
        ddout_flag = true;
      }
    }

    if (ddy) {
      auto ddy_mat = *ddy;
      if (ddy_mat.dims() != y.dims()) {
        ddy_mat.Resize(y.dims());
      }
      if (dx) {
        if (transpose_x && transpose_y) {
          // dx = ddy' * dout'
          CalcInputGrad(
              context, ddy_mat, true, true, dout, true, false, false, dx);
        } else if (transpose_x) {
          // dx = ddy * dout'
          CalcInputGrad(
              context, ddy_mat, false, false, dout, true, false, false, dx);
        } else if (transpose_y) {
          // dx = dout * ddy
          CalcInputGrad(
              context, dout, false, false, ddy_mat, false, true, false, dx);
        } else {
          // dx = dout * ddy'
          CalcInputGrad(
              context, dout, false, false, ddy_mat, true, false, false, dx);
        }
      }

      if (ddout) {
        CalcInputGrad(context,
                      x,
                      transpose_x,
                      true,
                      ddy_mat,
                      transpose_y,
                      false,
                      ddout_flag,
                      ddout);
      }
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

    if (ddout) {
      if (ddout_dims != dout.dims()) {
        ddout->Resize(ddout_dims);
      }
    }
  }
};

class MatMulOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *context) const override {
    OP_INOUT_CHECK(context->HasInput("X"), "Input", "X", "matmul");
    OP_INOUT_CHECK(context->HasInput("Y"), "Input", "Y", "matmul");
    OP_INOUT_CHECK(context->HasOutput("Out"), "Output", "Out", "matmul");

    auto dim_x = GetDimForInput(*context, "X");
    auto dim_y = GetDimForInput(*context, "Y");

#ifdef PADDLE_WITH_MKLDNN
    // (jczaja): For NHWC execution output shape needs
    // to be computed like instead x*y we are to do y*x
    bool channelwise_onednn =
        context->IsRunMKLDNNKernel() &&
        (platform::MKLDNNDeviceContext::tls().get_cur_paddle_data_layout() ==
         framework::DataLayout::kNHWC);
    if (channelwise_onednn) {
      std::swap(dim_x, dim_y);
    }
#endif

    auto mat_dim_x = phi::funcs::CreateMatrixDescriptor(
        RowMatrixFromVector(dim_x),
        0,
        context->Attrs().Get<bool>("transpose_X"));
    auto mat_dim_y = phi::funcs::CreateMatrixDescriptor(
        ColumnMatrixFromVector(dim_y),
        0,
        context->Attrs().Get<bool>("transpose_Y"));

    if (mat_dim_x.width_ == -1) {
      mat_dim_x.width_ = mat_dim_y.height_;
    }
    if (mat_dim_y.height_ == -1) {
      mat_dim_y.height_ = mat_dim_x.width_;
    }

    if (context->IsRuntime()) {
      PADDLE_ENFORCE_EQ(
          mat_dim_x.batch_size_ == mat_dim_y.batch_size_ ||
              mat_dim_x.batch_size_ == 0 || mat_dim_y.batch_size_ == 0,
          true,
          platform::errors::InvalidArgument(
              "The batch size of the two matrices should be equal, or "
              "at least one is zero.\n"
              "But received X's shape: %s, Y's shape: %s.",
              DumpMatrixShape(mat_dim_x).c_str(),
              DumpMatrixShape(mat_dim_y).c_str()));
    }
    int64_t dim_out_y = mat_dim_y.width_;
#if defined(PADDLE_WITH_MKLML) && !defined(PADDLE_WITH_CUDA) && \
    !defined(PADDLE_WITH_HIP)
    int head_number = context->Attrs().Get<int>("head_number");
    bool split_vertical_y = (mat_dim_x.width_ != mat_dim_y.height_);
    if (context->IsRuntime()) {
      PADDLE_ENFORCE_LE(
          head_number,
          mat_dim_x.width_,
          platform::errors::InvalidArgument(
              "Unsatisfied mkl acceleration library requirements: "
              "The number of heads "
              "(%d) must be equal to X's width. But received X's shape: %s.",
              head_number,
              DumpMatrixShape(mat_dim_x).c_str()));

      if (!split_vertical_y && head_number > 0) {
        dim_out_y = head_number * mat_dim_y.width_;
      }
    }
#else
    PADDLE_ENFORCE_EQ(mat_dim_x.width_,
                      mat_dim_y.height_,
                      platform::errors::InvalidArgument(
                          "Input X's width should be equal to the Y's height, "
                          "but received X's shape: [%s], "
                          "Y's shape: [%s].",
                          dim_x,
                          dim_y));
#endif

    std::vector<int64_t> dim_out;
    if (mat_dim_x.batch_size_ != 0) {
      dim_out = phi::vectorize(dim_x);
      dim_out[dim_out.size() - 2] = mat_dim_x.height_;
      dim_out[dim_out.size() - 1] = dim_out_y;
    } else if (mat_dim_y.batch_size_ != 0) {
      dim_out = phi::vectorize(dim_y);
      dim_out[dim_out.size() - 2] = mat_dim_x.height_;
      dim_out[dim_out.size() - 1] = dim_out_y;
    } else {
      dim_out = {mat_dim_x.height_, dim_out_y};
    }

    if (dim_x.size() == 1 && dim_out[dim_out.size() - 2] == 1) {
      std::swap(dim_out[dim_out.size() - 2], dim_out[dim_out.size() - 1]);
      dim_out.resize(dim_out.size() - 1);
    }

    if (dim_y.size() == 1 && dim_out[dim_out.size() - 1] == 1) {
      dim_out.resize(dim_out.size() - 1);
    }

    if (dim_out.empty()) {
      dim_out = {1};
    }

    framework::DDim ddim_out = phi::make_ddim(dim_out);

#ifdef PADDLE_WITH_MKLDNN
    auto shape = context->Attrs().Get<std::vector<int>>("fused_reshape_Out");
    auto axis = context->Attrs().Get<std::vector<int>>("fused_transpose_Out");

    if (!shape.empty() && !axis.empty()) {
      ddim_out = ddim_out.transpose(axis).reshape(shape);
    }
#endif
    context->SetOutputDim("Out", ddim_out);
    context->ShareLoD("X", "Out");
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type =
        OperatorWithKernel::IndicateOrPromoteVarDataTypes(ctx, "X", "Y");

#ifdef PADDLE_WITH_MKLDNN
    if (this->CanMKLDNNBeUsed(ctx, input_data_type)) {
      return framework::OpKernelType(input_data_type,
                                     ctx.GetPlace(),
                                     framework::DataLayout::kMKLDNN,
                                     framework::LibraryType::kMKLDNN);
    }
#endif
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string &var_name,
      const phi::DenseTensor &tensor,
      const framework::OpKernelType &expected_kernel_type) const override {
    if (framework::IsComplexType(expected_kernel_type.data_type_)) {
      // only promote inputs’s types when contains complex input
      return framework::OpKernelType(
          framework::TransToProtoVarType(tensor.dtype()),
          tensor.place(),
          tensor.layout());
    } else {
#ifdef PADDLE_WITH_MKLDNN
      // When matmul is first oneDNN op in a chain (there was some non oneDNN op
      // previously)
      // then we also need to rotate shape NHWC -> NCWH
      if ((expected_kernel_type.data_layout_ ==
           framework::DataLayout::kMKLDNN) &&
          (tensor.layout() != framework::DataLayout::kMKLDNN) &&
          paddle::platform::MKLDNNDeviceContext::tls()
                  .get_cur_paddle_data_layout() ==
              framework::DataLayout::kNHWC) {
        return framework::OpKernelType(expected_kernel_type.data_type_,
                                       tensor.place(),
                                       framework::DataLayout::kNHWC);
      }
#endif
      return framework::OpKernelType(
          expected_kernel_type.data_type_, tensor.place(), tensor.layout());
    }
  }
};

class MatMulOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The first input of MatMul op");
    AddInput("Y", "The second input of MatMul op");
    AddOutput("Out", "The output of MatMul op");
    AddAttr<bool>("transpose_X",
                  R"DOC(If true, use the transpose of `X`.
        )DOC")
        .SetDefault(false);
    AddAttr<bool>("transpose_Y",
                  R"DOC(If true, use the transpose of `Y`.
        )DOC")
        .SetDefault(false);
    AddAttr<float>("alpha", "The scale of Out").SetDefault(1.0f);
    AddAttr<bool>(
        "use_mkldnn",
        "(bool, default false) Indicates if MKL-DNN kernel will be used")
        .SetDefault(false)
        .AsExtra();
    AddAttr<std::vector<int>>("fused_reshape_X",
                              R"DOC(Shape of fused reshape of `X` input.)DOC")
        .SetDefault({})
        .AsExtra();
    AddAttr<std::vector<int>>("fused_reshape_Y",
                              R"DOC(Shape of fused reshape of `Y` input.)DOC")
        .SetDefault({})
        .AsExtra();
    AddAttr<std::vector<int>>("fused_transpose_X",
                              R"DOC(Axis of fused transpose of `X` input.)DOC")
        .SetDefault({})
        .AsExtra();
    AddAttr<std::vector<int>>("fused_transpose_Y",
                              R"DOC(Axis of fused transpose of `Y` input.)DOC")
        .SetDefault({})
        .AsExtra();
    AddAttr<std::vector<int>>(
        "fused_reshape_Out",
        R"DOC(When MKLDNN MatMul_transpose_reshape fuse activated, "
              "it's a shape atribute of fused reshape for `Out` output.)DOC")
        .SetDefault({})
        .AsExtra();
    AddAttr<std::vector<int>>(
        "fused_transpose_Out",
        R"DOC(When MKLDNN MatMul_transpose_reshape fuse activated, "
              "it's a axis atribute of fused transpose for `Out` output.)DOC")
        .SetDefault({})
        .AsExtra();
    AddAttr<bool>(
        "use_quantizer",
        "(bool, default false) "
        "This parameter is no longer used. Use 'mkldnn_data_type' instead.")
        .SetDefault(false)
        .AsExtra();
    AddAttr<std::string>(
        "mkldnn_data_type",
        "(string, default \"float32\"). Data type of mkldnn kernel")
        .SetDefault("float32")
        .InEnum({"float32", "int8", "bfloat16"})
        .AsExtra();
    /* int8 parameters */
    AddAttr<float>("Scale_x",
                   "(float, default 1.0f), The quantize scale of X tensor")
        .SetDefault(1.0f)
        .AsExtra();
    AddAttr<float>("Scale_y",
                   "(float, default 1.0f), The quantize scale of Y tensor")
        .SetDefault(1.0f)
        .AsExtra();
    AddAttr<float>("Scale_out",
                   "(float, default 1.0f), The quantize scale of output data")
        .SetDefault(1.0f)
        .AsExtra();
    AddAttr<bool>("force_fp32_output",
                  "(bool, default false) Force INT8 kernel output FP32, only "
                  "used in MKL-DNN INT8")
        .SetDefault(false)
        .AsExtra();

#if defined(PADDLE_WITH_MKLML) && !defined(PADDLE_WITH_CUDA) && \
    !defined(PADDLE_WITH_HIP)
    AddAttr<int>("head_number", "The number of heads of the matrix")
        .SetDefault(1);
#endif
    AddComment(R"DOC(
MatMul Operator.
This operator is used to perform (batched) matrix multiplication
over the last two dimensions of the input tensors `X` and `Y`.
If a transpose flag is specified, the last two dimensions of the
tensor are transposed. If the tensor is rank-1 of shape [D], then
for `X` it is treated as [1, D] in nontransposed form and as [D, 1]
in transposed form, whereas for `Y` it is the opposite: It is treated
as [D, 1] in nontransposed form and as [1, D] in transposed form.
Examples without transpose:
- X: [K], Y: [K] => Out: [1]
- X: [K], Y: [K, N] => Out: [N]
- X: [B, M, K], Y: [K] => Out: [B, M]
- X: [M, K], Y: [B, K, N] => Out: [B, M, N]
- X: [B, M, K], Y: [B, K, N] => Out: [B, M, N]
- X: [B, ..., M, K], Y: [B, ..., K, N] => Out: [B, ..., M, N]
Example of matrix multiplication with head_number of H
- X: [B, M, K], Y: [B, K, N] => Out: [B, M, H * N]
The behavior is designed to be similar to the `numpy.matmul` function.
The differences are:
- When the rank of the input data is less than or equal to 3, it
  is similar to the `numpy.matmul` function.
- When the rank of the input is greater than 3, the rank of X and
  Y must be equal, and the first `rank - 2` dimensions must be equal.
- We add `transpose_X` and `transpose_Y` flags.
- We add `head_number` attribute, which is used to multiple two matrixes head
  by head, and eventually concatenates the output of several (head_number)
  small matrixes multiplication.
Both the input `X` and `Y` can carry the LoD (Level of Details) information,
or not. But the output only shares the LoD information with input `X`.
)DOC");
  }
};

class MatMulOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *context) const override {
    OP_INOUT_CHECK(context->HasInput("X"), "Input", "X", "matmul");
    OP_INOUT_CHECK(context->HasInput("Y"), "Input", "Y", "matmul");
    OP_INOUT_CHECK(context->HasInput(framework::GradVarName("Out")),
                   "Input",
                   "Out@GRAD",
                   "matmul");
    auto x_dims = context->GetInputDim("X");
    auto y_dims = context->GetInputDim("Y");

    auto x_grad_name = framework::GradVarName("X");
    auto y_grad_name = framework::GradVarName("Y");

    if (context->HasOutput(x_grad_name)) {
      context->SetOutputDim(x_grad_name, x_dims);
    }
    if (context->HasOutput(y_grad_name)) {
      context->SetOutputDim(y_grad_name, y_dims);
    }
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type =
        OperatorWithKernel::IndicateOrPromoteVarDataTypes(ctx, "X", "Y");

#ifdef PADDLE_WITH_MKLDNN
    if (this->CanMKLDNNBeUsed(ctx, input_data_type)) {
      return framework::OpKernelType(input_data_type,
                                     ctx.GetPlace(),
                                     framework::DataLayout::kMKLDNN,
                                     framework::LibraryType::kMKLDNN);
    }
#endif
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
};

template <typename T>
class MatMulOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("matmul_grad");
    retv->SetInput("X", this->Input("X"));
    retv->SetInput("Y", this->Input("Y"));
    retv->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    retv->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    retv->SetOutput(framework::GradVarName("Y"), this->InputGrad("Y"));
    retv->SetAttrMap(this->Attrs());
  }
};

class MatMulOpDoubleGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *context) const override {
    OP_INOUT_CHECK(context->HasInput("X"), "Input", "X", "matmul");
    OP_INOUT_CHECK(context->HasInput("Y"), "Input", "Y", "matmul");
    OP_INOUT_CHECK(context->HasInput("DOut"), "Input", "DOut", "matmul");

    if (context->HasOutput("DX") && context->HasInput("DDY")) {
      context->ShareDim("X", "DX");
    }

    if (context->HasOutput("DY") && context->HasInput("DDX")) {
      context->ShareDim("Y", "DY");
    }

    if (context->HasOutput("DDOut") &&
        (context->HasInput("DDY") || context->HasInput("DDX"))) {
      context->ShareDim("DOut", "DDOut");
    }
  }
};

template <typename T>
class MatMulOpDoubleGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("matmul_grad_grad");
    retv->SetInput("X", this->Input("X"));
    retv->SetInput("Y", this->Input("Y"));
    retv->SetInput("DOut", this->Input(framework::GradVarName("Out")));
    retv->SetInput("DDX", this->OutputGrad(framework::GradVarName("X")));
    retv->SetInput("DDY", this->OutputGrad(framework::GradVarName("Y")));

    auto ddx = this->OutputGrad(framework::GradVarName("X"));
    auto ddy = this->OutputGrad(framework::GradVarName("Y"));

    if (!ddx.empty() || !ddy.empty()) {
      retv->SetOutput("DDOut", this->InputGrad(framework::GradVarName("Out")));
    }
    retv->SetOutput(
        "DX", ddy.empty() ? this->EmptyInputGrad() : this->InputGrad("X"));
    retv->SetOutput(
        "DY", ddx.empty() ? this->EmptyInputGrad() : this->InputGrad("Y"));

    retv->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(matmul,
                  ops::MatMulOp,
                  ops::MatMulOpMaker,
                  ops::MatMulOpGradMaker<paddle::framework::OpDesc>,
                  ops::MatMulOpGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(matmul_grad,
                  ops::MatMulOpGrad,
                  ops::MatMulOpDoubleGradMaker<paddle::framework::OpDesc>,
                  ops::MatMulOpDoubleGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(matmul_grad_grad, ops::MatMulOpDoubleGrad);
REGISTER_OP_CPU_KERNEL(matmul,
                       ops::MatMulKernel<phi::CPUContext, float>,
                       ops::MatMulKernel<phi::CPUContext, double>);
REGISTER_OP_CPU_KERNEL(matmul_grad,
                       ops::MatMulGradKernel<phi::CPUContext, float>,
                       ops::MatMulGradKernel<phi::CPUContext, double>);

REGISTER_OP_CPU_KERNEL(matmul_grad_grad,
                       ops::MatMulDoubleGradKernel<phi::CPUContext, float>,
                       ops::MatMulDoubleGradKernel<phi::CPUContext, double>);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
REGISTER_OP_CUDA_KERNEL(
    matmul,
    ops::MatMulKernel<phi::GPUContext, float>,
    ops::MatMulKernel<phi::GPUContext, double>,
    ops::MatMulKernel<phi::GPUContext, paddle::platform::float16>);
REGISTER_OP_CUDA_KERNEL(
    matmul_grad,
    ops::MatMulGradKernel<phi::GPUContext, float>,
    ops::MatMulGradKernel<phi::GPUContext, double>,
    ops::MatMulGradKernel<phi::GPUContext, paddle::platform::float16>);
REGISTER_OP_CUDA_KERNEL(matmul_grad_grad,
                        ops::MatMulDoubleGradKernel<phi::GPUContext, float>,
                        ops::MatMulDoubleGradKernel<phi::GPUContext, double>);
#endif

REGISTER_OP_VERSION(matmul).AddCheckpoint(
    R"ROC(Register matmul for adding the attribute of
       fused_reshape_Y)ROC",
    paddle::framework::compatible::OpVersionDesc().NewAttr(
        "fused_reshape_Y",
        "In order to support the function of fused the input Y "
        " and input X into the input X when "
        "using the operator of matmul, and get raw shape of input Y.",
        std::vector<int>{}));
