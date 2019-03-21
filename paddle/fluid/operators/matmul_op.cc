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
#include "paddle/fluid/operators/detail/safe_ref.h"
#include "paddle/fluid/operators/math/blas.h"

namespace paddle {
namespace operators {

#define MATMUL_COMPUTE_DIMENSION 2

/**
 * To validate the input dimension and return the final shape.
 */
static framework::DDim ValidateShape(const std::vector<int> shape,
                                     const framework::DDim &in_dims) {
  const int64_t in_size = framework::product(in_dims);
  auto in_dims_vec = framework::vectorize(in_dims);
  bool all_positive = std::all_of(in_dims_vec.cbegin(), in_dims_vec.cend(),
                                  [](int64_t i) { return i > 0; });
  // only one dimension can be set to -1, whose size will be automatically
  // infered.
  const int64_t unk_dim_val = -1;
  const int64_t copy_dim_val = 0;

  std::vector<int64_t> output_shape(shape.size(), 0);
  int64_t capacity = 1;
  int unk_dim_idx = -1;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] == unk_dim_val) {
      PADDLE_ENFORCE(unk_dim_idx == -1,
                     "Only one input dimension of Attr(shape) can be unknown.");
      unk_dim_idx = i;
    } else if (shape[i] == copy_dim_val) {
      PADDLE_ENFORCE(
          static_cast<int>(i) < in_dims.size(),
          "The index of dimension to copy from input shape must be less "
          "than the size of input shape.");
    } else {
      PADDLE_ENFORCE(
          shape[i] > 0,
          "Each input dimension of Attr(shape) must not be negtive except "
          "one unknown dimension.");
    }

    capacity *= (shape[i] ? shape[i] : in_dims[i]);
    output_shape[i] = (shape[i] ? static_cast<int64_t>(shape[i]) : in_dims[i]);
  }

  if (unk_dim_idx != -1) {
    if (all_positive) {
      // in_size < 0 and is un-determinate in compile time, skip the check,
      // for example, in_dims = [-1, 8, 1, 1], shape = [-1, 3, 8],
      // capacity = -24, in_size = -8, output_shape[0] = 0
      // the following check will fail.
      output_shape[unk_dim_idx] = -in_size / capacity;
      PADDLE_ENFORCE_EQ(output_shape[unk_dim_idx] * capacity, -in_size,
                        "Invalid shape is given.");
    } else {
      output_shape[unk_dim_idx] = -1;
    }
  } else {
    PADDLE_ENFORCE_EQ(capacity, in_size, "Invalid shape is given.");
  }
  return framework::make_ddim(output_shape);
}

/**
 * To do the transpose the dimension with axis.
 */
static framework::DDim transpose(const std::vector<int> axis,
                                 const framework::DDim &in_dims) {
  size_t axis_size = axis.size();
  size_t in_rank = in_dims.size();

  PADDLE_ENFORCE_EQ(in_rank, axis_size,
                    "The input tensor's rank(%d) "
                    "should be equal to the axis's size(%d)",
                    in_rank, axis_size);

  std::vector<int> count(axis_size, 0);
  for (size_t i = 0; i < axis_size; i++) {
    PADDLE_ENFORCE(
        axis[i] < static_cast<int>(axis_size) && ++count[axis[i]] == 1,
        "Each element of Attribute axis should be a unique value "
        "range from 0 to (dims - 1), "
        "where the dims is the axis's size");
  }

  framework::DDim out_dims(in_dims);
  for (size_t i = 0; i < axis_size; i++) {
    out_dims[i] = in_dims[axis[i]];
  }

  return out_dims;
}

/**
 * Use the input dimension and axis to update the stride and leading dimension.
 */
static void UpdateStrideLd(const framework::DDim &dims,
                           const std::vector<int> axis, int &stride,  // NOLINT
                           int &ld) {                                 // NOLINT
  int ndmis = dims.size();

  stride = 1;
  ld = 1;

  for (auto i = axis[ndmis - MATMUL_COMPUTE_DIMENSION - 1] + 1; i < ndmis;
       i++) {
    stride *= dims[i];
  }

  for (auto i = axis[ndmis - MATMUL_COMPUTE_DIMENSION] + 1; i < ndmis; i++) {
    ld *= dims[i];
  }
}

/**
 * Get row matrix shape from a vector shape. If the rank of x_dim > 1, the
 * original x_dim is returned.
 */
static framework::DDim RowMatrixFromVector(const framework::DDim &x_dim) {
  if (x_dim.size() > 1) {
    return x_dim;
  }
  return framework::make_ddim({1, x_dim[0]});
}

/**
 * Get column matrix shape from a vector shape. If the ran of y_dim > 1, the
 * original y_dim is returned.
 */
static framework::DDim ColumnMatrixFromVector(const framework::DDim &y_dim) {
  if (y_dim.size() > 1) {
    return y_dim;
  }
  return framework::make_ddim({y_dim[0], 1});
}

template <typename DeviceContext, typename T>
class MatMulKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto &x =
        detail::Ref(context.Input<framework::Tensor>("X"), "Cannot find X");
    auto &y =
        detail::Ref(context.Input<framework::Tensor>("Y"), "Cannot find Y");
    auto *out = context.Output<framework::Tensor>("Out");
    out->mutable_data<T>(context.GetPlace());

    auto transpose_x = context.Attr<bool>("transpose_X");
    auto transpose_y = context.Attr<bool>("transpose_Y");
    const std::vector<int> &shape_x = context.Attr<std::vector<int>>("shape_X");
    const std::vector<int> &shape_y = context.Attr<std::vector<int>>("shape_Y");
    const std::vector<int> &axis_x = context.Attr<std::vector<int>>("axis_X");
    const std::vector<int> &axis_y = context.Attr<std::vector<int>>("axis_Y");
    int ld_x = 0, ld_y = 0, ld_out = 0, stride_x = 0, stride_y = 0;
    auto dim_x = x.dims();
    auto dim_y = y.dims();

    if (shape_x.size() > 0 && shape_x.size() == axis_x.size()) {
      dim_x = ValidateShape(shape_x, dim_x);
      UpdateStrideLd(dim_x, axis_x, stride_x, ld_x);
      dim_x = transpose(axis_x, dim_x);
    }

    if (shape_y.size() > 0 && shape_y.size() == axis_y.size()) {
      dim_y = ValidateShape(shape_y, dim_y);
      UpdateStrideLd(dim_y, axis_y, stride_y, ld_y);
      dim_y = transpose(axis_y, dim_y);
    }

    auto blas = math::GetBlas<DeviceContext, T>(context);
    auto mat_dim_a = math::CreateMatrixDescriptor(RowMatrixFromVector(dim_x), 0,
                                                  transpose_x);
    auto mat_dim_b = math::CreateMatrixDescriptor(ColumnMatrixFromVector(dim_y),
                                                  0, transpose_y);
    auto scale = static_cast<T>(context.Attr<float>("alpha"));
    ld_x =
        ld_x != 0 ? ld_x : (transpose_x ? mat_dim_a.height_ : mat_dim_a.width_);
    ld_y =
        ld_y != 0 ? ld_y : (transpose_y ? mat_dim_a.width_ : mat_dim_b.width_);
    ld_out = ld_out != 0 ? ld_out : mat_dim_b.width_;
    stride_x = stride_x != 0 ? stride_x : mat_dim_a.stride_;
    stride_y = stride_y != 0 ? stride_y : mat_dim_b.stride_;

    blas.MatMul(x, mat_dim_a, ld_x, stride_x, y, mat_dim_b, ld_y, stride_y,
                scale, out, ld_out, T(0));
  }
};

// Reshape a rank-3 tensor from P x M x N to (P * M) x N.
// Identity op if the tensor is not of rank 3.
static framework::Tensor FoldInitDims(const framework::Tensor &input) {
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
static framework::Tensor FoldHeadAndLastDims(const DeviceContext &context,
                                             const framework::Tensor &input) {
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
 * Reshape a tensor to 3-D or 2-D tensor by matrix descriptor.
 *
 * The shape would be [BatchSize, H, W] or [H, W].
 * If transposed, `H,W` will be swapped.
 */
static void ReshapeTensorIntoMatrixSequence(
    framework::Tensor *x, const math::MatDescriptor &descriptor) {
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
  auto mat_dim_x = math::CreateMatrixDescriptor(x_dim, 0, trans_x);
  auto mat_dim_y = math::CreateMatrixDescriptor(y_dim, 0, trans_y);
  if (mat_dim_x.batch_size_ == 0 && mat_dim_y.batch_size_ == 0) {
    out->Resize({mat_dim_x.height_, mat_dim_y.width_});
  } else {
    out->Resize({std::max(mat_dim_x.batch_size_, mat_dim_y.batch_size_),
                 mat_dim_x.height_, mat_dim_y.width_});
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
              const framework::Tensor &a, bool trans_a,
              const framework::Tensor &b, bool trans_b,
              framework::Tensor *out) const {
    out->mutable_data<T>(context.GetPlace());
    auto blas = math::GetBlas<DeviceContext, T>(context);
    auto mat_dim_a = math::CreateMatrixDescriptor(a.dims(), 0, trans_a);
    auto mat_dim_b = math::CreateMatrixDescriptor(b.dims(), 0, trans_b);
    blas.MatMul(a, mat_dim_a, b, mat_dim_b,
                static_cast<T>(context.Attr<float>("alpha")), out, T(0));
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
      auto &ctx = context.template device_context<DeviceContext>();
      MatMul(context, is_fold_init_dims_a
                          ? FoldInitDims(a)
                          : FoldHeadAndLastDims<DeviceContext, T>(ctx, a),
             trans_a, is_fold_init_dims_b
                          ? FoldInitDims(b)
                          : FoldHeadAndLastDims<DeviceContext, T>(ctx, b),
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

class MatMulOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *context) const override {
    PADDLE_ENFORCE(context->HasInput("X"),
                   "Input(X) of MatMulOp should not be null.");
    PADDLE_ENFORCE(context->HasInput("Y"),
                   "Input(Y) of MatMulOp should not be null.");
    PADDLE_ENFORCE(context->HasOutput("Out"),
                   "Output(Out) of MatMulOp should not be null.");

    auto dim_x = context->GetInputDim("X");
    auto dim_y = context->GetInputDim("Y");

    const std::vector<int> &shape_x =
        context->Attrs().Get<std::vector<int>>("shape_X");
    const std::vector<int> &shape_y =
        context->Attrs().Get<std::vector<int>>("shape_Y");
    const std::vector<int> &axis_x =
        context->Attrs().Get<std::vector<int>>("axis_X");
    const std::vector<int> &axis_y =
        context->Attrs().Get<std::vector<int>>("axis_Y");

    if (shape_x.size() > 0 && shape_x.size() == axis_x.size()) {
      dim_x = ValidateShape(shape_x, dim_x);

      dim_x = transpose(axis_x, dim_x);
    }

    if (shape_y.size() > 0 && shape_y.size() == axis_y.size()) {
      dim_y = ValidateShape(shape_y, dim_y);
      dim_y = transpose(axis_y, dim_y);
    }

    auto mat_dim_x =
        math::CreateMatrixDescriptor(RowMatrixFromVector(dim_x), 0,
                                     context->Attrs().Get<bool>("transpose_X"));
    auto mat_dim_y =
        math::CreateMatrixDescriptor(ColumnMatrixFromVector(dim_y), 0,
                                     context->Attrs().Get<bool>("transpose_Y"));

    PADDLE_ENFORCE_EQ(mat_dim_x.width_, mat_dim_y.height_);
    PADDLE_ENFORCE(mat_dim_x.batch_size_ == mat_dim_y.batch_size_ ||
                   mat_dim_x.batch_size_ == 0 || mat_dim_y.batch_size_ == 0);
    std::vector<int64_t> dim_out;
    if (mat_dim_x.batch_size_ != 0) {
      dim_out = framework::vectorize(dim_x);
      dim_out[dim_out.size() - 2] = mat_dim_x.height_;
      dim_out[dim_out.size() - 1] = mat_dim_y.width_;
    } else if (mat_dim_y.batch_size_ != 0) {
      dim_out = framework::vectorize(dim_y);
      dim_out[dim_out.size() - 2] = mat_dim_x.height_;
      dim_out[dim_out.size() - 1] = mat_dim_y.width_;
    } else {
      dim_out = {mat_dim_x.height_, mat_dim_y.width_};
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
    context->SetOutputDim("Out", framework::make_ddim(dim_out));
    context->ShareLoD("X", /*->*/ "Out");
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
    AddAttr<std::vector<int>>("shape_X",
                              "(vector<int>) "
                              "the shape of x input. ")
        .SetDefault(std::vector<int>{});
    AddAttr<std::vector<int>>("shape_Y",
                              "(vector<int>) "
                              "the shape of y input.")
        .SetDefault(std::vector<int>{});
    AddAttr<std::vector<int>>("axis_X",
                              "(vector<int>) "
                              "the axis of x input. ")
        .SetDefault(std::vector<int>{});
    AddAttr<std::vector<int>>("axis_Y",
                              "(vector<int>) "
                              "the axis of y input.")
        .SetDefault(std::vector<int>{});
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

The behavior is designed to be similar to the `numpy.matmul` function.
The differences are:
- When the rank of the input data is less than or equal to 3, it
  is similar to the `numpy.matmul` function.
- When the rank of the input is greater than 3, the rank of X and
  Y must be equal, and the first `rank - 2` dimensions must be equal.
- We add `transpose_X` and `transpose_Y` flags.

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
    PADDLE_ENFORCE(context->HasInput("X"), "Input(X) should not be null");
    PADDLE_ENFORCE(context->HasInput("Y"), "Input(Y) should not be null");
    PADDLE_ENFORCE(context->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) should not be null");
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
};

class MatMulOpGradMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    auto *retv = new framework::OpDesc();
    retv->SetType("matmul_grad");
    retv->SetInput("X", Input("X"));
    retv->SetInput("Y", Input("Y"));
    retv->SetInput(framework::GradVarName("Out"), OutputGrad("Out"));
    retv->SetOutput(framework::GradVarName("X"), InputGrad("X"));
    retv->SetOutput(framework::GradVarName("Y"), InputGrad("Y"));
    retv->SetAttrMap(Attrs());
    return std::unique_ptr<framework::OpDesc>(retv);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(matmul, ops::MatMulOp, ops::MatMulOpMaker,
                  ops::MatMulOpGradMaker);
REGISTER_OPERATOR(matmul_grad, ops::MatMulOpGrad);
REGISTER_OP_CPU_KERNEL(
    matmul, ops::MatMulKernel<paddle::platform::CPUDeviceContext, float>,
    ops::MatMulKernel<paddle::platform::CPUDeviceContext, double>,
    ops::MatMulKernel<paddle::platform::CPUDeviceContext,
                      paddle::platform::float16>);
REGISTER_OP_CPU_KERNEL(
    matmul_grad,
    ops::MatMulGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::MatMulGradKernel<paddle::platform::CPUDeviceContext, double>,
    ops::MatMulGradKernel<paddle::platform::CPUDeviceContext,
                          paddle::platform::float16>);

#ifdef PADDLE_WITH_CUDA
REGISTER_OP_CUDA_KERNEL(
    matmul, ops::MatMulKernel<paddle::platform::CUDADeviceContext, float>,
    ops::MatMulKernel<paddle::platform::CUDADeviceContext, double>,
    ops::MatMulKernel<paddle::platform::CUDADeviceContext,
                      paddle::platform::float16>);
REGISTER_OP_CUDA_KERNEL(
    matmul_grad,
    ops::MatMulGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::MatMulGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::MatMulGradKernel<paddle::platform::CUDADeviceContext,
                          paddle::platform::float16>);
#endif
