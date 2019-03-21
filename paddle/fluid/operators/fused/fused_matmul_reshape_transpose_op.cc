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
 * Use the input dimension and axis to update the stride.
 */
static void UpdateStride(
    const framework::DDim &dims, const std::vector<int> axis,
    std::vector<std::pair<int64_t, int>> &stride) {  // NOLINT
  int ndmis = dims.size();

  for (auto i = 0; i < ndmis - MATMUL_COMPUTE_DIMENSION; i++) {
    int64_t count = 1;
    for (auto j = axis[i] + 1; j < ndmis; j++) {
      count *= dims[j];
    }
    stride.insert(stride.begin(), std::make_pair(count, dims[axis[i]]));
  }
}

/**
 * Use the input dimension and axis to update the input leading dimension.
 */
static void UpdateInputLeadingDimesion(const framework::DDim &dims,
                                       const std::vector<int> axis,
                                       int &ld) {  // NOLINT
  int ndmis = dims.size();

  ld = 1;

  for (auto i = axis[ndmis - MATMUL_COMPUTE_DIMENSION] + 1; i < ndmis; i++) {
    ld *= dims[i];
  }
}

/**
 * Use the input dimensions and axis to update the output leading dimension.
 */
static void UpdateOutputLeadingDimesion(const framework::DDim &dims_x,
                                        int transpose_x,
                                        const framework::DDim &dims_y,
                                        int transpose_y,
                                        const std::vector<int> axis,
                                        int &ld) {  // NOLINT
  auto ndmis_x = dims_x.size();
  auto ndmis_y = dims_y.size();

  std::vector<int> dims = paddle::framework::vectorize2int(dims_x);
  auto ndmis = ndmis_x;

  if (ndmis_x == ndmis_y && ndmis_x == static_cast<int>(axis.size())) {
    if (transpose_x) {
      dims[ndmis - 2] = dims_x[ndmis_x - 1];
    }
    dims[ndmis - 1] = dims_y[ndmis_y - 1];
    if (transpose_y) {
      dims[ndmis - 1] = dims_y[ndmis_y - 2];
    }

    ld = 1;

    for (auto i = 0; i < ndmis; i++) {
      if (axis[i] >= ndmis - MATMUL_COMPUTE_DIMENSION) {
        ld *= dims[i];
      }
    }
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
class FusedMatmulReshapeTransposeKernel : public framework::OpKernel<T> {
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
    const std::vector<int> &axis_out =
        context.Attr<std::vector<int>>("axis_Out");
    int ld_x = 0, ld_y = 0, ld_out = 0;
    std::vector<std::pair<int64_t, int>> stride_x, stride_y;
    auto dim_x = x.dims();
    auto dim_y = y.dims();

    if (shape_x.size() > 0 && shape_x.size() == axis_x.size()) {
      dim_x = ValidateShape(shape_x, dim_x);
      UpdateStride(dim_x, axis_x, stride_x);
      UpdateInputLeadingDimesion(dim_x, axis_x, ld_x);
      dim_x = transpose(axis_x, dim_x);
    }

    if (shape_y.size() > 0 && shape_y.size() == axis_y.size()) {
      dim_y = ValidateShape(shape_y, dim_y);
      UpdateStride(dim_y, axis_y, stride_y);
      UpdateInputLeadingDimesion(dim_y, axis_y, ld_y);
      dim_y = transpose(axis_y, dim_y);
    }

    if (axis_out.size() > 0) {
      UpdateOutputLeadingDimesion(dim_x, transpose_x, dim_y, transpose_y,
                                  axis_out, ld_out);
    }

    auto blas = math::GetBlas<DeviceContext, T>(context);
    auto mat_dim_a = math::CreateMatrixDescriptor(RowMatrixFromVector(dim_x), 0,
                                                  transpose_x);
    auto mat_dim_b = math::CreateMatrixDescriptor(ColumnMatrixFromVector(dim_y),
                                                  0, transpose_y);
    auto scale = static_cast<T>(context.Attr<float>("alpha"));
    auto batchCount = mat_dim_a.batch_size_ == 0 ? mat_dim_b.batch_size_
                                                 : mat_dim_a.batch_size_;
    ld_x =
        ld_x != 0 ? ld_x : (transpose_x ? mat_dim_a.height_ : mat_dim_a.width_);
    ld_y =
        ld_y != 0 ? ld_y : (transpose_y ? mat_dim_a.width_ : mat_dim_b.width_);
    ld_out = ld_out != 0 ? ld_out : mat_dim_b.width_;
    stride_x = stride_x.empty()
                   ? std::vector<std::pair<int64_t, int>>(
                         {std::make_pair(mat_dim_a.stride_, batchCount)})
                   : stride_x;
    stride_y = stride_y.empty()
                   ? std::vector<std::pair<int64_t, int>>(
                         {std::make_pair(mat_dim_b.stride_, batchCount)})
                   : stride_y;

    blas.MatMul(x, mat_dim_a, ld_x, stride_x, y, mat_dim_b, ld_y, stride_y,
                scale, out, ld_out, T(0));
  }
};

class FusedMatmulReshapeTransposeOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *context) const override {
    PADDLE_ENFORCE(
        context->HasInput("X"),
        "Input(X) of FusedMatmulReshapeTransposeOp should not be null.");
    PADDLE_ENFORCE(
        context->HasInput("Y"),
        "Input(Y) of FusedMatmulReshapeTransposeOp should not be null.");
    PADDLE_ENFORCE(
        context->HasOutput("Out"),
        "Output(Out) of FusedMatmulReshapeTransposeOp should not be null.");

    auto dim_x = context->GetInputDim("X");
    auto dim_y = context->GetInputDim("Y");

    const std::vector<int> &shape_x =
        context->Attrs().Get<std::vector<int>>("shape_X");
    const std::vector<int> &shape_y =
        context->Attrs().Get<std::vector<int>>("shape_Y");
    const std::vector<int> &shape_out =
        context->Attrs().Get<std::vector<int>>("shape_Out");
    const std::vector<int> &axis_x =
        context->Attrs().Get<std::vector<int>>("axis_X");
    const std::vector<int> &axis_y =
        context->Attrs().Get<std::vector<int>>("axis_Y");
    const std::vector<int> &axis_out =
        context->Attrs().Get<std::vector<int>>("axis_Out");

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

    if (shape_out.size() > 0 && dim_out.size() == axis_out.size()) {
      auto dim = transpose(axis_out, framework::make_ddim(dim_out));
      dim = ValidateShape(shape_out, dim);
      context->SetOutputDim("Out", dim);
    }

    context->ShareLoD("X", /*->*/ "Out");
  }
};

class FusedMatmulReshapeTransposeOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The first input of FusedMatmulReshapeTranspose op");
    AddInput("Y", "The second input of FusedMatmulReshapeTranspose op");
    AddOutput("Out", "The output of FusedMatmulReshapeTranspose op");
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
    AddAttr<std::vector<int>>("shape_Out",
                              "(vector<int>) "
                              "the shape of output.")
        .SetDefault(std::vector<int>{});
    AddAttr<std::vector<int>>("axis_X",
                              "(vector<int>) "
                              "the axis of x input. ")
        .SetDefault(std::vector<int>{});
    AddAttr<std::vector<int>>("axis_Y",
                              "(vector<int>) "
                              "the axis of y input.")
        .SetDefault(std::vector<int>{});
    AddAttr<std::vector<int>>("axis_Out",
                              "(vector<int>) "
                              "the axis of output.")
        .SetDefault(std::vector<int>{});
    AddComment(R"DOC(
      FusedMatmulReshapeTranspose Operator.
    )DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(fused_matmul_reshape_transpose,
                             ops::FusedMatmulReshapeTransposeOp,
                             ops::FusedMatmulReshapeTransposeOpMaker);

REGISTER_OP_CPU_KERNEL(
    fused_matmul_reshape_transpose,
    ops::FusedMatmulReshapeTransposeKernel<paddle::platform::CPUDeviceContext,
                                           float>,
    ops::FusedMatmulReshapeTransposeKernel<paddle::platform::CPUDeviceContext,
                                           double>);
