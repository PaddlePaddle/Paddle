/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/ddim.h"

namespace paddle {
namespace operators {

static phi::DDim ValidateShape(const std::vector<int> shape,
                               const framework::DDim& in_dims) {
  const int64_t in_size = phi::product(in_dims);
  auto in_dims_vec = phi::vectorize(in_dims);
  bool all_positive = std::all_of(in_dims_vec.cbegin(),
                                  in_dims_vec.cend(),
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
      PADDLE_ENFORCE_EQ(
          unk_dim_idx,
          -1,
          platform::errors::InvalidArgument(
              "Only one dimension value of 'shape' in ReshapeOp can "
              "be -1. But received shape = [%s], shape[%d] is also -1.",
              phi::make_ddim(shape),
              i));
      unk_dim_idx = i;
    } else if (shape[i] == copy_dim_val) {
      PADDLE_ENFORCE_LT(
          static_cast<int>(i),
          in_dims.size(),
          platform::errors::InvalidArgument(
              "The index of 0 in `shape` must be less than "
              "the input tensor X's dimensions. "
              "But received shape = [%s], shape[%d] = 0, X's shape = [%s], "
              "X's dimensions = %d.",
              phi::make_ddim(shape),
              i,
              in_dims,
              in_dims.size()));
    } else {
      PADDLE_ENFORCE_GT(
          shape[i],
          0,
          platform::errors::InvalidArgument(
              "Each dimension value of 'shape' in ReshapeOp must not "
              "be negative except one unknown dimension. "
              "But received  shape = [%s], shape[%d] = %d.",
              phi::make_ddim(shape),
              i,
              shape[i]));
    }

    // NOTE all non-zero values will be converted to True (include negative
    // value)
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
      PADDLE_ENFORCE_EQ(
          output_shape[unk_dim_idx] * capacity,
          -in_size,
          platform::errors::InvalidArgument(
              "The 'shape' attribute in ReshapeOp is invalid. "
              "The input tensor X'size must be divisible by known "
              "capacity of 'shape'. "
              "But received X's shape = [%s], X's size = %d, "
              "'shape' is [%s], known capacity of 'shape' is %d.",
              in_dims,
              in_size,
              phi::make_ddim(shape),
              capacity));
    } else {
      output_shape[unk_dim_idx] = -1;
    }
  } else {
    if (all_positive) {
      PADDLE_ENFORCE_EQ(
          capacity,
          in_size,
          platform::errors::InvalidArgument(
              "The 'shape' in ReshapeOp is invalid. "
              "The input tensor X'size must be equal to the capacity of "
              "'shape'. "
              "But received X's shape = [%s], X's size = %d, 'shape' is "
              "[%s], the capacity of 'shape' is %d.",
              in_dims,
              in_size,
              phi::make_ddim(shape),
              capacity));
    }
  }

  // support reshape with zero-input(input tensor with product(shape) == 0)
  // by now we require that if the input tensor is zero shape, the target
  // shape of output must be zero
  if (in_size == 0) {
    PADDLE_ENFORCE_LE(
        capacity,
        in_size,
        platform::errors::InvalidArgument(
            "The 'shape' in ReshapeOp is invalid. "
            "The input tensor X's shape = [%s], X's capacity = %d."
            "But the target shape of Out is [%s],  the "
            "capacity of 'Out' is %d.",
            in_dims,
            in_size,
            phi::make_ddim(shape),
            capacity));
  }

  return phi::make_ddim(output_shape);
}

static phi::DDim transpose_infer_shape(const phi::DDim& x_dims,
                                       const std::vector<int>& axis) {
  size_t x_rank = x_dims.size();
  size_t axis_size = axis.size();

  PADDLE_ENFORCE_EQ(x_rank,
                    axis_size,
                    platform::errors::InvalidArgument(
                        "The input tensor's dimension "
                        "should be equal to the axis's size. "
                        "But received input tensor's dimension is %d, "
                        "axis's size is %d",
                        x_rank,
                        axis_size));

  std::vector<int> count(axis_size, 0);
  for (size_t i = 0; i < axis_size; i++) {
    PADDLE_ENFORCE_GE(axis[i],
                      0,
                      platform::errors::InvalidArgument(
                          "The axis should be greater than or equal to 0."
                          "But received %d of axis[%d]",
                          axis[i],
                          i));

    PADDLE_ENFORCE_EQ(
        axis[i] < static_cast<int>(axis_size) && ++count[axis[i]] == 1,
        true,
        platform::errors::InvalidArgument(
            "Each element of Attribute axis should "
            "be a unique value range from 0 to (dims - 1), "
            "where the dims is the axis's size, "
            "unique value means this axis value can appear only once. "
            "But received axis[%d] is %d, axis_size is %d, "
            "count[axis[%d]] is %d",
            i,
            axis[i],
            axis_size,
            i,
            count[axis[i]]));
  }

  framework::DDim out_dims(x_dims);
  for (size_t i = 0; i < axis_size; i++) {
    out_dims[i] = x_dims[axis[i]];
  }
  return out_dims;
}

static phi::DDim reshape_infer_shape(const phi::DDim& x_dims,
                                     const std::vector<int>& shape) {
  PADDLE_ENFORCE_EQ(!shape.empty(),
                    true,
                    platform::errors::InvalidArgument(
                        "The parameter 'shape' in ReshapeOp must be set. "
                        "But received 'shape' is empty."));
  return ValidateShape(shape, x_dims);
}

class TransposeReshapeFusionOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(
        ctx->HasInput("X"), "Input", "X", "TransposeReshapeFusionOp");
    OP_INOUT_CHECK(
        ctx->HasOutput("Out"), "Output", "Out", "TransposeReshapeFusionOp");

    bool trans_first = ctx->Attrs().Get<bool>("trans_first");
    std::vector<int> axis = ctx->Attrs().Get<std::vector<int>>("axis");
    std::vector<int> shape = ctx->Attrs().Get<std::vector<int>>("shape");

    auto x_dims = ctx->GetInputDim("X");

    phi::DDim out_dims;
    if (trans_first) {
      auto dims = transpose_infer_shape(x_dims, axis);
      out_dims = reshape_infer_shape(dims, shape);
    } else {
      auto dims = reshape_infer_shape(x_dims, shape);
      out_dims = transpose_infer_shape(dims, axis);
    }

    ctx->SetOutputDim("Out", out_dims);
    ctx->ShareLoD("X", "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto input_data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
};

class TransposeReshapeFusionOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(
        "X",
        "(Tensor), The input tensor of transpose_reshape_fusion operator.");
    AddOutput(
        "Out",
        "(Tensor) The output tensor of transpose_reshape_fusion operator. ");

    AddAttr<bool>("trans_first",
                  "(bool, default true) transpose first or reshape first.")
        .SetDefault(true);
    AddAttr<std::vector<int>>(
        "axis",
        "(vector<int>) A list of values, and the size of the list should be "
        "the same with the input tensor rank. This operator permutes the input "
        "tensor's axes according to the values given.")
        .SetDefault({});
    AddAttr<std::vector<int>>(
        "shape", "(std::vector<int>) Target shape of reshape operator.")
        .SetDefault({});

    AddComment(R"DOC(
Transpose Reshape Fusion Operator.

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(
    transpose_reshape_fusion,
    ops::TransposeReshapeFusionOp,
    ops::TransposeReshapeFusionOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
