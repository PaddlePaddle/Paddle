// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/operators/diagonal_op.h"

namespace paddle {
namespace operators {

class DiagonalOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Input"), true,
        platform::errors::NotFound("Input of DiagonalOp is not found."));

    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("Out"), true,
        platform::errors::NotFound("Output of DiagonalOp is not found."));

    int offset_ = ctx->Attrs().Get<int>("offset");

    int dim1 = ctx->Attrs().Get<int>("axis1");
    int dim2 = ctx->Attrs().Get<int>("axis2");

    auto x_dims = ctx->GetInputDim("Input");

    int dim1_ = dim1 < 0 ? x_dims.size() + dim1 : dim1;
    int dim2_ = dim2 < 0 ? x_dims.size() + dim2 : dim2;

    PADDLE_ENFORCE_GE(
        x_dims.size(), 2,
        platform::errors::OutOfRange("Input's dim is out of range (expected at "
                                     "least 2 dimensions, but got %ld).",
                                     x_dims.size()));
    PADDLE_ENFORCE_LT(
        dim1_, x_dims.size(),
        platform::errors::OutOfRange(
            "Attr(axis1) is out of range (expected to be in range of [%ld, "
            "%ld], but got %ld).",
            -(x_dims.size()), (x_dims.size() - 1), dim1));
    PADDLE_ENFORCE_LT(
        dim2_, x_dims.size(),
        platform::errors::OutOfRange(
            "Attr(axis2) is out of range (expected to be in range of [%ld, "
            "%ld], but got %ld).",
            -(x_dims.size()), (x_dims.size() - 1), dim2));
    PADDLE_ENFORCE_NE(dim1_, dim2_,
                      platform::errors::InvalidArgument(
                          "The dimensions should not be identical "
                          "%ld vs %ld.",
                          dim1, dim2));

    auto out_dims = vectorize(x_dims);
    auto dim1_size = out_dims[dim1_];
    auto dim2_size = out_dims[dim2_];
    out_dims.erase(out_dims.begin() + std::max(dim1_, dim2_));
    out_dims.erase(out_dims.begin() + std::min(dim1_, dim2_));

    if (offset_ == 0) {
      out_dims.push_back(std::min(dim1_size, dim2_size));
    } else if (offset_ > 0) {
      out_dims.push_back(std::min(dim1_size, dim2_size - offset_));
    } else {
      out_dims.push_back(std::min(dim1_size + offset_, dim2_size));
    }
    ctx->SetOutputDim("Out", framework::make_ddim(out_dims));
  }
};

class DiagonalOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input",
             "(Tensor) The input tensor, from which the diagonals are taken.");
    AddOutput(
        "Out",
        "(Tensor) The partial view of input with the its diagonal elements.");
    AddAttr<int>(
        "offset",
        R"DOC((int, default 0), offset of the diagonal from the main diagonal. Can be both positive and negative. Defaults to 0.
        )DOC")
        .SetDefault(0);
    AddAttr<int>(
        "axis1",
        R"DOC((int, default 0), the first axis of the 2-D planes from which the diagonals should be taken. 
        Can be either positive or negative. Default: 0.
        )DOC")
        .SetDefault(0);
    AddAttr<int>(
        "axis2",
        R"DOC((int, default 1), the second axis of the 2-D planes from which the diagonals should be taken. 
        Can be either positive or negative. Default: 1.
        )DOC")
        .SetDefault(1);
    AddComment(R"DOC(
Diagonal Operator.
Return a partial view of input with the its diagonal elements of the input tensor.
The behavior of this operator is similar to how `numpy.diagonal` works.

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    diagonal, ops::DiagonalOp, ops::DiagonalOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OP_CPU_KERNEL(diagonal, ops::DiagonalKernel<int>,
                       ops::DiagonalKernel<int64_t>, ops::DiagonalKernel<float>,
                       ops::DiagonalKernel<double>);
