// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/diag_embed_op.h"

namespace paddle {
namespace operators {

class DiagEmbedOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Input"), true,
        platform::errors::NotFound("Input of DiagEmbedOp is not found."));

    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("Out"), true,
        platform::errors::NotFound("Output of DiagEmbedOp is not found."));

    int offset = ctx->Attrs().Get<int>("offset");
    int dim1 = ctx->Attrs().Get<int>("dim1");
    int dim2 = ctx->Attrs().Get<int>("dim2");

    auto x_dims = ctx->GetInputDim("Input");

    PADDLE_ENFORCE_GE(
        dim1, -(x_dims.size() + 1),
        platform::errors::OutOfRange(
            "Dim1 is out of range (expected to be in range of [%ld, "
            "%ld], but got %ld).",
            -(x_dims.size() + 1), x_dims.size(), dim1));
    PADDLE_ENFORCE_LE(
        dim1, x_dims.size(),
        platform::errors::OutOfRange(
            "Dim1 is out of range (expected to be in range of [%ld, "
            "%ld], but got %ld).",
            -(x_dims.size() + 1), x_dims.size(), dim1));

    PADDLE_ENFORCE_GE(
        dim2, -(x_dims.size() + 1),
        platform::errors::OutOfRange(
            "Dim2 is out of range (expected to be in range of [%ld, "
            "%ld], but got %ld).",
            -(x_dims.size() + 1), x_dims.size(), dim2));
    PADDLE_ENFORCE_LE(
        dim2, x_dims.size(),
        platform::errors::OutOfRange(
            "Dim2 is out of range (expected to be in range of [%ld, "
            "%ld], but got %ld).",
            -(x_dims.size() + 1), x_dims.size(), dim2));

    int dim1_ = dim1 < 0 ? x_dims.size() + dim1 + 1 : dim1;
    int dim2_ = dim2 < 0 ? x_dims.size() + dim2 + 1 : dim2;
    int offset_ = std::abs(offset);

    PADDLE_ENFORCE_NE(dim1_, dim2_,
                      platform::errors::InvalidArgument(
                          "diagonal dimensions should not be identical "
                          "%ld vs %ld.",
                          dim1, dim2));

    int new_dim_len = offset_ + x_dims[x_dims.size() - 1];
    auto sizes = vectorize(x_dims);
    sizes.pop_back();
    sizes.insert(sizes.begin() + std::min(dim1_, dim2_), new_dim_len);
    sizes.insert(sizes.begin() + std::max(dim1_, dim2_), new_dim_len);
    ctx->SetOutputDim("Out", phi::make_ddim(sizes));
  }
};

class DiagEmbedOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input", "The input tensor. Must be at least 1-dimensional.");
    AddOutput("Out", "A matrix whose certain 2D planes is diagonal matrix.");

    AddAttr<int>(
        "offset",
        R"DOC((int, default 0), which diagonal to consider. Default: 0 (main diagonal).
        )DOC")
        .SetDefault(0);
    AddAttr<int>(
        "dim1",
        R"DOC((int, default -2), first dimension with respect to which to take diagonal. Default: -2.
        )DOC")
        .SetDefault(-2);
    AddAttr<int>(
        "dim2",
        R"DOC((int, default -1), second dimension with respect to which to take diagonal. Default: -1.
        )DOC")
        .SetDefault(-1);

    AddComment(R"DOC(Creates a tensor whose diagonals of certain 2D planes 
              (specified by dim1 and dim2) are filled by input. 
              To facilitate creating batched diagonal matrices, 
              the 2D planes formed by the last two dimensions of the returned tensor
              are chosen by default. 
              )DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace platform = paddle::platform;
REGISTER_OPERATOR(
    diag_embed, ops::DiagEmbedOp, ops::DiagEmbedOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(
    diag_embed, ops::DiagEmbedKernel<paddle::platform::CPUDeviceContext, int>,
    ops::DiagEmbedKernel<paddle::platform::CPUDeviceContext, float>,
    ops::DiagEmbedKernel<paddle::platform::CPUDeviceContext, double>,
    ops::DiagEmbedKernel<paddle::platform::CPUDeviceContext, int64_t>);
