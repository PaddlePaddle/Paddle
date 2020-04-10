/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/trace_op.h"

namespace paddle {
namespace operators {

class TraceOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Input"),
                   "Input(Input) of TraceOp should not be null.");

    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of TraceOp should not be null.");

    int dim1 = ctx->Attrs().Get<int>("dim1");
    int dim2 = ctx->Attrs().Get<int>("dim2");

    auto x_dims = ctx->GetInputDim("Input");

    int dim1_ = dim1 < 0 ? x_dims.size() + dim1 : dim1;
    int dim2_ = dim2 < 0 ? x_dims.size() + dim2 : dim2;

    PADDLE_ENFORCE_GE(x_dims.size(), 2,
                      platform::errors::OutOfRange(
                          "diag requires an array of at least two dimensions"));
    PADDLE_ENFORCE_LT(
        dim1_, x_dims.size(),
        platform::errors::OutOfRange(
            "Dim1 is out of range (expected to be in range of [%ld, "
            "%ld], but got %ld).",
            -(x_dims.size()), (x_dims.size() - 1), dim1));
    PADDLE_ENFORCE_LT(
        dim2_, x_dims.size(),
        platform::errors::OutOfRange(
            "Dim2 is out of range (expected to be in range of [%ld, "
            "%ld], but got %ld).",
            -(x_dims.size()), (x_dims.size() - 1), dim2));
    PADDLE_ENFORCE_NE(dim1_, dim2_,
                      platform::errors::InvalidArgument(
                          "diagonal dimensions should not be identical "
                          "%ld vs %ld.",
                          dim1, dim2));

    auto sizes = vectorize(x_dims);
    if (x_dims.size() == 2) {
      sizes.clear();
      sizes.push_back(1);
    } else {
      sizes.erase(sizes.begin() + std::max(dim1_, dim2_));
      sizes.erase(sizes.begin() + std::min(dim1_, dim2_));
    }
    ctx->SetOutputDim("Out", framework::make_ddim(sizes));
  }
};

class TraceOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input", "Input array, from which the diagonals are taken.");
    AddOutput("Out", "the sum along the diagonal");
    AddAttr<int>(
        "offset",
        R"DOC((int, default 0), Offset of the diagonal from the main diagonal. Can be both positive and negative. Defaults to 0.
        )DOC")
        .SetDefault(0);
    AddAttr<int>(
        "dim1",
        R"DOC((int, default 0), the first axis of the 2-D sub-arrays from which the diagonals should be taken.  Default: 0.
        )DOC")
        .SetDefault(-2);
    AddAttr<int>(
        "dim2",
        R"DOC((int, default 1), the second axis of the 2-D sub-arrays from which the diagonals should be taken. Default: 1.
        )DOC")
        .SetDefault(-1);
    AddComment(R"DOC(
    If a is 2-D, the sum along the diagonal is returned. If a has larger dimensions, then an array of sums along diagonals is returned. 
)DOC");
  }
};
class TraceOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("Input"), true,
                      "Input(Input) must not be null.");
    PADDLE_ENFORCE_EQ(ctx->HasOutput(framework::GradVarName("Input")), true,
                      "Input(Input@GRAD) should not be null.");
    ctx->SetOutputDim(framework::GradVarName("Input"),
                      ctx->GetInputDim("Input"));
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    trace, ops::TraceOp, ops::TraceOpMaker,
    paddle::framework::DefaultGradOpMaker<paddle::framework::OpDesc, true>,
    paddle::framework::DefaultGradOpMaker<paddle::imperative::OpBase, true>);
REGISTER_OPERATOR(trace_grad, ops::TraceOpGrad);
REGISTER_OP_CPU_KERNEL(
    trace, ops::TraceKernel<paddle::platform::CPUDeviceContext, int>,
    ops::TraceKernel<paddle::platform::CPUDeviceContext, float>,
    ops::TraceKernel<paddle::platform::CPUDeviceContext, double>,
    ops::TraceKernel<paddle::platform::CPUDeviceContext, int64_t>);
REGISTER_OP_CPU_KERNEL(
    trace_grad, ops::TraceGradKernel<paddle::platform::CPUDeviceContext, int>,
    ops::TraceGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::TraceGradKernel<paddle::platform::CPUDeviceContext, double>,
    ops::TraceGradKernel<paddle::platform::CPUDeviceContext, int64_t>);
