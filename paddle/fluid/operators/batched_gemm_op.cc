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

#include "paddle/fluid/operators/batched_gemm_op.h"

namespace paddle {
namespace operators {

class BatchedGEMMOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true,
                      platform::errors::InvalidArgument(
                          "Input(X) of BatchedGEMM should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasInput("Y"), true,
                      platform::errors::InvalidArgument(
                          "Input(Y) of BatchedGEMM should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"), true,
                      platform::errors::InvalidArgument(
                          "Output(Out) of BatchedGEMM should not be null."));

    auto batch_count = ctx->Attrs().Get<int>("BatchCount");
    auto mat_m = ctx->Attrs().Get<int>("Mat_M");
    auto mat_n = ctx->Attrs().Get<int>("Mat_N");
    ctx->SetOutputDim("Out", {batch_count, mat_m, mat_n});
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }
};

class BatchedGEMMGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("X"), true,
        platform::errors::InvalidArgument("Input(X) should not be null"));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Y"), true,
        platform::errors::InvalidArgument("Input(Y) should not be null"));

    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
    ctx->SetOutputDim(framework::GradVarName("Y"), ctx->GetInputDim("Y"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.device_context());
  }
};

class BatchedGEMMOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) Input tensor of batched_gemm_op operator.");
    AddInput("Y", "(Tensor) Input tensor of batched_gemm_op operator.");
    AddOutput("Out", "Output tensor of batched_gemm_op operator.");
    AddAttr<int>("Mat_M", "(int, default 1) rows of X in batched_gemm_op")
        .SetDefault(1);
    AddAttr<int>("Mat_N", "(int, default 1) columns of Y in batched_gemm_op")
        .SetDefault(1);
    AddAttr<int>(
        "Mat_K",
        "(int, default 1) columns of X or rows of Y in batched_gemm_op")
        .SetDefault(1);
    AddAttr<int>("BatchCount", "(int, default 1) batch_cout of batched_gemm_op")
        .SetDefault(1);
    AddComment(R"DOC(
BatchedGEMM Operator.
This Op can calculate rank attention between input and rank_param, 
and rank_param gives the organization of data. Notice: It currently supports GPU device.
This Op exists in contrib, which means that it is not shown to the public.
)DOC");
  }
};

template <typename T>
class BatchedGEMMGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("batched_gemm_grad");

    op->SetInput("X", this->Input("X"));
    op->SetInput("Y", this->Input("Y"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));

    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("Y"), this->InputGrad("Y"));
    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(batched_gemm, ops::BatchedGEMMOp, ops::BatchedGEMMOpMaker,
                  ops::BatchedGEMMGradOpMaker<paddle::framework::OpDesc>,
                  ops::BatchedGEMMGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(batched_gemm_grad, ops::BatchedGEMMGradOp);

REGISTER_OP_CPU_KERNEL(
    batched_gemm,
    ops::BatchedGEMMKernel<paddle::platform::CPUDeviceContext, float>,
    ops::BatchedGEMMKernel<paddle::platform::CPUDeviceContext, double>);
