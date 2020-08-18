/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/gather_v2_op.h"
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/framework/ddim.h"

namespace paddle {
namespace operators {

class GatherV2Op : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true,
                      platform::errors::InvalidArgument(
                          "Input(X) of GatherOp should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasInput("Index"), true,
                      platform::errors::InvalidArgument(
                          "Input(Index) of GatherOp should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasInput("Axis"), true,
                      platform::errors::InvalidArgument(
                          "Input(Axis) of GatherOp should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Y"), true,
                      platform::errors::InvalidArgument(
                          "Output(Y) of GatherOp should not be null."));

    auto index_dims = ctx->GetInputDim("Index");
    PADDLE_ENFORCE(index_dims.size() == 1 ||
                   (index_dims.size() == 2 && index_dims[1] == 1));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }
};

class GatherV2GradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
    ctx->ShareLoD("X", /*-->*/ framework::GradVarName("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Y")),
                                   ctx.device_context());
  }
};

class GatherV2OpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The source input of gather op");
    AddInput("Index", "The index input of gather op");
    AddOutput("Y", "The output of gather op");
    AddInput("Axis",
             "The Tensor which contains the axis that we do gather operation.");
    AddComment(R"DOC(
Y is obtained by gathering entries of the axis dimension
of X indexed by Index and concatenate them together.
)DOC");
  }
};

template <typename T>
class GatherV2GradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("gather_v2_grad");
    op->SetInput("Index", this->Input("Index"));
    op->SetInput("X", this->Input("X"));
    op->SetInput("Axis", this->Input("Axis"));
    op->SetInput(framework::GradVarName("Y"), this->OutputGrad("Y"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(GatherV2GradNoNeedBufferVarInferer, "X");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(gather_v2, ops::GatherV2Op, ops::GatherV2OpMaker,
                  ops::GatherV2GradOpMaker<paddle::framework::OpDesc>,
                  ops::GatherV2GradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(gather_v2_grad, ops::GatherV2GradOp,
                  ops::GatherV2GradNoNeedBufferVarInferer);
REGISTER_OP_CPU_KERNEL(gather_v2, ops::GatherV2OpKernel<float>,
                       ops::GatherV2OpKernel<double>,
                       ops::GatherV2OpKernel<int>,
                       ops::GatherV2OpKernel<uint8_t>,
                       ops::GatherV2OpKernel<int64_t>);
REGISTER_OP_CPU_KERNEL(gather_grad_v2, ops::GatherV2GradientOpKernel<float>,
                       ops::GatherV2GradientOpKernel<double>,
                       ops::GatherV2GradientOpKernel<int>,
                       ops::GatherV2GradientOpKernel<uint8_t>,
                       ops::GatherV2GradientOpKernel<int64_t>);
