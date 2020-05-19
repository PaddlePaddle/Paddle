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

#include "paddle/fluid/operators/gather_op.h"
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/framework/ddim.h"

namespace paddle {
namespace operators {

class GatherOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true,
                      platform::errors::InvalidArgument(
                          "Input(X) of GatherOp should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasInput("Index"), true,
                      platform::errors::InvalidArgument(
                          "Input(Index) of GatherOp should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"), true,
                      platform::errors::InvalidArgument(
                          "Output(Out) of GatherOp should not be null."));

    auto index_dims = ctx->GetInputDim("Index");
    PADDLE_ENFORCE(index_dims.size() == 1 ||
                   (index_dims.size() == 2 && index_dims[1] == 1));
    int batch_size = ctx->GetInputDim("Index")[0];
    framework::DDim output_dims(ctx->GetInputDim("X"));
    output_dims[0] = batch_size;
    ctx->SetOutputDim("Out", output_dims);
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

class GatherGradOp : public framework::OperatorWithKernel {
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
                                       ctx, framework::GradVarName("Out")),
                                   ctx.device_context());
  }
};

class GatherOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The source input of gather op");
    AddInput("Index", "The index input of gather op");
    AddOutput("Out", "The output of gather op");
    AddAttr<bool>(
        "overwrite",
        "(bool, default: False) "
        "In backward process, calc the grad when has same index,"
        "If true, update the grad using the overwrite mode in same index,"
        "If false, using the accumulate mode in same index.")
        .SetDefault(true);
    AddComment(R"DOC(
Gather Operator.

$Out = X[Index]$

Out is obtained by gathering entries of the outer-most dimension
of X indexed by Index and concatenate them together.

Example:

X = [[1, 2],
     [3, 4],
     [5, 6]]

Index = [[1, 2]]

Then:

Out = [[3, 4],
       [5, 6]]

)DOC");
  }
};

template <typename T>
class GatherGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("gather_grad");
    op->SetInput("Index", this->Input("Index"));
    op->SetInput("X", this->Input("X"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(GatherGradNoNeedBufferVarInference, "X");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(gather, ops::GatherOp, ops::GatherOpMaker,
                  ops::GatherGradOpMaker<paddle::framework::OpDesc>,
                  ops::GatherGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(gather_grad, ops::GatherGradOp,
                  ops::GatherGradNoNeedBufferVarInference);
REGISTER_OP_CPU_KERNEL(gather, ops::GatherOpKernel<float>,
                       ops::GatherOpKernel<double>, ops::GatherOpKernel<int>,
                       ops::GatherOpKernel<uint8_t>,
                       ops::GatherOpKernel<int64_t>);
REGISTER_OP_CPU_KERNEL(gather_grad, ops::GatherGradientOpKernel<float>,
                       ops::GatherGradientOpKernel<double>,
                       ops::GatherGradientOpKernel<int>,
                       ops::GatherGradientOpKernel<uint8_t>,
                       ops::GatherGradientOpKernel<int64_t>);
