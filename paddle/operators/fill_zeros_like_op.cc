/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/operators/fill_zeros_like_op.h"

namespace paddle {
namespace operators {

class FillZerosLikeOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of FillZerosLikeOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of FillZerosLikeOp should not be null.");
    if (ctx->IsRuntime() &&
        ctx->GetInputsVarType("X")[0] ==
            framework::proto::VarDesc::LOD_TENSOR_ARRAY) {
      return;  // skip runtime infershape when is tensor array;
    }
    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
    ctx->ShareLoD("X", /*->*/ "Out");
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto *in_var = ctx.InputVar("X");
    if (in_var->IsType<framework::LoDTensor>()) {
      return framework::OpKernelType(
          framework::ToDataType(in_var->Get<framework::LoDTensor>().type()),
          ctx.GetPlace());
    } else {
      return framework::OpKernelType(
          framework::ToDataType(typeid(float)),  // NOLINT
          ctx.GetPlace());
    }
  }
};

class FillZerosLikeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  FillZerosLikeOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : framework::OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "The input of fill-zeros-like op.");
    AddOutput("Out", "The variable will be filled up with zeros.");
    AddComment(R"DOC(
FillZerosLike Operator.

Fill up a variable with zeros.
The output will have the same size as the input.

)DOC");
  }
};

class FillZeroVarTypeInferer : public framework::VarTypeInference {
 public:
  void operator()(const framework::OpDesc &op_desc,
                  framework::BlockDesc *block) const override {
    auto vtype =
        block->FindRecursiveOrCreateVar(op_desc.Input("X").at(0)).GetType();
    block->FindRecursiveOrCreateVar(op_desc.Output("Out").at(0)).SetType(vtype);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(fill_zeros_like, ops::FillZerosLikeOp,
                  ops::FillZerosLikeOpMaker, ops::FillZeroVarTypeInferer);
REGISTER_OP_CPU_KERNEL(
    fill_zeros_like,
    ops::FillZerosLikeKernel<paddle::platform::CPUDeviceContext, int>,
    ops::FillZerosLikeKernel<paddle::platform::CPUDeviceContext, int64_t>,
    ops::FillZerosLikeKernel<paddle::platform::CPUDeviceContext, float>,
    ops::FillZerosLikeKernel<paddle::platform::CPUDeviceContext, double>,
    ops::FillZerosLikeKernel<paddle::platform::CPUDeviceContext, bool>);
