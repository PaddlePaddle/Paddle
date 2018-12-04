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

#include "paddle/fluid/operators/selected_rows_mul_tensor_op.h"

namespace paddle {
namespace operators {

class SelectedRowsMulTensorOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of SelectedRowsMulTensorOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Y"),
                   "Input(Y) of SelectedRowsMulTensorOp should not be null.");
    PADDLE_ENFORCE(
        ctx->HasOutput("Out"),
        "Output(Out) of SelectedRowsMulTensorOp should not be null.");

    PADDLE_ENFORCE_EQ(ctx->GetInputDim("Y").size(), 1);
    PADDLE_ENFORCE_EQ(ctx->GetInputDim("Y")[0], 1);

    ctx->ShareDim("X", /*->*/ "Out");
  }
};

class SelectedRowsMulTensorOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input type is SelectedRows.");
    AddInput("Y", "The input type is LoDTensor.");
    AddOutput("Out", "The output type is SelectedRows.");
    AddComment(
        R"DOC(
SelectedRowsMulTensor Operator.

SelectedRowsMulTensor is used to merge the duplicated rows of the input.
)DOC");
  }
};

class SelectedRowsMulTensorOpInferVarType
    : public framework::PassInDtypeAndVarTypeToOutput {
 protected:
  std::unordered_map<std::string, std::string> GetInputOutputWithSameType()
      const override {
    return std::unordered_map<std::string, std::string>{{"X", /*->*/ "Out"}};
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OPERATOR(selected_rows_mul_tensor, ops::SelectedRowsMulTensorOp,
                  ops::SelectedRowsMulTensorOpMaker,
                  ops::SelectedRowsMulTensorOpInferVarType);

REGISTER_OP_CPU_KERNEL(
    selected_rows_mul_tensor,
    ops::SelectedRowsMulTensorKernel<plat::CPUDeviceContext, float>,
    ops::SelectedRowsMulTensorKernel<plat::CPUDeviceContext, double>);
