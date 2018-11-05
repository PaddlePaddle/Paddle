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

#include "paddle/fluid/operators/get_sparse_as_op.h"
#include "paddle/fluid/framework/var_type_inference.h"

namespace paddle {
namespace operators {

class GetSparseAsOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("W"),
                   "Input(W) of GetSparseAsOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of GetSparseAsOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of GetSparseAsOp should not be null.");
    PADDLE_ENFORCE_EQ(
        ctx->GetInputsVarType("W")[0], framework::proto::VarType::LOD_TENSOR,
        "The type of Input(W) of GetSparseAsOp should be LoDTensor, but the "
        "received type is %s.",
        ctx->GetInputsVarType("W")[0]);
    PADDLE_ENFORCE_EQ(ctx->GetInputsVarType("X")[0],
                      framework::proto::VarType::SELECTED_ROWS,
                      "The type of Input(X) of GetSparseAsOp should be "
                      "SelectedRows, but the received type is %s",
                      ctx->GetInputsVarType("X")[0]);
    PADDLE_ENFORCE_EQ(ctx->GetOutputsVarType("Out")[0],
                      framework::proto::VarType::SELECTED_ROWS,
                      "The type of Output(Out) of GetSparseAsOp should be "
                      "SelectedRows, but the received type is %s",
                      ctx->GetInputsVarType("Out")[0]);

    auto table_dims = ctx->GetInputDim("W");
    auto x_dims = ctx->GetInputDim("X");

    PADDLE_ENFORCE_EQ(table_dims.size(), 2);
    PADDLE_ENFORCE_EQ(table_dims, x_dims);

    ctx->ShareDim("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = framework::GetDataTypeOfVar(ctx.InputVar("W"));
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

class GetSparseAsOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("W", "(LoDTensor) The input represents look up table.");
    AddInput("X", "(SelectedRows).");
    AddOutput("Out", "(SelectedRows).");

    AddComment(R"DOC(
Get Sparse As Operator.

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(get_sparse_as, ops::GetSparseAsOp, ops::GetSparseAsOpMaker);
REGISTER_OP_CPU_KERNEL(get_sparse_as, ops::GetSparseAsKernel<float>,
                       ops::GetSparseAsKernel<double>);
