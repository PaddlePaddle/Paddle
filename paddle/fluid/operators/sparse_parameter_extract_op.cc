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

#include "paddle/fluid/operators/sparse_parameter_extract_op.h"

namespace paddle {
namespace operators {

class SparseParameterExtractOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(
        ctx->HasInput("Param"),
        "Input(Param) of SparseParameterExtractOp should not be null.");
    PADDLE_ENFORCE(
        ctx->HasInput("Grad"),
        "Input(Grad) of SparseParameterExtractOp should not be null.");
    PADDLE_ENFORCE(
        ctx->HasOutput("ParamOut"),
        "Output(ParamOut) of SparseParameterExtractOp should not be null.");

    auto param_dim = ctx->GetInputDim("Param");
    ctx->SetOutputDim("ParamOut", param_dim);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto data_type = framework::GetDataTypeOfVar(ctx.InputVar("Param"));
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

class SparseParameterExtractOpInferVarType
    : public framework::VarTypeInference {
 public:
  void operator()(const framework::OpDesc &op_desc,
                  framework::BlockDesc *block) const override {
    auto out_var_name = op_desc.Output("ParamOut").front();
    block->Var(out_var_name)->SetType(framework::proto::VarType::SELECTED_ROWS);
    block->Var(out_var_name)->SetDataType(block->Var("Grad")->GetDataType());
  }
};

class SparseParameterExtractOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Param", "(Tensor or SelectedRows) Input parameter");
    AddInput("Grad", "(SelectedRows) Input gradient");
    AddOutput("ParamOut",
              "(SelectedRows) "
              "sparse parameter updated by the sparse gradient");
    AddComment(R"DOC(

sparse parameter extract operator

extract the parameter updated by sparse gradient, used in distributed training.

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(sparse_parameter_extract, ops::SparseParameterExtractOp,
                  ops::SparseParameterExtractOpMaker,
                  paddle::framework::EmptyGradOpMaker,
                  ops::SparseParameterExtractOpInferVarType);
REGISTER_OP_CPU_KERNEL(sparse_parameter_extract,
                       ops::SparseParameterExtractOpKernel<float>,
                       ops::SparseParameterExtractOpKernel<double>);
