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

#include "paddle/fluid/operators/sparse_parameter_update_op.h"

namespace paddle {
namespace operators {

class SparseParameterUpdateOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(
        ctx->HasInput("Param"),
        "Input(Param) of SparseParameterUpdateOp should not be null.");
    PADDLE_ENFORCE(
        ctx->HasInput("SparseParam"),
        "Input(SparseParam) of SparseParameterUpdateOp should not be null.");
    PADDLE_ENFORCE(
        ctx->HasOutput("ParamOut"),
        "Output(ParamOut) of SparseParameterUpdateOp should not be null.");
    PADDLE_ENFORCE_EQ(ctx->Inputs("Param"), ctx->Outputs("ParamOut"),
                      "Param and ParamOut should be the same");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto data_type = framework::GetDataTypeOfVar(ctx.InputVar("Param"));
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

class SparseParameterUpdateOpInferVarType : public framework::VarTypeInference {
 public:
  void operator()(const framework::OpDesc &op_desc,
                  framework::BlockDesc *block) const override {}
};

class SparseParameterUpdateOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Param", "(Tensor) Input parameter");
    AddInput("SparseParam",
             "(SelectedRows) Input Sparse parameter get from parameter server");
    AddOutput("ParamOut",
              "(Tensor) "
              "parameter updated by the sparse Parameter, should be same the "
              "input Param");
    AddComment(R"DOC(

sparse parameter update operator

update the value in SparseParam to Param.

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(sparse_parameter_update, ops::SparseParameterUpdateOp,
                  ops::SparseParameterUpdateOpMaker,
                  paddle::framework::EmptySparseParamOpMaker,
                  ops::SparseParameterUpdateOpInferVarType);
REGISTER_OP_CPU_KERNEL(sparse_parameter_update,
                       ops::SparseParameterUpdateOpKernel<float>,
                       ops::SparseParameterUpdateOpKernel<double>);
