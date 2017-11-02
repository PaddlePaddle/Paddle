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

#include "paddle/operators/to_array_op.h"
#include "paddle/framework/data_type.h"

namespace paddle {
namespace operators {

class ToArrayOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of ToArrayOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("RankSortTable"),
                   "Input(RankSortTable) of ToArrayOp should not be null.");

    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of ToArrayOp should not be null.");
  }
  framework::DataType IndicateDataType(
      const framework::ExecutionContext &ctx) const override {
    auto *input_tensor = ctx.Input<framework::Tensor>("X");
    return framework::ToDataType(input_tensor->type());
  }
};

class ToArrayOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  ToArrayOpMaker(framework::OpProto *proto,
                 framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Input LoDTensor");
    AddInput("RankSortTable", "RankSortTable");
    AddOutput("Out", "std::vector<LoDTensor>");
    AddComment(
        "to_array, unpack LoDTensor to std::vector<LoDTensor> instructed by "
        "RankSortTable");
  }
};

class ToArrayOpVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(const framework::OpDescBind &op_desc,
                  framework::BlockDescBind *block) const override {
    auto out_var_name = op_desc.Output("Out").front();
    block->Var(out_var_name)->SetType(framework::VarDesc::FETCH_LIST);
  }
};

}  // operators
}  // paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(to_array, ops::ToArrayOp, paddle::framework::EmptyGradOpMaker,
                  ops::ToArrayOpMaker, ops::ToArrayOpVarTypeInference);

REGISTER_OP_CPU_KERNEL(
    to_array, ops::ToArrayOpKernel<paddle::platform::CPUPlace, float>,
    ops::ToArrayOpKernel<paddle::platform::CPUPlace, double>);
