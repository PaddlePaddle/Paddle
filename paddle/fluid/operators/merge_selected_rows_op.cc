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

#include "paddle/fluid/operators/merge_selected_rows_op.h"

namespace paddle {
namespace operators {

class MergeSelectedRowsOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of MergeSelectedRowsOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of MergeSelectedRowsOp should not be null.");
    ctx->ShareDim("X", /*->*/ "Out");
  }
};

class MergeSelectedRowsOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "The input type is SelectedRows, and the selected rows may be "
             "duplicated.");
    AddOutput("Out",
              "The output type is SelectedRows, and the selected rows are not "
              "duplicated.");
    AddComment(
        R"DOC(
MergeSelectedRows Operator.

MergeSelectedRows is used to merge the duplicated rows of the input.
)DOC");
  }
};

class MergeSelectedRowsOpInferVarType
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
REGISTER_OPERATOR(merge_selected_rows, ops::MergeSelectedRowsOp,
                  ops::MergeSelectedRowsOpMaker,
                  ops::MergeSelectedRowsOpInferVarType);

REGISTER_OP_CPU_KERNEL(
    merge_selected_rows,
    ops::MergeSelectedRowsKernel<plat::CPUDeviceContext, float>,
    ops::MergeSelectedRowsKernel<plat::CPUDeviceContext, double>);
