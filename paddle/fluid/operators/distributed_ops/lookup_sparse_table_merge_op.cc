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

#include "paddle/fluid/operators/distributed_ops/lookup_sparse_table_merge_op.h"

namespace paddle {
namespace operators {

class LookupSparseTableMergeOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInputs("X"), true,
        platform::errors::InvalidArgument("Input(X) should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("Out"), true,
        platform::errors::InvalidArgument("Output(Out) should not be null."));

    PADDLE_ENFORCE_EQ(ctx->GetInputsVarType("X").front(),
                      framework::proto::VarType::SELECTED_ROWS,
                      platform::errors::InvalidArgument(
                          "Input X only should be SelectedRows."));
    PADDLE_ENFORCE_EQ(ctx->GetOutputsVarType("Out").front(),
                      framework::proto::VarType::SELECTED_ROWS,
                      platform::errors::InvalidArgument(
                          "Output Y only should be SelectedRows."));

    ctx->ShareDim("X", /*->*/ "Out");
  }
};

class LookupSparseTableMergeMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "The input type is SelectedRows, and the selected rows may be "
             "duplicated.")
        .AsDuplicable();
    AddOutput("Out",
              "The output type is SelectedRows, and the selected rows are not "
              "duplicated.");
    AddComment(
        R"DOC(
Merge sparse lookup table(selected rows as parameter).
)DOC");
  }
};

class LookupSparseTableMergeOpInferVarType
    : public framework::PassInDtypeAndVarTypeToOutput {
 protected:
  std::unordered_map<std::string, std::string>& GetInputOutputWithSameType()
      const override {
    static std::unordered_map<std::string, std::string> m{{"X", /*->*/ "Out"}};
    return m;
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OPERATOR(lookup_sparse_table_merge, ops::LookupSparseTableMergeOp,
                  ops::LookupSparseTableMergeMaker,
                  ops::LookupSparseTableMergeOpInferVarType);

REGISTER_OP_CPU_KERNEL(
    lookup_sparse_table_merge,
    ops::LookupSparseTableMergeKernel<plat::CPUDeviceContext, float>,
    ops::LookupSparseTableMergeKernel<plat::CPUDeviceContext, double>);
