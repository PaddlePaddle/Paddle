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

#include "paddle/fluid/operators/distributed_ops/lookup_sparse_table_grad_split_op.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

class LookupSparseTableGradSplitOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {}
};

class LookupSparseTableGradSplitOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Grad",
             "(SelectedRows) Ids's type should be SelectedRows"
             "THe ids to be looked up in W.");

    AddAttr<bool>("is_entry",
                  "(bool)"
                  "sparse table need entry");

    AddAttr<std::string>("tablename",
                         "(string)"
                         "sparse table name");

    AddOutput("Row",
              "(LoDTensor) The lookup results, which have the "
              "same type as W.");
    AddOutput("Value",
              "(LoDTensor) The lookup results, which have the "
              "same type as W.");
    AddComment(R"DOC(
Lookup Sprase Tablel Operator.

This operator is used to perform lookup on parameter W,
then concatenated into a sparse tensor.

The type of Ids(Input) is SelectedRows, the rows of Ids contains
the ids to be looked up in W;
if the Id is not in the sparse table, this operator will return a
random value and set the value into the table for the next looking up.

)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    lookup_sparse_table_grad_split, ops::LookupSparseTableGradSplitOp,
    ops::LookupSparseTableGradSplitOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OP_CPU_KERNEL(
    lookup_sparse_table_grad_split,
    ops::LookupSparseTableGradSplitKernel<paddle::platform::CPUDeviceContext,
                                          float>,
    ops::LookupSparseTableGradSplitKernel<paddle::platform::CPUDeviceContext,
                                          double>);
