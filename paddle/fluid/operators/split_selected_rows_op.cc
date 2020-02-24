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

#include "paddle/fluid/operators/split_selected_rows_op.h"

#include <memory>

namespace paddle {
namespace operators {

class SplitSelectedRowsOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input SelectedRows.");
    AddOutput("Out", "The outputs of the input SelectedRows.").AsDuplicable();
    AddAttr<std::vector<int64_t>>("height_sections",
                                  "Height for each output SelectedRows.")
        .SetDefault(std::vector<int64_t>({}));

    AddComment(R"DOC(
Split a SelectedRows with a specified rows section.
height_sections is only needed when need to split the dims of the original tensor.

Example:
  Input:
    X.rows = {7, 5}
    X.height = 12
  Attr:
    height_sections = {4, 8}
  Out:
    out0.rows = {}
    out0.height = 4

    out1.rows = {5, 7}
    out2.height = 8

)DOC");
  }
};

class SplitSelectedRowsOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "SplitSelectedRowsOp must have input X.");
    PADDLE_ENFORCE(ctx->HasOutputs("Out"),
                   "SplitSelectedRowsOp must have output Out.");
  }
};

class SplitSelectedRowsOpInferVarType : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    for (auto &out_var : ctx->Output("Out")) {
      ctx->SetType(out_var, framework::proto::VarType::SELECTED_ROWS);
    }
  }
};

template <typename T>
class SplitSelectedRowsGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  std::unique_ptr<T> Apply() const override {
    auto *grad_op = new T();
    grad_op->SetType("sum");
    grad_op->SetInput("X", this->OutputGrad("Out"));
    grad_op->SetOutput("Out", this->InputGrad("X"));
    grad_op->SetAttrMap(this->Attrs());
    return std::unique_ptr<T>(grad_op);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(split_selected_rows, ops::SplitSelectedRowsOp,
                  ops::SplitSelectedRowsOpMaker,
                  ops::SplitSelectedRowsGradMaker<paddle::framework::OpDesc>,
                  ops::SplitSelectedRowsGradMaker<paddle::imperative::OpBase>,
                  ops::SplitSelectedRowsOpInferVarType);
REGISTER_OP_CPU_KERNEL(
    split_selected_rows,
    ops::SplitSelectedRowsOpKernel<paddle::platform::CPUPlace, float>);
