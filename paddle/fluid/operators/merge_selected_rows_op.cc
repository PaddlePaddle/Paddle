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

#include "paddle/fluid/operators/merge_selected_rows_op.h"

namespace paddle {
namespace operators {

class MergeSelectedRowsOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(vector<SelectedRows>) The input SelectedRows.")
        .AsDuplicable();
    AddOutput("Out", "The merged output of the input SelectedRows.");
    AddAttr<std::vector<int64_t>>("height_sections",
                                  "Height for each input SelectedRows.")
        .SetDefault(std::vector<int64_t>({}));

    AddComment(R"DOC(
Merge several SelectedRows that is splited by split_selected_rows_op with a specified rows section.
height_sections is only needed when need to split the dims of the original selected rows.

Example:
  Input:
    x0.rows = {}
    x0.height = 4

    x1.rows = {1, 3}
    x1.height = 4

    x2.rows = {0}
    x2.height = 4

  Attr:
    height_sections = {4, 8}

  Out:
    out.rows = {5, 7}
    out.height = 12

)DOC");
  }
};

class MergeSelectedRowsOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInputs("X"),
                   "MergeSelectedRowsOp must has input X.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "MergeSelectedRowsOp must has output Out.");
  }

 private:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.MultiInput<framework::SelectedRows>("X")
                                  .front()
                                  ->value()
                                  .type()),
        ctx.GetPlace());
  }
};

class MergeSelectedRowsOpInferVarType : public framework::VarTypeInference {
 public:
  void operator()(const framework::OpDesc &op_desc,
                  framework::BlockDesc *block) const override {
    for (auto &out_var : op_desc.Output("Out")) {
      block->Var(out_var)->SetType(framework::proto::VarType::SELECTED_ROWS);
    }
  }
};

class MergeSelectedRowsGradMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    auto *grad_op = new framework::OpDesc();
    grad_op->SetType("split_selected_rows");
    grad_op->SetInput("X", OutputGrad("Out"));
    grad_op->SetOutput("Out", InputGrad("X"));
    grad_op->SetAttrMap(Attrs());
    return std::unique_ptr<framework::OpDesc>(grad_op);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(merge_selected_rows, ops::MergeSelectedRowsOp,
                  ops::MergeSelectedRowsOpMaker,
                  ops::MergeSelectedRowsGradMaker,
                  ops::MergeSelectedRowsOpInferVarType);
REGISTER_OP_CPU_KERNEL(
    merge_selected_rows,
    ops::MergeSelectedRowsOpKernel<paddle::platform::CPUDeviceContext, float>);
