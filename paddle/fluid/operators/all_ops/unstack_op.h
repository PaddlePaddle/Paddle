// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

class UnStackOpInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) must exist.");

    int axis = ctx->Attrs().Get<int>("axis");
    int num = ctx->Attrs().Get<int>("num");
    auto x_dim = ctx->GetInputDim("X");
    int rank = x_dim.size();
    PADDLE_ENFORCE(axis >= -rank && axis < rank,
                   "Attr(axis) must be inside [-rank, rank), where rank = %d",
                   rank);
    if (axis < 0) axis += rank;

    PADDLE_ENFORCE_EQ(ctx->Outputs("Y").size(), static_cast<size_t>(num),
                      "Number of Outputs(Y) is wrong");
    if (x_dim[axis] > 0) {
      PADDLE_ENFORCE_EQ(num, x_dim[axis], "Number of Outputs(Y) is wrong");
    }
    auto vec = framework::vectorize2int(x_dim);
    vec.erase(vec.begin() + axis);
    ctx->SetOutputsDim("Y", std::vector<framework::DDim>(  // NOLINT
                                x_dim[axis], framework::make_ddim(vec)));
  }
};

class UnStackOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input of unstack op.");
    AddOutput("Y", "The output of unstack op.").AsDuplicable();
    AddAttr<int>("axis", "The axis along which Input(X) should be unstacked.")
        .SetDefault(0);
    AddAttr<int>("num", "The number of outputs(Y).").GreaterThan(0);
    AddComment(R"DOC(
      UnStack Operator.

      UnStack Input(X) into several tensors along Attr(axis).
    )DOC");
  }
};

class UnStackOp : public framework::OperatorBase {
 public:
  using OperatorBase::OperatorBase;

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &place) const override {
    auto stack_grad_op = framework::OpRegistry::CreateOp(
        "stack_grad", {{framework::GradVarName("Y"), {Input("X")}}},
        {{framework::GradVarName("X"), Outputs("Y")}}, Attrs());
    stack_grad_op->Run(scope, place);
  }
};

class UnStackOpGradInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_GT(ctx->Inputs(framework::GradVarName("Y")).size(), 0,
                      "Number of Inputs(Y@Grad) must be larger than 0");
    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("X")),
                   "Output(X@Grad) must exist.");

    auto input_dims = ctx->GetInputsDim(framework::GradVarName("Y"));
    for (size_t i = 1; i < input_dims.size(); ++i) {
      PADDLE_ENFORCE_EQ(input_dims[i], input_dims[0],
                        "Dims of all Inputs(Y@Grad) must be the same");
    }

    int axis = ctx->Attrs().Get<int>("axis");
    int rank = input_dims[0].size();
    PADDLE_ENFORCE(
        axis >= -(rank + 1) && axis < rank + 1,
        "Attr(axis) must be inside [-(rank+1), rank+1), where rank = %d", rank);
    if (axis < 0) axis += (rank + 1);

    auto vec = framework::vectorize2int(input_dims[0]);
    vec.insert(vec.begin() + axis, input_dims.size());
    ctx->SetOutputDim(framework::GradVarName("X"), framework::make_ddim(vec));
  }
};

class UnStackGradOpDescMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    std::unique_ptr<framework::OpDesc> op(new framework::OpDesc());
    op->SetType("unstack_grad");
    op->SetInput(framework::GradVarName("Y"), OutputGrad("Y"));
    op->SetOutput(framework::GradVarName("X"), InputGrad("X"));
    op->SetAttrMap(Attrs());
    return op;
  }
};

class UnStackGradOp : public framework::OperatorBase {
 public:
  using OperatorBase::OperatorBase;

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &place) const override {
    auto stack_op = framework::OpRegistry::CreateOp(
        "stack", {{"X", Inputs(framework::GradVarName("Y"))}},
        {{"Y", {Output(framework::GradVarName("X"))}}}, Attrs());
    stack_op->Run(scope, place);
  }
};

}  // namespace operators
}  // namespace paddle
