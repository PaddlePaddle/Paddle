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

#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class AllGatherOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of AllGatherOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of AllGatherOp should not be null.");
    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
    ctx->ShareLoD("X", "Out");
  }
};

class AllGatherOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The input of all gather op.");
    AddOutput("Out",
              "(Tensor), The output of all gather op with the same"
              " rank as Input(X).");
    AddComment(R"DOC(
The all gather op gather data from all devices ordered by device id.

Given 3 devices and each device hold a tensor with shape [2]:

    device 0: [2, 1]
    device 1: [0, 4]
    device 2: [5, 3]

Apply gather op on all devices, achieves:
    
    device 0: [2, 1, 0, 4, 5, 3]
    device 1: [2, 1, 0, 4, 5, 3]
    device 2: [2, 1, 0, 4, 5, 3]

)DOC");
  }
};

class AllGatherOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) should not be null");
    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("X")),
                   "Output(X@GRAD) of should not be null.");
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
    ctx->ShareLoD("X", framework::GradVarName("X"));
  }
};

class AllGatherOpGradMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    auto* bind = new framework::OpDesc();
    bind->SetInput("X", Input("X"));
    bind->SetInput(framework::GradVarName("Out"), OutputGrad("Out"));
    bind->SetOutput(framework::GradVarName("X"), InputGrad("X"));
    bind->SetAttrMap(Attrs());
    bind->SetType("all_gather_grad");
    return std::unique_ptr<framework::OpDesc>(bind);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(all_gather, ops::AllGatherOp, ops::AllGatherOpMaker,
                  ops::AllGatherOpGradMaker);
REGISTER_OPERATOR(all_gather_grad, ops::AllGatherOpGrad);
