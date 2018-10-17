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

#include "paddle/fluid/operators/hash_op.h"
#include <string>

namespace paddle {
namespace operators {

class HashOp : public framework::OperatorWithKernel {
 public:
  HashOp(const std::string &type, const framework::VariableNameMap &inputs,
         const framework::VariableNameMap &outputs,
         const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of HashOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of HashOp should not be null.");

    auto dims = ctx->GetInputDim("X");
    int num_hash = ctx->Attrs().Get<int>("num_hash");
    int ids_rank = dims.size();
    dims[ids_rank - 1] = num_hash;

    ctx->SetOutputDim("Out", dims);
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

class HashOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) Input tensor of scale operator.");
    AddOutput("Out", "(Tensor) Output tensor of scale operator.");
    AddComment(R"DOC(
**Chen Xuyi Operator**
$$Out = scale*X$$
)DOC");
    AddAttr<int>("num_hash", "").SetDefault(1);
    AddAttr<int>("mod_by", "").SetDefault(100000);
  }
};

// class HashGradMaker : public framework::SingleGradOpDescMaker {
// public:
//  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;
//
//  std::unique_ptr<framework::OpDesc> Apply() const override {
//    auto *grad_op = new framework::OpDesc();
//    grad_op->SetType("hash");
//    grad_op->SetInput("X", OutputGrad("Out"));
//    grad_op->SetOutput("Out", InputGrad("X"));
//    grad_op->SetAttr("scale", GetAttr("scale"));
//    return std::unique_ptr<framework::OpDesc>(grad_op);
//  }
//};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_WITHOUT_GRADIENT(hash, ops::HashOp, ops::HashOpMaker);
REGISTER_OP_CPU_KERNEL(hash, ops::HashKerel<int>, ops::HashKerel<int64_t>);
