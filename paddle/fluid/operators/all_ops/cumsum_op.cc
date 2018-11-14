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

#include "paddle/fluid/operators/cum_op.h"

namespace paddle {
namespace operators {

class CumOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

class CumsumOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Input of cumsum operator");
    AddOutput("Out", "Output of cumsum operator");
    AddAttr<int>("axis",
                 "The dimenstion to accumulate along. -1 means the last "
                 "dimenstion [default -1].")
        .SetDefault(-1)
        .EqualGreaterThan(-1);
    AddAttr<bool>("exclusive",
                  "Whether to perform exclusive cumsum. [default false].")
        .SetDefault(false);
    AddAttr<bool>("reverse",
                  "If true, the cumsum is performed in the reversed direction. "
                  "[default false].")
        .SetDefault(false);
    AddComment(R"DOC(
The cumulative sum of the elements along a given axis.
By default, the first element of the result is the same of the first element of
the input. If exlusive is true, the first element of the result is 0.
)DOC");
  }
};

class CumsumGradMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    auto *grad_op = new framework::OpDesc();
    grad_op->SetType("cumsum");
    grad_op->SetInput("X", OutputGrad("Out"));
    grad_op->SetOutput("Out", InputGrad("X"));
    grad_op->SetAttr("axis", Attr<int>("axis"));
    grad_op->SetAttr("reverse", !Attr<bool>("reverse"));
    grad_op->SetAttr("exclusive", Attr<bool>("exclusive"));
    return std::unique_ptr<framework::OpDesc>(grad_op);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CPU = paddle::platform::CPUDeviceContext;

REGISTER_OPERATOR(cumsum, ops::CumOp, ops::CumsumOpMaker, ops::CumsumGradMaker);
REGISTER_OP_CPU_KERNEL(cumsum, ops::CumKernel<CPU, ops::CumsumFunctor<float>>,
                       ops::CumKernel<CPU, ops::CumsumFunctor<double>>,
                       ops::CumKernel<CPU, ops::CumsumFunctor<int>>);
