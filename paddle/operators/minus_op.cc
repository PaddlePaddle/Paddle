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

#include "paddle/operators/minus_op.h"
#include "paddle/operators/net_op.h"

namespace paddle {
namespace operators {

class MinusOp : public framework::OperatorWithKernel {
 public:
  MinusOp(const std::string &type, const framework::VariableNameMap &inputs,
          const framework::VariableNameMap &outputs,
          const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

 protected:
  void InferShape(framework::InferShapeContextBase *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of MinusOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Y"),
                   "Input(Y) of MinusOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of MinusOp should not be null.");

    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");

    PADDLE_ENFORCE_EQ(
        x_dims, y_dims,
        "Minus operator must take two tensor with same num of elements");
    ctx->SetOutputDim("Out", x_dims);
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

class MinusOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  MinusOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "The left tensor of minus operator.");
    AddInput("Y", "The right tensor of minus operator.");
    AddOutput("Out", "The output tensor of minus operator.");

    AddComment(R"DOC(Minus Operator

Equation:

    Out = X - Y

Both the input `X` and `Y` can carry the LoD (Level of Details) information,
or not. But the output only shares the LoD with input `X`.
)DOC");
  }
};

class MinusGradMaker : public framework::GradOpDescMakerBase {
 public:
  using framework::GradOpDescMakerBase::GradOpDescMakerBase;

  std::vector<framework::OpDescBind> operator()() const override {
    std::vector<framework::OpDescBind> ops;
    ops.resize(2);

    ops[0].SetType("scale");
    ops[0].SetInput("X", OutputGrad("Out"));
    ops[0].SetOutput("Out", InputGrad("X"));
    ops[0].SetAttr("scale", 1.0f);

    ops[1].SetType("scale");
    ops[1].SetInput("X", OutputGrad("Out"));
    ops[1].SetOutput("Out", InputGrad("Y"));
    ops[1].SetAttr("scale", -1.0f);
    return ops;
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(minus, ops::MinusOp, ops::MinusOpMaker, ops::MinusGradMaker);
REGISTER_OP_CPU_KERNEL(minus,
                       ops::MinusKernel<paddle::platform::CPUPlace, float>);
