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

#include "paddle/operators/lstm_unit_op.h"

namespace paddle {
namespace operators {

class LstmUnitOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("X"),
                            "Input(X) of LSTM should not be null.");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("C_prev"),
                            "Input(C_prev) of LSTM should not be null.");
    PADDLE_ENFORCE_NOT_NULL(ctx.OutputVar("C"),
                            "Output(C) of LSTM should not be null.");
    PADDLE_ENFORCE_NOT_NULL(ctx.OutputVar("H"),
                            "Output(H) of LSTM should not be null.");

    auto *x = ctx.Input<framework::Tensor>("X");
    auto *c_prev = ctx.Input<framework::Tensor>("C_prev");

    PADDLE_ENFORCE_EQ(x->dims().size(), 2, "Input(X)'s rank must be 2.");
    PADDLE_ENFORCE(x->dims()[0] == c_prev->dims()[0],
                   "Batch size of inputs and states must be equal");
    PADDLE_ENFORCE(x->dims()[1] == c_prev->dims()[1] * 4,
                   "Dimension of FC should equal to prev state * 4");

    int b_size = c_prev->dims()[0];  // batch size
    int s_dim = c_prev->dims()[1];   // state dim
    ctx.Output<framework::LoDTensor>("C")->Resize({b_size, s_dim});
    ctx.Output<framework::LoDTensor>("H")->Resize({b_size, s_dim});
  }
};

template <typename AttrType>
class LstmUnitOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  LstmUnitOpMaker(framework::OpProto *proto,
                  framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "FC input before the non-linear activation.");
    AddInput(
        "C_prev",
        "The cell state tensor of last time-step in the Lstm Unit operator.");
    AddOutput("C", "The cell tensor of Lstm Unit operator.");
    AddOutput("H", "The hidden state tensor of Lstm Unit operator.");

    AddComment(R"DOC(Lstm-Unit Operator

Equation: 
  i, j, f, o = split(X)
  C = C_prev * sigm(f + forget_bias) + sigm(i) * tanh(j)
  H = C * sigm(o)
   
)DOC");
    AddAttr<AttrType>("forget_bias", "The forget bias of Lstm Unit.")
        .SetDefault(0.0);
  }
};

class LstmUnitGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar(framework::GradVarName("C")),
                            "Input(C@GRAD) should not be null");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar(framework::GradVarName("H")),
                            "Input(H@GRAD) should not be null");
    ctx.Output<framework::LoDTensor>(framework::GradVarName("X"))
        ->Resize(ctx.Input<Tensor>("X")->dims());
    ctx.Output<framework::LoDTensor>(framework::GradVarName("C_prev"))
        ->Resize(ctx.Input<Tensor>("C_prev")->dims());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(lstm_unit, ops::LstmUnitOp, ops::LstmUnitOpMaker<float>,
            lstm_unit_grad, ops::LstmUnitGradOp);
REGISTER_OP_CPU_KERNEL(lstm_unit,
                       ops::LstmUnitKernel<paddle::platform::CPUPlace, float>);
