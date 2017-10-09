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

#include "paddle/framework/op_registry.h"
#include "paddle/operators/net_op.h"

namespace paddle {
namespace operators {

class InterpOp : public NetOp {
 public:
  InterpOp(const std::string &type, const framework::VariableNameMap &inputs,
           const framework::VariableNameMap &outputs,
           const framework::AttributeMap &attrs)
      : NetOp(type, inputs, outputs, attrs) {
    PADDLE_ENFORCE_NE(Input("X"), framework::kEmptyVarName,
                      "Input(X) of InterpOp should not be null.");
    PADDLE_ENFORCE_NE(Input("Y"), framework::kEmptyVarName,
                      "Input(Y) of InterpOp should not be null.");
    PADDLE_ENFORCE_NE(Input("W"), framework::kEmptyVarName,
                      "Input(W) of InterpOp should not be null.");
    PADDLE_ENFORCE_NE(Output("MinusOut"), framework::kEmptyVarName,
                      "Output(MinusOut) of InterpOp should not be null.");
    PADDLE_ENFORCE_NE(Output("MulOut"), framework::kEmptyVarName,
                      "Output(MulOut) of InterpOp should not be null.");
    PADDLE_ENFORCE_NE(Output("Out"), framework::kEmptyVarName,
                      "Output(Out) of InterpOp should not be null.");

    // MinusOut = X - Y
    auto x = Input("X");
    auto y = Input("Y");
    auto minus_out = Output("MinusOut");
    AppendOp(framework::OpRegistry::CreateOp("elementwise_sub",
                                             {{"X", {x}}, {"Y", {y}}},
                                             {{"Out", {minus_out}}}, {}));

    // MulOut = MinusOut * W = (X - Y) * W
    auto w = Input("W");
    auto mul_out = Output("MulOut");
    AppendOp(framework::OpRegistry::CreateOp(
        "elementwise_mul", {{"X", {minus_out}}, {"Y", {w}}},
        {{"Out", {mul_out}}}, {{"axis", 0}}));

    // Out = MulOut + Y = (X - Y) * W + Y = X * W + Y * (1 - W)
    AppendOp(framework::OpRegistry::CreateOp("elementwise_add",
                                             {{"X", {mul_out}}, {"Y", {y}}},
                                             {{"Out", {Output("Out")}}}, {}));

    CompleteAddOp(false);
  }
};

class InterpOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  InterpOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "A 2-D Tensor, the first input of interp_op");
    AddInput("Y", "A 2-D Tensor, the second input of interp_op");
    AddInput("W", "A 1-D Tensor, the interpolated values");
    AddOutput("MinusOut",
              "A 2-D Tensor, the intermediate outputs, saving X - Y.")
        .AsIntermediate();
    AddOutput("MulOut",
              "A 2-D Tensor, the intermediate outputs,"
              "saving the mul mul of (X - Y) and W")
        .AsIntermediate();
    AddOutput("Out",
              "A 2-D Tensor, the output of interp_op, same shape with X");
    AddComment(R"DOC(
    Linear Interpolation with two inputs, used in NEURAL TURING MACHINE.

    Equation:
      Out.row[i] = X.row[i] * W[i] + Y.row[i] * (1 - W[i])
                 = (X.row[i] - Y.row[i]) * W[i] + Y.row[i]

    Example:
      X = [[1,2],[3,4]],
      Y = [[2,1],[4,3]],
      W = [0.3, 0.4]

      Then, Out = [[1.7,1.3],[3.6,3.4]]

      where 1.7 = 1*0.3+2*(1-0.3),
            1.3 = 2*0.3+1*(1-0.3),
            3.6 = 3*0.4+4*(1-0.4),
            3.4 = 4*0.4+3*(1-0.4)
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(interp, ops::InterpOp, ops::InterpOpMaker);
