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

class MinusOpMaker : public framework::OpInfoMaker {
 public:
  MinusOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpInfoMaker(proto, op_checker) {
    AddInput("X", "The left tensor of minus operator.").NotInGradient();
    AddInput("Y", "The right tensor of minus operator.").NotInGradient();
    AddOutput("Out", "The output tensor of minus operator.").NotInGradient();

    AddComment(R"DOC(Minus Operator

Equation:

    Out = X - Y

Both the input `X` and `Y` can carry the LoD (Level of Details) information,
or not. But the output only shares the LoD with input `X`.
)DOC");
  }
};
template <typename AttrType>
class MinusGradOp : public NetOp {
 public:
  MinusGradOp(const std::string &type, const framework::VariableNameMap &inputs,
              const framework::VariableNameMap &outputs,
              const framework::AttributeMap &attrs)
      : NetOp(type, inputs, outputs, attrs) {
    auto out_grad = Input(framework::GradVarName("Out"));
    auto x_grad = Output(framework::GradVarName("X"));
    auto y_grad = Output(framework::GradVarName("Y"));

    // x_grad = out_grad
    AppendOp(framework::OpRegistry::CreateOp("identity", {{"X", {out_grad}}},
                                             {{"Y", {x_grad}}}, {}));

    framework::AttributeMap scale_attr;
    scale_attr["scale"] = static_cast<AttrType>(-1);
    AppendOp(framework::OpRegistry::CreateOp("scale", {{"X", {out_grad}}},
                                             {{"Out", {y_grad}}}, scale_attr));
    CompleteAddOp(false);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(minus, ops::MinusOp, ops::MinusOpMaker, minus_grad,
            ops::MinusGradOp<float>);
REGISTER_OP_CPU_KERNEL(minus,
                       ops::MinusKernel<paddle::platform::CPUPlace, float>);
