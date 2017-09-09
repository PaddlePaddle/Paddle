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

#include "paddle/operators/scaling_op.h"

namespace paddle {
namespace operators {

class ScalingOp : public framework::OperatorWithKernel {
 public:
  ScalingOp(const std::string &type, const framework::VariableNameMap &inputs,
            const framework::VariableNameMap &outputs,
            const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    auto *in = ctx.Input<framework::Tensor>("X");
    auto *weight = ctx.Input<framework::Tensor>("weight");
    PADDLE_ENFORCE_EQ(1, weight->dims().size(),
                      "The Input(weight) must be a vector");
    PADDLE_ENFORCE_EQ(2, in->dims().size(), "The Input(X) must be a matrix.");
    PADDLE_ENFORCE_EQ(in->dims()[0], weight->dims()[0],
                      "The rows' number of Input(X) must be equal to the size"
                      " of Input(weight).");
    auto *out = ctx.Output<framework::Tensor>("Out");
    out->Resize(in->dims());
  }
};

class ScalingOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  ScalingOpMaker(framework::OpProto *proto,
                 framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "The input tensor of scaling operator.");
    AddInput("weight", "The weight vector of scaling operator.");
    AddOutput("Out", "The output tensor of scaling operator.");
    AddComment(R"DOC(Scaling operator

The equation is: Out.row[i] = weight[i] * X.row[i]
)DOC");
  }
};

class ScalingGradOp : public framework::OperatorWithKernel {
 public:
  ScalingGradOp(const std::string &type,
                const framework::VariableNameMap &inputs,
                const framework::VariableNameMap &outputs,
                const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    auto in_dims = ctx.Input<framework::Tensor>("X")->dims();
    auto weight_dims = ctx.Input<framework::Tensor>("weight")->dims();
    ctx.Output<framework::Tensor>(framework::GradVarName("X"))->Resize(in_dims);
    ctx.Output<framework::Tensor>(framework::GradVarName("weight"))
        ->Resize(weight_dims);
  }
};

}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;

REGISTER_OP(scaling, ops::ScalingOp, ops::ScalingOpMaker, scaling_grad,
            ops::ScalingGradOp);
REGISTER_OP_CPU_KERNEL(scaling,
                       ops::ScalingKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    scaling_grad, ops::ScalingGradKernel<paddle::platform::CPUPlace, float>);
