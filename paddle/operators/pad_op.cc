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

#include "paddle/operators/pad_op.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class PadOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    auto dim0 = ctx.Input<Tensor>("X")->dims();
    auto paddings = GetAttr<std::vector<std::pair<int, int>>>("paddings");
    std::vector<int> dim1(dim0.size());
    for (int i = 0; i < dim0.size(); ++i) {
      dim1[i] = dim0[i] + paddings[i].first + paddings[i].second;
    }
    ctx.Output<Tensor>("Out")->Resize(paddle::framework::make_ddim(dim1));
  }
};

class PadOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  PadOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "The input of pad op");
    AddOutput("Out", "The output of pad op");
    AddComment(R"DOC(
Pad Operator.
)DOC");
    AddAttr<std::vector<std::pair<int, int>>>(
        "paddings", "The padding rules for each dimension");
    AddAttr<float>("pad_value", "The value to be padded into tensor")
        .SetDefault(0.0f);
  }
};

class PadOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("X"), "Input(X) should not be null");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar(framework::GradVarName("Out")),
                            "Input(Out@GRAD) should not be null");
    auto x_dims = ctx.Input<Tensor>("X")->dims();
    auto *x_grad = ctx.Output<Tensor>(framework::GradVarName("X"));

    x_grad->Resize(x_dims);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(pad, ops::PadOp, ops::PadOpMaker, pad_grad, ops::PadOpGrad);
REGISTER_OP_CPU_KERNEL(pad, ops::PadKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(pad_grad,
                       ops::PadGradKernel<paddle::platform::CPUPlace, float>);
