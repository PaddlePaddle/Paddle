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

#include "paddle/operators/dropout_op.h"

namespace paddle {
namespace operators {

using framework::Tensor;
using framework::LoDTensor;

class DropoutOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("X"), "Input(X) must not be null.");
    PADDLE_ENFORCE_GE(ctx.Attr<float>("dropout_prob"), 0);
    PADDLE_ENFORCE_LE(ctx.Attr<float>("dropout_prob"), 1);
    // TODO(xinghai-sun): remove this check after swtiching to bool
    PADDLE_ENFORCE(ctx.Attr<int>("is_training") == 0 ||
                   ctx.Attr<int>("is_training") == 1);

    auto dims = ctx.Input<Tensor>("X")->dims();
    ctx.Output<LoDTensor>("Out")->Resize(dims);
    if (ctx.Attr<int>("is_training") == 1) {
      ctx.Output<LoDTensor>("Mask")->Resize(dims);
    }
  }
};

template <typename AttrType>
class DropoutOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  DropoutOpMaker(framework::OpProto *proto,
                 framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddAttr<AttrType>("dropout_prob", "Probability of setting units to zero.")
        .SetDefault(.5f);
    // TODO(xinghai-sun): use bool for is_training after bool is supported.
    AddAttr<int>("is_training", "Whether in training phase.").SetDefault(1);
    AddAttr<int>("seed", "Dropout random seed.").SetDefault(0);
    AddInput("X", "The input of dropout op.");
    AddOutput("Out", "The output of dropout op.");
    AddOutput("Mask", "The random sampled dropout mask.").AsIntermediate();

    AddComment(R"DOC(
Dropout Operator.

"Dropout" refers to randomly dropping out units in a nerual network. It is a
regularization technique for reducing overfitting by preventing neuron
co-adaption during training. The dropout operator randomly set (according to
the given dropout probability) the outputs of some units to zero, while others
being set to their inputs.
)DOC");
  }
};

template <typename AttrType>
class DropoutOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    PADDLE_ENFORCE_EQ(ctx.Attr<int>("is_training"), 1,
                      "GradOp is only callable when is_training is true");

    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("X"), "Input(X) must not be null.");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("Mask"), "Mask must not be null.");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar(framework::GradVarName("Out")),
                            "Input(Out@GRAD) must not be null.");

    PADDLE_ENFORCE_GE(ctx.Attr<AttrType>("dropout_prob"), 0);
    PADDLE_ENFORCE_LE(ctx.Attr<AttrType>("dropout_prob"), 1);
    // TODO(xinghai-sun): remove this check after swtiching to bool
    PADDLE_ENFORCE(ctx.Attr<int>("is_training") == 0 ||
                   ctx.Attr<int>("is_training") == 1);
    auto x_dims = ctx.Input<Tensor>("X")->dims();
    auto out_dims = ctx.Input<Tensor>(framework::GradVarName("Out"))->dims();
    PADDLE_ENFORCE_EQ(x_dims, out_dims,
                      "Dimensions of Input(X) and Out@Grad must be the same.");
    auto mask_dims = ctx.Input<Tensor>("Mask")->dims();
    PADDLE_ENFORCE_EQ(x_dims, mask_dims,
                      "Dimensions of Input(X) and Mask must be the same.");

    auto *x_grad = ctx.Output<LoDTensor>(framework::GradVarName("X"));
    x_grad->Resize(x_dims);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(dropout, ops::DropoutOp, ops::DropoutOpMaker<float>, dropout_grad,
            ops::DropoutOpGrad<float>);
REGISTER_OP_CPU_KERNEL(
    dropout, ops::CPUDropoutKernel<paddle::platform::CPUPlace, float, float>);
REGISTER_OP_CPU_KERNEL(
    dropout_grad, ops::DropoutGradKernel<paddle::platform::CPUPlace, float>);
