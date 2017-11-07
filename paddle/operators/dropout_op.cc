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

class DropoutOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) must not be null.");
    PADDLE_ENFORCE_GE(ctx->Attrs().Get<float>("dropout_prob"), 0);
    PADDLE_ENFORCE_LE(ctx->Attrs().Get<float>("dropout_prob"), 1);

    auto x_dims = ctx->GetInputDim("X");
    ctx->SetOutputDim("Out", x_dims);
    if (ctx->Attrs().Get<bool>("is_training") == true) {
      ctx->SetOutputDim("Mask", x_dims);
    }
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

template <typename AttrType>
class DropoutOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  DropoutOpMaker(framework::OpProto* proto,
                 framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "The input of dropout op.");
    AddOutput("Out", "The output of dropout op.");
    AddOutput("Mask", "The random sampled dropout mask.").AsIntermediate();

    AddAttr<float>("dropout_prob", "Probability of setting units to zero.")
        .SetDefault(.5f);
    AddAttr<bool>("is_training", "True if in training phase.").SetDefault(true);
    AddAttr<int>("seed", "Dropout random seed.").SetDefault(0);

    AddComment(R"DOC(
Dropout Operator.

Dropout refers to randomly dropping out units in a nerual network. It is a
regularization technique for reducing overfitting by preventing neuron
co-adaption during training. The dropout operator randomly set (according to
the given dropout probability) the outputs of some units to zero, while others
are set equal to their corresponding inputs.

)DOC");
  }
};

template <typename AttrType>
class DropoutOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->Attrs().Get<bool>("is_training"), true,
                      "GradOp is only callable when is_training is true");

    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) must not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Mask"), "Mask must not be null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) must not be null.");

    PADDLE_ENFORCE_GE(ctx->Attrs().Get<float>("dropout_prob"), 0);
    PADDLE_ENFORCE_LE(ctx->Attrs().Get<float>("dropout_prob"), 1);
    auto x_dims = ctx->GetInputDim("X");
    auto out_dims = ctx->GetInputDim(framework::GradVarName("Out"));
    PADDLE_ENFORCE_EQ(x_dims, out_dims,
                      "Dimensions of Input(X) and Out@Grad must be the same.");
    auto mask_dims = ctx->GetInputDim("Mask");
    PADDLE_ENFORCE_EQ(x_dims, mask_dims,
                      "Dimensions of Input(X) and Mask must be the same.");

    ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
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
