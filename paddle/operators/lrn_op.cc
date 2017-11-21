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

#include "paddle/operators/lrn_op.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class LRNOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) of LRNOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of LRNOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("MidOut"),
                   "MidOut(Out) of LRNOp should not be null.");

    auto x_dim = ctx->GetInputDim("X");
    PADDLE_ENFORCE_EQ(x_dim.size(), 4, "Input(X)'rank of LRNOp should be 4.");

    ctx->SetOutputDim("Out", x_dim);
    ctx->SetOutputDim("MidOut", x_dim);
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

template <typename T>
class LRNOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  LRNOpMaker(framework::OpProto* proto, framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X",
             "(Tensor) The input of LRN operator. "
             "It must be a 4D tenor with NCHW format.");
    AddOutput("Out",
              "(Tensor) The output of LRN operator, which is also the 4D "
              "tensor with NCHW format.");
    AddOutput("MidOut",
              "(Tensor) Middle result of LRN operator. It's computed in "
              "forward process and also used in backward process.");

    AddAttr<int>("n",
                 "(int default 5) "
                 "n is the \"adjacent\" kernel that maps "
                 "at the same spatial position.")
        .SetDefault(5)
        .GreaterThan(0);

    AddAttr<T>("k",
               "(float, default 2.0) "
               "k is the bias.")
        .SetDefault(2.0)
        .GreaterThan(0.0);

    AddAttr<T>("alpha",
               "(float, default 0.0001) "
               "alpha is the scale number.")
        .SetDefault(0.0001)
        .GreaterThan(0.0);

    AddAttr<T>("beta",
               "(float, default 0.75) "
               "beta is the power number.")
        .SetDefault(0.75)
        .GreaterThan(0.0);

    AddComment(R"DOC(
Local Response Normalization Operator.

This operator comes from the paper
"ImageNet Classification with Deep Convolutional Neural Networks".

The original formula is:

$$
Output(i, x, y) = Input(i, x, y) / \left(
k + \alpha \sum\limits^{\min(C, c + n/2)}_{j = \max(0, c - n/2)}
(Input(j, x, y))^2
\right)^{\beta}
$$

Function implementation:

Inputs and outpus are in NCHW format, while input.shape.ndims() equals 4.
And dimensions 0 ~ 3 represent batch size, feature maps, rows,
and columns, respectively.

Input and Output in the formula above is for each map(i) of one image, and
Input(i, x, y), Output(i, x, y) represents an element in an image.

C is the number of feature maps of one image. n is a hyper-parameter
configured when operator is initialized. The sum in the denominator
is the sum of the same positions in the neighboring maps.
    
)DOC");
  }
};

class LRNOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("MidOut")),
                   "Input(MidOut@GRAD) should not be null");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) should not be null");

    auto x_dims = ctx->GetInputDim("X");
    ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(lrn, ops::LRNOp, ops::LRNOpMaker<float>, lrn_grad, ops::LRNOpGrad);
REGISTER_OP_CPU_KERNEL(lrn, ops::LRNKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(lrn_grad,
                       ops::LRNGradKernel<paddle::platform::CPUPlace, float>);
