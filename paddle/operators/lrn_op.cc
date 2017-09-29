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
  void InferShape(const framework::InferShapeContext &ctx) const override {
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("X"),
                            "Input(X) of LRNOp should not be null.");
    PADDLE_ENFORCE_NOT_NULL(ctx.OutputVar("Out"),
                            "Output(Out) of LRNOp should not be null.");
    PADDLE_ENFORCE_NOT_NULL(ctx.OutputVar("mid_out"),
                            "mid_out(Out) of LRNOp should not be null.");

    auto x_dim = ctx.Input<Tensor>("X")->dims();
    PADDLE_ENFORCE_EQ(x_dim.size(), 4, "Input(X)'rank of LRNOp should be 4.");

    ctx.Output<Tensor>("Out")->Resize(x_dim);
    ctx.Output<Tensor>("mid_out")->Resize(x_dim);
    ctx.ShareLoD("X", /*->*/ "Out");
  }
};

class LRNOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  LRNOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", R"DOC(
 Input of lrn op.It must be a 4 rank tenor with NCHW format.
 )DOC");

    AddOutput("Out", "The output of lrn op");
    AddOutput("mid_out", R"Doc(
Middle result of lrn op.It's computed in forward process 
and also used in backward process.
    )Doc");

    AddAttr<int>("n", R"DOC(
n is “adjacent” kernel maps at the same spatial position.
        )DOC")
        .SetDefault(5)
        .GreaterThan(0);

    AddAttr<float>("k", R"DOC(
k is the bias.
        )DOC")
        .SetDefault(2.0)
        .GreaterThan(0.0);

    AddAttr<float>("alpha", R"DOC(
alpha is the scale number.
        )DOC")
        .SetDefault(0.0001)
        .GreaterThan(0.0);

    AddAttr<float>("beta", R"DOC(
beta is the power number.
        )DOC")
        .SetDefault(0.75)
        .GreaterThan(0.0);

    AddComment(R"DOC(
 Local Response Normalization.

 This Function comes from the paper
 "ImageNet Classification with Deep Convolutional Neural Networks".

 The original formula is:

                                Input(i, x, y)
 Output(i, x, y) = ----------------------------------------------
                                 -- upper
                    (k + alpha * >  (Input(j, x, y))^2) ^ (beta)
                                 -- j = lower

 upper is `min(C, c + n/2)`
 lower if `max(0, c - n/2)`

 Function implementation:

 inputs and outpus is NCHW format, while input.shape.ndims() is equal 4.
 And the meaning of each dimension(0-3) is respectively batch size,
 feature maps, rows and columns.

 Input and Output in the above formula is for each map(i) of one image, and
 Input(i, x, y), Output(i, x, y) represents an element in an image.

 C is the number of feature maps of one image, and n is a hyper-parameters
 is configured when Function is initialized. The sum in the denominator
 is the sum of the same position in the neighboring maps.
    )DOC");
  }
};

class LRNOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("X"), "Input(X) should not be null");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar(framework::GradVarName("mid_out")),
                            "Input(mid_out@GRAD) should not be null");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar(framework::GradVarName("Out")),
                            "Input(Out@GRAD) should not be null");

    auto x_dims = ctx.Input<Tensor>("X")->dims();
    auto *x_g = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    x_g->Resize(x_dims);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(lrn, ops::LRNOp, ops::LRNOpMaker, lrn_grad, ops::LRNOpGrad);
REGISTER_OP_CPU_KERNEL(lrn, ops::LRNKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(lrn_grad,
                       ops::LRNGradKernel<paddle::platform::CPUPlace, float>);
