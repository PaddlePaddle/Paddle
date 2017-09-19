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

#include "paddle/operators/cos_sim_op.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class CosSimOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    // notnull check
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("X"),
                            "Input(X) of CosSimOp should not be null.");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("Y"),
                            "Input(Y) of CosSimOp should not be null.");
    PADDLE_ENFORCE_NOT_NULL(ctx.OutputVar("Out"),
                            "Output(Out) of CosSimOp should not be null.");
    PADDLE_ENFORCE_NOT_NULL(ctx.OutputVar("XNorm"),
                            "Output(XNorm) of CosSimOp should not be null.");
    PADDLE_ENFORCE_NOT_NULL(ctx.OutputVar("YNorm"),
                            "Output(YNorm) of CosSimOp should not be null.");

    // shape check
    auto x_dims = ctx.Input<Tensor>("X")->dims();
    auto y_dims = ctx.Input<Tensor>("Y")->dims();

    PADDLE_ENFORCE_EQ(x_dims.size(), y_dims.size(),
                      "Ranks of Input(X) and Input(Y) must be equal.");
    PADDLE_ENFORCE_GE(x_dims.size(), 2,
                      "Rank of Input(X) must not be less than 2.");
    PADDLE_ENFORCE_EQ(framework::slice_ddim(x_dims, 1, x_dims.size()),
                      framework::slice_ddim(y_dims, 1, y_dims.size()),
                      "All dimensions except the 1st of Input(X) and Input(Y) "
                      "must be equal.");
    PADDLE_ENFORCE(x_dims[0] == y_dims[0] || y_dims[0] == 1,
                   "The 1st dimension of Input(Y) must be equal to Input(X) or"
                   " just 1 (which will be broadcasted to match Input(X)).");

    // resize tensor
    ctx.Output<framework::LoDTensor>("Out")->Resize({x_dims[0], 1});
    ctx.Output<framework::LoDTensor>("XNorm")->Resize({x_dims[0], 1});
    ctx.Output<framework::LoDTensor>("YNorm")->Resize({y_dims[0], 1});
  }
};

class CosSimOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  CosSimOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "The 1st input of cos_sim op.");
    AddInput("Y", "The 2nd input of cos_sim op.");
    AddOutput("Out", "The output of cos_sim op.");
    AddOutput("XNorm",
              "Norm of the first input, reduced along the 1st "
              "dimension.")
        .AsIntermediate();
    AddOutput("YNorm",
              "Norm of the second input, reduced along the 1st "
              "dimension.")
        .AsIntermediate();

    AddComment(R"DOC(
Cosine Similarity Operator.

The equation is: Out = X^T * Y / (sqrt(X^T * X) * sqrt(Y^T * Y)).

Input(X) and Input(Y) must have the same shape, except that the 1st dimension
of Input(Y) could be just 1 (different from Input(X)), which will be
broadcasted to match the shape of Input(X) before computing their cosine
similarity.
)DOC");
  }
};

class CosSimOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    // notnull check
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("X"), "Input(X) must not be null.");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("Y"), "Input(Y) must not be null.");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("XNorm"),
                            "Input(XNorm) must not be null.");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("YNorm"),
                            "Input(YNorm) must not be null.");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("Out"),
                            "Input(Out) must not be null.");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar(framework::GradVarName("Out")),
                            "Input(Out@GRAD) must not be null.");

    // shape check
    auto x_dims = ctx.Input<Tensor>("X")->dims();
    auto y_dims = ctx.Input<Tensor>("Y")->dims();
    auto xnorm_dims = ctx.Input<Tensor>("XNorm")->dims();
    auto ynorm_dims = ctx.Input<Tensor>("YNorm")->dims();
    auto out_dims = ctx.Input<Tensor>("Out")->dims();
    auto out_grad_dims =
        ctx.Input<Tensor>(framework::GradVarName("Out"))->dims();

    PADDLE_ENFORCE_GE(x_dims.size(), y_dims.size(),
                      "Ranks of Input(X) and Input(Y) must be equal.");
    PADDLE_ENFORCE_GE(x_dims.size(), 2,
                      "Rank of Input(X) must not be less than 2.");
    PADDLE_ENFORCE_EQ(framework::slice_ddim(x_dims, 1, x_dims.size()),
                      framework::slice_ddim(y_dims, 1, y_dims.size()),
                      "All dimensions except the 1st of Input(X) and Input(Y) "
                      "must be equal.");
    PADDLE_ENFORCE(x_dims[0] == y_dims[0] || y_dims[0] == 1,
                   "The 1st dimension of Input(Y) must be equal to Input(X) or"
                   " just 1 (which will be broadcasted to match Input(X)).");
    auto target_xnorm_dims = framework::make_ddim({x_dims[0], 1});
    auto target_ynorm_dims = framework::make_ddim({y_dims[0], 1});
    PADDLE_ENFORCE_EQ(xnorm_dims, target_xnorm_dims,
                      "Shape of Input(XNorm) must be [X.Dim(0), 1].");
    PADDLE_ENFORCE_EQ(ynorm_dims, target_ynorm_dims,
                      "Shape of Input(YNorm) must be [Y.Dim(0), 1].");
    PADDLE_ENFORCE_EQ(out_dims, target_xnorm_dims,
                      "Shape of Input(Out) must be [X.Dim(0), 1].");
    PADDLE_ENFORCE_EQ(out_grad_dims, target_xnorm_dims,
                      "Shape of Input(Out@Grad) must be [X.Dim(0), 1].");

    // resize tensor
    auto *x_grad =
        ctx.Output<framework::LoDTensor>(framework::GradVarName("X"));
    auto *y_grad =
        ctx.Output<framework::LoDTensor>(framework::GradVarName("Y"));
    if (x_grad) x_grad->Resize(x_dims);
    if (y_grad) y_grad->Resize(y_dims);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(cos_sim, ops::CosSimOp, ops::CosSimOpMaker, cos_sim_grad,
            ops::CosSimOpGrad);
REGISTER_OP_CPU_KERNEL(cos_sim,
                       ops::CosSimKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    cos_sim_grad, ops::CosSimGradKernel<paddle::platform::CPUPlace, float>);
