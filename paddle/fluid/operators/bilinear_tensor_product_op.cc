/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/bilinear_tensor_product_op.h"
#include <memory>
#include <string>
#include <vector>

namespace paddle {
namespace operators {

using framework::Tensor;

class BilinearTensorProductOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Y"), "Input(Y) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Weight"),
                   "Input(Weight) should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"), "Output(Out) should not be null.");
    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");
    auto weight_dims = ctx->GetInputDim("Weight");

    PADDLE_ENFORCE_EQ(x_dims.size(), 2UL, "The input(X) must be a 2D Tensor.");
    PADDLE_ENFORCE_EQ(y_dims.size(), 2UL, "The input(Y) must be a 2D Tensor.");
    PADDLE_ENFORCE_EQ(weight_dims.size(), 3UL,
                      "The input(Weight) must be a 3D tensor.");
    PADDLE_ENFORCE_EQ(x_dims[0], y_dims[0],
                      "The first dimension(batch_size) of input(X) must be "
                      "equal to the first dimension of the input(Y).");
    PADDLE_ENFORCE_EQ(x_dims[1], weight_dims[1],
                      "The second dimension of input(X) must be equal to "
                      "the second dimension of the input(Weight).");
    PADDLE_ENFORCE_EQ(y_dims[1], weight_dims[2],
                      "The second dimension of input(Y) must be equal to "
                      "the third dimension of the input(Weight).");

    if (ctx->HasInput("Bias")) {
      auto bias_dims = ctx->GetInputDim("Bias");
      PADDLE_ENFORCE(bias_dims.size() == 2UL && bias_dims[0] == 1UL,
                     "The Input(Bias) must be a 2-D tensor with "
                     "the 2nd dimension fixed to 1 (a row vector).");
      PADDLE_ENFORCE_EQ(bias_dims[1], weight_dims[0],
                        "The second dimension of input(Bias) must be equal "
                        "to the first dimension of the input(Weight).");
    }

    ctx->SetOutputDim("Out", {x_dims[0], weight_dims[0]});
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

class BilinearTensorProductOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The first input of bilinear_tensor_product operator.");
    AddInput("Y", "The second input of bilinear_tensor_product operator.");
    AddInput("Weight",
             "The learnable parameters of bilinear_tensor_product operator.");
    AddInput("Bias", "The learnable bias of bilinear_tensor_product operator.")
        .AsDispensable();
    AddOutput("Out", "The output of bilinear_tensor_product operator.");
    AddComment(R"DOC(
Bilinear Tensor Product operator.
Given input X and Y, a 3D tensor Weight and a Bias. Each column of the
Output is computed by one slice $i = 1, . . . , k$ of the tensor:

$$
M =  (X W_i) * Y \\
Out_i = \sum_j {M_j} + Bias_i
$$

Where $W_i$ is the $i$-th slice of Input(Weight);
      $M_j$ is the $j$-th column of $M$;
      $Out_i$ is the $i$-th column of Output(Out);
      $Bias_i$ is a column vector, each element of it is equal to
        the $i$-th element of $Bias$;

)DOC");
  }
};

class BilinearTensorProductOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Y"), "Input(Y) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Weight"),
                   "Input(Weight) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) should not be null.");
    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");
    auto weight_dims = ctx->GetInputDim("Weight");
    auto out_dims = ctx->GetInputDim(framework::GradVarName("Out"));

    PADDLE_ENFORCE_EQ(out_dims.size(), 2UL,
                      "The input(Out@GRAD) must be a 2D Tensor.");
    PADDLE_ENFORCE_EQ(
        x_dims[0], out_dims[0],
        "The first dimension(batch_size) of input(Out@GRAD) must be "
        "equal to the first dimension of the Input(X).");
    PADDLE_ENFORCE_EQ(
        weight_dims[0], out_dims[1],
        "The second dimension of input(Out@GRAD) must be equal to "
        "the third dimension of the Input(Weight).");

    auto bias_grad_name = framework::GradVarName("Bias");
    if (ctx->HasOutput(bias_grad_name)) {
      ctx->SetOutputDim(bias_grad_name, {1, out_dims[1]});
    }

    auto x_grad_name = framework::GradVarName("X");
    auto y_grad_name = framework::GradVarName("Y");
    auto weight_grad_name = framework::GradVarName("Weight");

    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, x_dims);
    }
    if (ctx->HasOutput(y_grad_name)) {
      ctx->SetOutputDim(y_grad_name, y_dims);
    }
    if (ctx->HasOutput(weight_grad_name)) {
      ctx->SetOutputDim(weight_grad_name, weight_dims);
    }
  }
};

class BilinearTensorProductGradOpDescMaker
    : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    std::unique_ptr<framework::OpDesc> op(new framework::OpDesc());
    op->SetType("bilinear_tensor_product_grad");
    op->SetAttrMap(Attrs());
    op->SetInput("X", Input("X"));
    op->SetInput("Y", Input("Y"));
    op->SetInput("Weight", Input("Weight"));
    if (ForwardOp().Inputs().count("Bias") > 0) {
      op->SetOutput(framework::GradVarName("Bias"), InputGrad("Bias"));
    }

    op->SetOutput(framework::GradVarName("X"), InputGrad("X"));
    op->SetOutput(framework::GradVarName("Y"), InputGrad("Y"));
    op->SetOutput(framework::GradVarName("Weight"), InputGrad("Weight"));
    op->SetInput(framework::GradVarName("Out"), OutputGrad("Out"));

    return op;
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(bilinear_tensor_product, ops::BilinearTensorProductOp,
                  ops::BilinearTensorProductOpMaker,
                  ops::BilinearTensorProductGradOpDescMaker);
REGISTER_OPERATOR(bilinear_tensor_product_grad,
                  ops::BilinearTensorProductOpGrad);
REGISTER_OP_CPU_KERNEL(
    bilinear_tensor_product,
    ops::BilinearTensorProductKernel<paddle::platform::CPUDeviceContext, float>,
    ops::BilinearTensorProductKernel<paddle::platform::CPUDeviceContext,
                                     double>);
REGISTER_OP_CPU_KERNEL(
    bilinear_tensor_product_grad,
    ops::BilinearTensorProductGradKernel<paddle::platform::CPUDeviceContext,
                                         float>,
    ops::BilinearTensorProductGradKernel<paddle::platform::CPUDeviceContext,
                                         double>);
