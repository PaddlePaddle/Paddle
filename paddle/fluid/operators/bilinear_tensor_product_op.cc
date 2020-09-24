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
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("X"), true,
        platform::errors::InvalidArgument("Input(X) should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Y"), true,
        platform::errors::InvalidArgument("Input(Y) should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Weight"), true,
        platform::errors::InvalidArgument("Input(Weight) should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("Out"), true,
        platform::errors::InvalidArgument("Output(Out) should not be null."));
    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");
    auto weight_dims = ctx->GetInputDim("Weight");

    PADDLE_ENFORCE_EQ(
        x_dims.size(), 2UL,
        platform::errors::InvalidArgument("The input(X) must be a 2D Tensor."));
    PADDLE_ENFORCE_EQ(
        y_dims.size(), 2UL,
        platform::errors::InvalidArgument("The input(Y) must be a 2D Tensor."));
    PADDLE_ENFORCE_EQ(weight_dims.size(), 3UL,
                      platform::errors::InvalidArgument(
                          "The input(Weight) must be a 3D tensor."));
    if (ctx->IsRuntime() || (x_dims[0] > 0 && y_dims[0] > 0)) {
      PADDLE_ENFORCE_EQ(
          x_dims[0], y_dims[0],
          platform::errors::InvalidArgument(
              "The first dimension(batch_size) of input(X) must be "
              "equal to the first dimension of the input(Y)."));
    }
    PADDLE_ENFORCE_EQ(x_dims[1], weight_dims[1],
                      platform::errors::InvalidArgument(
                          "The second dimension of input(X) must be equal to "
                          "the second dimension of the input(Weight)."));
    PADDLE_ENFORCE_EQ(y_dims[1], weight_dims[2],
                      platform::errors::InvalidArgument(
                          "The second dimension of input(Y) must be equal to "
                          "the third dimension of the input(Weight)."));

    if (ctx->HasInput("Bias")) {
      auto bias_dims = ctx->GetInputDim("Bias");
      PADDLE_ENFORCE_EQ(bias_dims.size(), 2UL,
                        platform::errors::InvalidArgument(
                            "The Input(Bias) must be a 2-D tensor with "
                            "the 2nd dimension fixed to 1 (a row vector)."));
      PADDLE_ENFORCE_EQ(bias_dims[0], 1UL,
                        platform::errors::InvalidArgument(
                            "The Input(Bias) must be a 2-D tensor with "
                            "the 2nd dimension fixed to 1 (a row vector)."));
      PADDLE_ENFORCE_EQ(bias_dims[1], weight_dims[0],
                        platform::errors::InvalidArgument(
                            "The second dimension of input(Bias) must be equal "
                            "to the first dimension of the input(Weight)."));
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
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("X"), true,
        platform::errors::InvalidArgument("Input(X) should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Y"), true,
        platform::errors::InvalidArgument("Input(Y) should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Weight"), true,
        platform::errors::InvalidArgument("Input(Weight) should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasInput(framework::GradVarName("Out")), true,
                      platform::errors::InvalidArgument(
                          "Input(Out@GRAD) should not be null."));
    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");
    auto weight_dims = ctx->GetInputDim("Weight");
    auto out_dims = ctx->GetInputDim(framework::GradVarName("Out"));

    PADDLE_ENFORCE_EQ(out_dims.size(), 2UL,
                      platform::errors::InvalidArgument(
                          "The input(Out@GRAD) must be a 2D Tensor."));
    PADDLE_ENFORCE_EQ(
        x_dims[0], out_dims[0],
        platform::errors::InvalidArgument(
            "The first dimension(batch_size) of input(Out@GRAD) must be "
            "equal to the first dimension of the Input(X)."));
    PADDLE_ENFORCE_EQ(
        weight_dims[0], out_dims[1],
        platform::errors::InvalidArgument(
            "The second dimension of input(Out@GRAD) must be equal to "
            "the third dimension of the Input(Weight)."));

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

template <typename T>
class BilinearTensorProductGradOpMaker
    : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("bilinear_tensor_product_grad");
    op->SetAttrMap(this->Attrs());
    op->SetInput("X", this->Input("X"));
    op->SetInput("Y", this->Input("Y"));
    op->SetInput("Weight", this->Input("Weight"));
    if (this->HasInput("Bias")) {
      op->SetOutput(framework::GradVarName("Bias"), this->InputGrad("Bias"));
    }

    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("Y"), this->InputGrad("Y"));
    op->SetOutput(framework::GradVarName("Weight"), this->InputGrad("Weight"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    bilinear_tensor_product, ops::BilinearTensorProductOp,
    ops::BilinearTensorProductOpMaker,
    ops::BilinearTensorProductGradOpMaker<paddle::framework::OpDesc>,
    ops::BilinearTensorProductGradOpMaker<paddle::imperative::OpBase>);
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
