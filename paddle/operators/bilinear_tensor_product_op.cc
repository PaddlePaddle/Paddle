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

#include "paddle/operators/bilinear_tensor_product_op.h"

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

    PADDLE_ENFORCE_EQ(x_dims.size(), 2, "The input X must be a 2D Tensor.");
    PADDLE_ENFORCE_EQ(y_dims.size(), 2, "The input Y must be a 2D Tensor.");
    PADDLE_ENFORCE_EQ(weight_dims.size(), 3,
                      "The input Weight must be a 3D tensor.");
    PADDLE_ENFORCE_GT(weight_dims[0], 0,
                      "The first dimension of Weight must be larger than 0.");
    PADDLE_ENFORCE_GT(weight_dims[1], 0,
                      "The second dimension of Weight must be larger than 0.");
    PADDLE_ENFORCE_GT(weight_dims[2], 0,
                      "The third dimension of Weight must be larger than 0.");
    PADDLE_ENFORCE_EQ(x_dims[0], y_dims[0],
                      "The first dimension(batch_size) of X must be "
                      "equal with the first dimension of the Y.");
    PADDLE_ENFORCE_EQ(x_dims[1], weight_dims[1],
                      "The second dimension of X must be equal with the second "
                      "dimension of the Weight.");
    PADDLE_ENFORCE_EQ(y_dims[1], weight_dims[2],
                      "The second dimension of Y must be equal with the third "
                      "dimension of the Weight.");

    if (ctx->HasInput("Bias")) {
      auto bias_dims = ctx->GetInputDim("Bias");
      PADDLE_ENFORCE_EQ(bias_dims.size(), 2,
                        "The input Bias must have 2 dimensions.");
      PADDLE_ENFORCE_EQ(bias_dims[0], 1,
                        "The first dimention of input Bias must be 1.");
      PADDLE_ENFORCE_EQ(bias_dims[1], weight_dims[0],
                        "The second dimension of Bias must be equal with the  "
                        "first dimension of the Weight.");
    }

    ctx->SetOutputDim("Out", {x_dims[0], weight_dims[0]});
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

class BilinearTensorProductOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  BilinearTensorProductOpMaker(framework::OpProto* proto,
                               framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "The first input of BilinearTensorProduct op");
    AddInput("Y", "The second input of BilinearTensorProduct op");
    AddInput("Weight", "The input weight of BilinearTensorProduct op");
    AddInput("Bias", "The input bias of BilinearTensorProduct op")
        .AsDispensable();
    AddOutput("Out", "The output of BilinearTensorProduct op");
    AddComment(R"DOC(
Bilinear Tensor Product operator.
Given input X and Y, a 3D tensor weight, and bias. Each column of the
output is computed by one slice i = 1, . . . , k of the tensor:

    M =  (X W_i) \cdot Y
    Out_i = \sum_i {M_i} + Bias_i

)DOC");
  }
};

class BilinearTensorProductOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null");
    PADDLE_ENFORCE(ctx->HasInput("Y"), "Input(Y) should not be null");
    PADDLE_ENFORCE(ctx->HasInput("Weight"), "Input(Weight) should not be null");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input (Out@GRAD) should not be null");
    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");
    auto weight_dims = ctx->GetInputDim("Weight");
    auto out_dims = ctx->GetInputDim(framework::GradVarName("Out"));

    PADDLE_ENFORCE_EQ(out_dims.size(), 2, "The Out@GRAD must be a 2D Tensor.");
    PADDLE_ENFORCE_EQ(
        x_dims[0], out_dims[0],
        "The first dimension(batch_size) of Out@GRAD must be equal with "
        "the first dimension of the X.");
    PADDLE_ENFORCE_EQ(weight_dims[0], out_dims[1],
                      "The second dimension of Out@GRAD must be equal with "
                      "the third dimension of the Weight.");

    if (ctx->HasInput("Bias")) {
      auto bias_dims = ctx->GetInputDim("Bias");
      PADDLE_ENFORCE_EQ(bias_dims[1], out_dims[1],
                        "The second dimension of Bias must be equal with "
                        "the second dimension of the Out@GRAD.");
      auto bias_grad_name = framework::GradVarName("Bias");
      if (ctx->HasOutput(bias_grad_name))
        ctx->SetOutputDim(bias_grad_name, bias_dims);
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

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(bilinear_tensor_product, ops::BilinearTensorProductOp,
            ops::BilinearTensorProductOpMaker, bilinear_tensor_product_grad,
            ops::BilinearTensorProductOpGrad);
REGISTER_OP_CPU_KERNEL(
    bilinear_tensor_product,
    ops::BilinearTensorProductKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    bilinear_tensor_product_grad,
    ops::BilinearTensorProductGradKernel<paddle::platform::CPUPlace, float>);
