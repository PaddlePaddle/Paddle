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

#include "paddle/fluid/operators/cos_sim_op.h"
#include <memory>

namespace paddle {
namespace operators {

using framework::Tensor;

class CosSimOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    // notnull check
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "CosSim");
    OP_INOUT_CHECK(ctx->HasInput("Y"), "Input", "Y", "CosSim");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "CosSim");
    OP_INOUT_CHECK(ctx->HasOutput("XNorm"), "Output", "XNorm", "CosSim");
    OP_INOUT_CHECK(ctx->HasOutput("YNorm"), "Output", "YNorm", "CosSim");

    // shape check
    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");

    bool check = true;
    if ((!ctx->IsRuntime()) &&
        (phi::product(x_dims) <= 0 || phi::product(y_dims) <= 0)) {
      check = false;
    }

    if (check) {
      PADDLE_ENFORCE_EQ(
          x_dims.size(), y_dims.size(),
          platform::errors::InvalidArgument(
              "ShapeError: Ranks of Input(X) and Input(Y) must be equal."
              "But received: Ranks of Input(X) is [%d], Ranks of Input(Y) is "
              "[%d]",
              x_dims.size(), y_dims.size()));
      PADDLE_ENFORCE_GE(
          x_dims.size(), 2,
          platform::errors::InvalidArgument(
              "ShapeError: Rank of Input(X) must not be less than 2."
              "But received: Ranks of Input(X) is [%d]",
              x_dims.size()));
      PADDLE_ENFORCE_EQ(
          phi::slice_ddim(x_dims, 1, x_dims.size()),
          phi::slice_ddim(y_dims, 1, y_dims.size()),
          platform::errors::InvalidArgument(
              "All dimensions except the 1st of Input(X) and Input(Y) "
              "must be equal."));
      PADDLE_ENFORCE_EQ(
          x_dims[0] == y_dims[0] || y_dims[0] == 1, true,
          platform::errors::InvalidArgument(
              "The 1st dimension of Input(Y) %d must be equal to Input(X) %d or"
              " just 1 (which will be broadcasted to match Input(X)).",
              y_dims[0], x_dims[0]));
    }

    // resize tensor
    ctx->SetOutputDim("Out", {x_dims[0], 1});
    ctx->SetOutputDim("XNorm", {x_dims[0], 1});
    ctx->SetOutputDim("YNorm", {y_dims[0], 1});
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

class CosSimOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "The 1st input of cos_sim op, Tensor with shape ``[N_1, N_2, "
             "..., N_k]``, the data type is float32.");
    AddInput("Y",
             "The 2nd input of cos_sim op, Tensor with shape ``[N_1 or 1, N_2, "
             "..., N_k]``, the data type is float32.");
    AddOutput("Out", "The output of cos_sim op.");
    AddOutput("XNorm",
              "Norm of the first input, reduced along the 1st "
              "dimension.")
        .AsIntermediate();
    AddOutput("YNorm",
              "Norm of the second input, reduced along the 1st "
              "dimension.")
        .AsIntermediate();
    AddAttr<bool>(framework::kAllKernelsMustComputeRuntimeShape,
                  "Skip calling InferShape() function in the runtime.")
        .SetDefault(true);

    AddComment(R"DOC(
**Cosine Similarity Operator**

$Out = \frac{X^T * Y}{(\sqrt{X^T * X} * \sqrt{Y^T * Y})}$

The input X and Y must have the same shape, except that the 1st dimension
of input Y could be just 1 (different from input X), which will be
broadcasted to match the shape of input X before computing their cosine
similarity.

)DOC");
  }
};

class CosSimOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    // notnull check
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "CosSimGrad");
    OP_INOUT_CHECK(ctx->HasInput("Y"), "Input", "Y", "CosSimGrad");
    OP_INOUT_CHECK(ctx->HasInput("XNorm"), "Input", "XNorm", "CosSimGrad");
    OP_INOUT_CHECK(ctx->HasInput("YNorm"), "Input", "YNorm", "CosSimGrad");
    OP_INOUT_CHECK(ctx->HasInput("Out"), "Input", "Out", "CosSimGrad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   framework::GradVarName("Out"), "CosSimGrad");

    // shape check
    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");
    auto xnorm_dims = ctx->GetInputDim("XNorm");
    auto ynorm_dims = ctx->GetInputDim("YNorm");
    auto out_dims = ctx->GetInputDim("Out");
    auto out_grad_dims = ctx->GetInputDim(framework::GradVarName("Out"));

    PADDLE_ENFORCE_GE(
        x_dims.size(), y_dims.size(),
        platform::errors::InvalidArgument(
            "ShapeError: Ranks of Input(X) and Input(Y) must be equal."
            "But received: Ranks of Input(X) is [%d], Ranks of Input(Y) is "
            "[%d]",
            x_dims.size(), y_dims.size()));
    PADDLE_ENFORCE_GE(
        x_dims.size(), 2,
        platform::errors::InvalidArgument(
            "ShapeError: Rank of Input(X) must not be less than 2."
            "But received: Ranks of Input(X) is [%d]",
            x_dims.size()));
    PADDLE_ENFORCE_EQ(
        phi::slice_ddim(x_dims, 1, x_dims.size()),
        phi::slice_ddim(y_dims, 1, y_dims.size()),
        platform::errors::InvalidArgument(
            "All dimensions except the 1st of Input(X) [%s] and Input(Y) [%s] "
            "must be equal.",
            x_dims, y_dims));
    PADDLE_ENFORCE_EQ(
        true, x_dims[0] == y_dims[0] || y_dims[0] == 1,
        platform::errors::InvalidArgument(
            "The 1st dimension of Input(Y) %d must be equal to Input(X) %d or"
            " just 1 (which will be broadcasted to match Input(X)).",
            y_dims[0], x_dims[0]));
    auto target_xnorm_dims = phi::make_ddim({x_dims[0], 1});
    auto target_ynorm_dims = phi::make_ddim({y_dims[0], 1});
    PADDLE_ENFORCE_EQ(
        xnorm_dims, target_xnorm_dims,
        platform::errors::InvalidArgument(
            "Shape of Input(XNorm) [%s] must be (X.Dim(0), 1) - [%s]",
            xnorm_dims, target_xnorm_dims));
    PADDLE_ENFORCE_EQ(
        ynorm_dims, target_ynorm_dims,
        platform::errors::InvalidArgument(
            "Shape of Input(YNorm) [%s] must be (Y.Dim(0), 1) - [%s]",
            ynorm_dims, target_ynorm_dims));
    PADDLE_ENFORCE_EQ(
        out_dims, target_xnorm_dims,
        platform::errors::InvalidArgument(
            "Shape of Input(Out) [%s] must be (X.Dim(0), 1) - [%s]", out_dims,
            target_xnorm_dims));
    PADDLE_ENFORCE_EQ(
        out_grad_dims, target_xnorm_dims,
        platform::errors::InvalidArgument(
            "Shape of Input(Out@Grad) [%s] must be (X.Dim(0), 1) - [%s]",
            out_grad_dims, target_xnorm_dims));

    // resize tensor
    auto x_grad_name = framework::GradVarName("X");
    auto y_grad_name = framework::GradVarName("Y");
    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, x_dims);
    }
    if (ctx->HasOutput(y_grad_name)) {
      ctx->SetOutputDim(y_grad_name, y_dims);
    }
  }
};

template <typename T>
class CosSimGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("cos_sim_grad");
    grad_op->SetInput("X", this->Input("X"));
    grad_op->SetInput("Y", this->Input("Y"));
    grad_op->SetInput("XNorm", this->Output("XNorm"));
    grad_op->SetInput("YNorm", this->Output("YNorm"));
    grad_op->SetInput("Out", this->Output("Out"));
    grad_op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    grad_op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    grad_op->SetOutput(framework::GradVarName("Y"), this->InputGrad("Y"));
    grad_op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(cos_sim, ops::CosSimOp, ops::CosSimOpMaker,
                  ops::CosSimGradOpMaker<paddle::framework::OpDesc>,
                  ops::CosSimGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(cos_sim_grad, ops::CosSimOpGrad);
REGISTER_OP_CPU_KERNEL(
    cos_sim, ops::CosSimKernel<paddle::platform::CPUDeviceContext, float>);
REGISTER_OP_CPU_KERNEL(
    cos_sim_grad,
    ops::CosSimGradKernel<paddle::platform::CPUDeviceContext, float>);
