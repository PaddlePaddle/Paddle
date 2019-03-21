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

#include "paddle/fluid/operators/layer_norm_op.h"
#include <memory>

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using DataLayout = framework::DataLayout;

class LayerNormOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of LayerNormOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Y"),
                   "Output(Y) of LayerNormOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Mean"),
                   "Output(Mean) of LayerNormOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Variance"),
                   "Output(Variance) of LayerNormOp should not be null.");

    auto x_dim = ctx->GetInputDim("X");
    auto begin_norm_axis = ctx->Attrs().Get<int>("begin_norm_axis");
    PADDLE_ENFORCE_LT(begin_norm_axis, x_dim.size(),
                      "'begin_norm_axis' must be less than the rank of X.");

    auto matrix_dim = framework::flatten_to_2d(x_dim, begin_norm_axis);
    int left = static_cast<int>(matrix_dim[0]);
    int right = static_cast<int>(matrix_dim[1]);
    if (ctx->HasInput("Scale")) {
      PADDLE_ENFORCE_EQ(ctx->GetInputDim("Scale").size(), 1);
      PADDLE_ENFORCE_EQ(ctx->GetInputDim("Scale")[0], right);
    }
    if (ctx->HasInput("Bias")) {
      PADDLE_ENFORCE_EQ(ctx->GetInputDim("Bias").size(), 1);
      PADDLE_ENFORCE_EQ(ctx->GetInputDim("Bias")[0], right);
    }

    ctx->SetOutputDim("Y", ctx->GetInputDim("X"));
    ctx->SetOutputDim("Mean", {left});
    ctx->SetOutputDim("Variance", {left});
    ctx->ShareLoD("X", "Y");
  }
};

class LayerNormOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input tensor.");
    AddInput("Scale",
             "(optional) Scale is a 1-dimensional tensor of size "
             "H(`begin_norm_axis` splits the tensor(`X`) to a matrix [N,H])."
             "It is applied to the output.")
        .AsDispensable();
    AddInput("Bias",
             "(optional) Bias is a 1-dimensional tensor of size "
             "H(`begin_norm_axis` splits the tensor(`X`) to a matrix [N,H])."
             "It is applied to the output.")
        .AsDispensable();
    AddOutput("Y", "Result after normalization.");
    AddOutput("Mean", "Mean of the current mini batch.").AsIntermediate();
    AddOutput("Variance", "Variance of the current mini batch.")
        .AsIntermediate();

    AddAttr<float>("epsilon",
                   "Constant for numerical stability [default 1e-5].")
        .SetDefault(1e-5)
        .AddCustomChecker([](const float &epsilon) {
          PADDLE_ENFORCE(epsilon >= 0.0f && epsilon <= 0.001f,
                         "'epsilon' should be between 0.0 and 0.001.");
        });
    AddAttr<int>("begin_norm_axis",
                 "the axis of `begin_norm_axis ... Rank(X) - 1` will be "
                 "normalized. `begin_norm_axis` splits the tensor(`X`) to a "
                 "matrix [N,H]. [default 1].")
        .SetDefault(1)
        .AddCustomChecker([](const int &begin_norm_axis) {
          PADDLE_ENFORCE_GT(begin_norm_axis, 0,
                            "'begin_norm_axis' should be greater than zero.");
        });

    AddComment(R"DOC(
Assume feature vectors exist on dimensions
:attr:`begin_norm_axis ... rank(input)` and calculate the moment statistics
along these dimensions for each feature vector :math:`a` with size
:math:`H`, then normalize each feature vector using the corresponding
statistics. After that, apply learnable gain and bias on the normalized
tensor to scale and shift if :attr:`scale` and :attr:`shift` are set.

Refer to `Layer Normalization <https://arxiv.org/pdf/1607.06450v1.pdf>`_
)DOC");
  }
};

class LayerNormGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    // check input
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of LayerNormOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Mean"),
                   "Input(Mean) of LayerNormOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Variance"),
                   "Input(Variance) of LayerNormOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Y")),
                   "Input(Y@GRAD) of LayerNormOp should not be null.");

    // check output
    if (ctx->HasOutput(framework::GradVarName("X"))) {
      ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
    }
    if (ctx->HasOutput(framework::GradVarName("Scale"))) {
      ctx->SetOutputDim(framework::GradVarName("Scale"),
                        ctx->GetInputDim("Scale"));
    }
    if (ctx->HasOutput(framework::GradVarName("Bias"))) {
      ctx->SetOutputDim(framework::GradVarName("Bias"),
                        ctx->GetInputDim("Scale"));
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    const auto *var = ctx.InputVar(framework::GradVarName("Y"));
    if (var == nullptr) {
      PADDLE_THROW("can't find Y@GRAD");
    }
    const Tensor *t = nullptr;
    if (var->IsType<Tensor>()) {
      t = &var->Get<Tensor>();
    } else if (var->IsType<LoDTensor>()) {
      t = &var->Get<LoDTensor>();
    }
    if (t == nullptr) {
      PADDLE_THROW("can't find Y@GRAD");
    }
    return framework::OpKernelType(t->type(), ctx.GetPlace());
  }
};

class LayerNormGradOpDescMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    std::unique_ptr<framework::OpDesc> op(new framework::OpDesc());
    op->SetType("layer_norm_grad");
    op->SetInput("X", Input("X"));
    op->SetInput("Mean", Output("Mean"));
    op->SetInput("Variance", Output("Variance"));
    if (ForwardOp().Inputs().count("Scale") > 0) {
      op->SetInput("Scale", Input("Scale"));
      op->SetOutput(framework::GradVarName("Scale"), InputGrad("Scale"));
    }

    if (ForwardOp().Inputs().count("Bias") > 0) {
      op->SetOutput(framework::GradVarName("Bias"), InputGrad("Bias"));
    }

    op->SetInput(framework::GradVarName("Y"), OutputGrad("Y"));
    op->SetOutput(framework::GradVarName("X"), InputGrad("X"));
    op->SetAttrMap(Attrs());
    return op;
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(layer_norm, ops::LayerNormOp, ops::LayerNormOpMaker,
                  ops::LayerNormGradOpDescMaker);
REGISTER_OPERATOR(layer_norm_grad, ops::LayerNormGradOp);
REGISTER_OP_CPU_KERNEL(
    layer_norm, ops::LayerNormKernel<paddle::platform::CPUDeviceContext, float>,
    ops::LayerNormKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    layer_norm_grad,
    ops::LayerNormGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::LayerNormGradKernel<paddle::platform::CPUDeviceContext, double>);
