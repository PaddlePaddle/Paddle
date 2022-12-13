/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <memory>
#include <string>

#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

class FusedBiasDropoutResidualLnOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(
        ctx->HasInput("X"), "Input", "X", "FusedBiasDropoutResidualLnOp");
    OP_INOUT_CHECK(ctx->HasOutput("LnMean"),
                   "Output",
                   "LnMean",
                   "FusedBiasDropoutResidualLnOp");
    OP_INOUT_CHECK(ctx->HasOutput("LnVariance"),
                   "Output",
                   "LnVariance",
                   "FusedBiasDropoutResidualLnOp");
    OP_INOUT_CHECK(ctx->HasOutput("BiasDropoutResidualOut"),
                   "Output",
                   "BiasDropoutResidualOut",
                   "FusedBiasDropoutResidualLnOp");
    OP_INOUT_CHECK(ctx->HasOutput("DropoutMaskOut"),
                   "Output",
                   "DropoutMaskOut",
                   "FusedBiasDropoutResidualLnOp");
    OP_INOUT_CHECK(
        ctx->HasOutput("Y"), "Output", "Y", "FusedBiasDropoutResidualLnOp");
    auto x_dim = ctx->GetInputDim("X");
    int left = 1;
    for (int i = 0; i < x_dim.size() - 1; i++) {
      left *= x_dim[i];
    }
    ctx->SetOutputDim("BiasDropoutResidualOut", ctx->GetInputDim("X"));
    if (ctx->Attrs().Get<bool>("is_test") == false) {
      ctx->SetOutputDim("DropoutMaskOut", ctx->GetInputDim("X"));
    }
    ctx->SetOutputDim("LnMean", {left});
    ctx->SetOutputDim("LnVariance", {left});
    ctx->SetOutputDim("Y", ctx->GetInputDim("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input = ctx.Input<phi::DenseTensor>("X");
    auto input_data_type = framework::TransToProtoVarType(input->dtype());
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
};

class FusedBiasDropoutResidualLnOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input tensor.");
    AddInput("Residual", "The residual tensor.");
    AddInput("Bias", "The linear bias tensor.").AsDispensable();
    AddInput("LnScale",
             "(optional) Scale is a 1-dimensional tensor of size "
             "H. Here, H represents the last dimension of its input tensor.")
        .AsDispensable();
    AddInput("LnBias",
             "(optional) Bias is a 1-dimensional tensor of size "
             "H. Here, H represents the last dimension of its input tensor.")
        .AsDispensable();
    AddOutput("BiasDropoutResidualOut", "Output of bias + dropout + residual.")
        .AsIntermediate();
    AddOutput("DropoutMaskOut", "The random sampled dropout mask.")
        .AsIntermediate();
    AddOutput("LnMean", "Mean of the current mini batch.").AsIntermediate();
    AddOutput("LnVariance", "Variance of the current mini batch.")
        .AsIntermediate();
    AddOutput("Y", "Result.");
    AddAttr<float>("dropout_rate", "Probability of setting units to zero.")
        .SetDefault(.5f)
        .AddCustomChecker([](const float &drop_p) {
          PADDLE_ENFORCE_EQ(drop_p >= 0.0f && drop_p <= 1.0f,
                            true,
                            platform::errors::InvalidArgument(
                                "'dropout_rate' must be between 0.0 and 1.0."));
        });
    AddAttr<bool>("is_test",
                  "(bool, default false) Set to true for inference only, false "
                  "for training. Some layers may run faster when this is true.")
        .SetDefault(false);
    AddAttr<bool>("dropout_fix_seed",
                  "A flag indicating whether to use a fixed seed to generate "
                  "random mask. NOTE: DO NOT set this flag to true in "
                  "training. Setting this flag to true is only useful in "
                  "unittest or for debug that always the same output units "
                  "will be dropped.")
        .SetDefault(true);
    AddAttr<int>("dropout_seed", "Dropout random seed.").SetDefault(0);
    AddAttr<std::string>(
        "dropout_implementation",
        "[\"downgrade_in_infer\"|\"upscale_in_train\"]"
        "The meaning is the same as 'attn_dropout_implementation'.")
        .SetDefault("downgrade_in_infer")
        .AddCustomChecker([](const std::string &type) {
          PADDLE_ENFORCE_EQ(
              type == "downgrade_in_infer" || type == "upscale_in_train",
              true,
              platform::errors::InvalidArgument(
                  "dropout_implementation can only be downgrade_in_infer or "
                  "upscale_in_train"));
        });
    AddAttr<float>("ln_epsilon",
                   "Constant for numerical stability [default 1e-5].")
        .SetDefault(1e-5)
        .AddCustomChecker([](const float &ln_epsilon) {
          PADDLE_ENFORCE_EQ(ln_epsilon >= 0.0f && ln_epsilon <= 0.001f,
                            true,
                            platform::errors::InvalidArgument(
                                "'epsilon' of the LayerNorm should be between "
                                "0.0 and 0.001, But received [%s].",
                                ln_epsilon));
        });

    AddComment(R"DOC(
    Add fused bias_dropout_residual_layer_norm op whose logic is as follows:
    // @input: [batch_size, seq_len, embed_dim]
    // @final_out: [batch_size, seq_len, embed_dim]
    y = layer_norm(residual + dropout(bias + x));
    )DOC");
  }
};

class FusedBiasDropoutResidualLnGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->Attrs().Get<bool>("is_test"),
                      false,
                      platform::errors::InvalidArgument(
                          "GradOp is only callable when is_test is false"));
    OP_INOUT_CHECK(ctx->HasInput("LnMean"),
                   "Input",
                   "LnMean",
                   "FusedBiasDropoutResidualLnGrad");
    OP_INOUT_CHECK(ctx->HasInput("LnVariance"),
                   "Input",
                   "LnVariance",
                   "FusedBiasDropoutResidualLnGrad");
    OP_INOUT_CHECK(ctx->HasInput("BiasDropoutResidualOut"),
                   "Input",
                   "BiasDropoutResidualOut",
                   "FusedBiasDropoutResidualLnGrad");
    if (ctx->HasOutput(framework::GradVarName("LnScale"))) {
      ctx->SetOutputDim(framework::GradVarName("LnScale"),
                        ctx->GetInputDim("LnScale"));
    }
    if (ctx->HasOutput(framework::GradVarName("LnBias"))) {
      ctx->SetOutputDim(framework::GradVarName("LnBias"),
                        ctx->GetInputDim("LnBias"));
    }
    if (ctx->HasOutput(framework::GradVarName("Residual"))) {
      ctx->SetOutputDim(framework::GradVarName("Residual"),
                        ctx->GetInputDim("Residual"));
    }
    if (ctx->HasOutput(framework::GradVarName("Bias"))) {
      ctx->SetOutputDim(framework::GradVarName("Bias"),
                        ctx->GetInputDim("Bias"));
    }
    if (ctx->HasOutput(framework::GradVarName("X"))) {
      ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
    }
    ctx->SetOutputDim(framework::GradVarName("BiasDropoutResidualOut"),
                      ctx->GetInputDim("BiasDropoutResidualOut"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input = ctx.Input<phi::DenseTensor>("X");
    auto input_data_type = framework::TransToProtoVarType(input->dtype());
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
};

template <typename T>
class FusedBiasDropoutResidualLnGradOpMaker
    : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("fused_bias_dropout_residual_layer_norm_grad");
    op->SetInput(framework::GradVarName("Y"), this->OutputGrad("Y"));
    op->SetInput("X", this->Input("X"));
    op->SetInput("Residual", this->Input("Residual"));
    if (this->HasInput("Bias")) {
      op->SetInput("Bias", this->Input("Bias"));
      op->SetOutput(framework::GradVarName("Bias"), this->InputGrad("Bias"));
    }
    if (this->HasInput("LnScale")) {
      op->SetInput("LnScale", this->Input("LnScale"));
      op->SetOutput(framework::GradVarName("LnScale"),
                    this->InputGrad("LnScale"));
    }
    if (this->HasInput("LnBias")) {
      op->SetInput("LnBias", this->Input("LnBias"));
      op->SetOutput(framework::GradVarName("LnBias"),
                    this->InputGrad("LnBias"));
    }
    if (this->HasOutput("LnMean")) {
      op->SetInput("LnMean", this->Output("LnMean"));
    }
    if (this->HasOutput("LnVariance")) {
      op->SetInput("LnVariance", this->Output("LnVariance"));
    }
    if (this->HasOutput("BiasDropoutResidualOut")) {
      op->SetInput("BiasDropoutResidualOut",
                   this->Output("BiasDropoutResidualOut"));
    }
    op->SetInput("DropoutMaskOut", this->Output("DropoutMaskOut"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("Residual"),
                  this->InputGrad("Residual"));
    op->SetOutput(framework::GradVarName("BiasDropoutResidualOut"),
                  this->OutputGrad("BiasDropoutResidualOut"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    fused_bias_dropout_residual_layer_norm,
    ops::FusedBiasDropoutResidualLnOp,
    ops::FusedBiasDropoutResidualLnOpMaker,
    ops::FusedBiasDropoutResidualLnGradOpMaker<paddle::framework::OpDesc>,
    ops::FusedBiasDropoutResidualLnGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(fused_bias_dropout_residual_layer_norm_grad,
                  ops::FusedBiasDropoutResidualLnGradOp);
