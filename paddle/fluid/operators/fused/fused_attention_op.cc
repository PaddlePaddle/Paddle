/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/fused/fused_attention_op.h"

#include <memory>
#include <string>

#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

class FusedAttentionOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    // std::cout << "i am in op infershape\n";

    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "FusedAttentionOp");
    OP_INOUT_CHECK(ctx->HasInput("SrcMask"), "Input", "SrcMask",
                   "FusedAttentionOp");
    OP_INOUT_CHECK(ctx->HasInput("QKVW"), "Input", "QKVW", "FusedAttentionOp");
    OP_INOUT_CHECK(ctx->HasInput("QKVBias"), "Input", "QKVBias",
                   "FusedAttentionOp");
    OP_INOUT_CHECK(ctx->HasInput("OutLinearW"), "Input", "OutLinearW",
                   "FusedAttentionOp");
    OP_INOUT_CHECK(ctx->HasInput("OutLinearBias"), "Input", "OutLinearBias",
                   "FusedAttentionOp");

    // qkv_out: [batch_size, seq_len, 3, num_head, dim_head]
    OP_INOUT_CHECK(ctx->HasOutput("LnMean"), "Output", "LnMean",
                   "FusedAttentionOp");
    OP_INOUT_CHECK(ctx->HasOutput("LnVariance"), "Output", "LnVariance",
                   "FusedAttentionOp");
    OP_INOUT_CHECK(ctx->HasOutput("LnOut"), "Output", "LnOut",
                   "FusedAttentionOp");
    OP_INOUT_CHECK(ctx->HasOutput("QKVOut"), "Output", "QKVOut",
                   "FusedAttentionOp");
    OP_INOUT_CHECK(ctx->HasOutput("QKVBiasOut"), "Output", "QKVBiasOut",
                   "FusedAttentionOp");
    OP_INOUT_CHECK(ctx->HasOutput("TransposeOut2"), "Output", "TransposeOut2",
                   "FusedAttentionOp");
    OP_INOUT_CHECK(ctx->HasOutput("QKOut"), "Output", "QKOut",
                   "FusedAttentionOp");
    OP_INOUT_CHECK(ctx->HasOutput("QKTVOut"), "Output", "QKTVOut",
                   "FusedAttentionOp");
    OP_INOUT_CHECK(ctx->HasOutput("SrcMaskOut"), "Output", "SrcMaskOut",
                   "FusedAttentionOp");
    OP_INOUT_CHECK(ctx->HasOutput("SoftmaxOut"), "Output", "SoftmaxOut",
                   "FusedAttentionOp");
    OP_INOUT_CHECK(ctx->HasOutput("AttnDropoutMaskOut"), "Output",
                   "AttnDropoutMaskOut", "FusedAttentionOp");
    OP_INOUT_CHECK(ctx->HasOutput("AttnDropoutOut"), "Output", "AttnDropoutOut",
                   "FusedAttentionOp");
    OP_INOUT_CHECK(ctx->HasOutput("FMHAOut"), "Output", "FMHAOut",
                   "FusedAttentionOp");
    OP_INOUT_CHECK(ctx->HasOutput("OutLinearOut"), "Output", "OutLinearOut",
                   "FusedAttentionOp");
#if 1
    OP_INOUT_CHECK(ctx->HasOutput("Ln2Mean"), "Output", "Ln2Mean",
                   "FusedAttentionOp");
    OP_INOUT_CHECK(ctx->HasOutput("Ln2Variance"), "Output", "Ln2Variance",
                   "FusedAttentionOp");
    OP_INOUT_CHECK(ctx->HasOutput("BiasDropoutResidualOut"), "Output",
                   "BiasDropoutResidualOut", "FusedAttentionOp");
    OP_INOUT_CHECK(ctx->HasOutput("DropoutMaskOut"), "Output", "DropoutMaskOut",
                   "FusedAttentionOp");
#endif
    OP_INOUT_CHECK(ctx->HasOutput("Y"), "Output", "Y", "FusedAttentionOp");

    // x: qkv's input [batch_size, seq_len, dim_embed]
    // y: qkv's weight: [3, num_head, dim_head, dim_embed]
    auto x_dim = ctx->GetInputDim("X");
    auto y_dim = ctx->GetInputDim("QKVW");
    // auto qkv_bias_dim = ctx->GetInputDim("QKVBias");
    // auto src_mask_dim = ctx->GetInputDim("SrcMask");
    // std::cout << "x_dim = " << x_dim << std::endl;
    // std::cout << "qkv_weight_dim = " << y_dim << std::endl;
    // std::cout << "qkv_bias_dim = " << qkv_bias_dim << std::endl;
    // // src_mask_dim = 32, 16, 128, 128
    // std::cout << "src_mask_dim = " << src_mask_dim << std::endl;

    PADDLE_ENFORCE_EQ(x_dim.size(), 3,
                      platform::errors::InvalidArgument(
                          "The dimensions of QKV_input must be 3"
                          "(batch_size, seq_len, dim_embed),"
                          "but received dimensions of"
                          "Input is [%d]",
                          x_dim.size()));

    PADDLE_ENFORCE_EQ(y_dim.size(), 4,
                      platform::errors::InvalidArgument(
                          "The dimensions of QKV_weight must be 4"
                          "(3, num_head, dim_head, dim_embed),"
                          "but received dimensions of"
                          "Input is [%d]",
                          y_dim.size()));

    // limin-todo: polish the expression.
    PADDLE_ENFORCE_EQ(x_dim[2], y_dim[3],
                      platform::errors::InvalidArgument(
                          "ShapeError: the dimension of x_dim[2] and y_dim[3]"
                          "must be equal. But received: the shape "
                          "of input X = [%s], and the shape of "
                          "input Y = [%s]",
                          x_dim, y_dim));

    ctx->SetOutputDim("LnMean", {x_dim[0] * x_dim[1]});
    ctx->SetOutputDim("LnVariance", {x_dim[0] * x_dim[1]});
    ctx->SetOutputDim("LnOut", ctx->GetInputDim("X"));
    // [batch_size, seq_len, 3, num_head, head_size]
    ctx->SetOutputDim("QKVOut",
                      {x_dim[0], x_dim[1], y_dim[0], y_dim[1], y_dim[2]});
    ctx->SetOutputDim("QKVBiasOut",
                      {x_dim[0], x_dim[1], y_dim[0], y_dim[1], y_dim[2]});
    // limin-todo: [3, batch_size, seq_len, num_head, head_size]
    // check shape: [3, batch_size, num_head, seq_len, head_size]
    ctx->SetOutputDim("TransposeOut2",
                      {y_dim[0], x_dim[0], y_dim[1], x_dim[1], y_dim[2]});
    // check shape: batch, num_head, seq_len, seq_len
    ctx->SetOutputDim("QKOut", {x_dim[0], y_dim[1], x_dim[1], x_dim[1]});
    ctx->SetOutputDim("SrcMaskOut", {x_dim[0], y_dim[1], x_dim[1], x_dim[1]});
    // the same as QKOut's shape.
    ctx->SetOutputDim("AttnDropoutOut",
                      {x_dim[0], y_dim[1], x_dim[1], x_dim[1]});
    if (ctx->Attrs().Get<bool>("is_test1") == false) {
      ctx->SetOutputDim("AttnDropoutMaskOut",
                        {x_dim[0], y_dim[1], x_dim[1], x_dim[1]});
    }
    ctx->SetOutputDim("SoftmaxOut", {x_dim[0], y_dim[1], x_dim[1], x_dim[1]});
    // check shape [batch_size, num_heads, seq_len, head_dim]
    ctx->SetOutputDim("QKTVOut", {x_dim[0], y_dim[1], x_dim[1], y_dim[2]});
    // check shape, [batch_size, seq_len, number of heads*head size]
    ctx->SetOutputDim("FMHAOut", {x_dim[0], x_dim[1], y_dim[1], y_dim[2]});
    ctx->SetOutputDim("OutLinearOut", ctx->GetInputDim("X"));

#if 1
    ctx->SetOutputDim("Ln2Mean", {x_dim[0] * x_dim[1]});
    ctx->SetOutputDim("Ln2Variance", {x_dim[0] * x_dim[1]});
    if (ctx->Attrs().Get<bool>("is_test") == false) {
      ctx->SetOutputDim("DropoutMaskOut", ctx->GetInputDim("X"));
    }
    ctx->SetOutputDim("BiasDropoutResidualOut", ctx->GetInputDim("X"));
#endif
    ctx->SetOutputDim("Y", ctx->GetInputDim("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input = ctx.Input<Tensor>("X");
    auto input_data_type = input->type();
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
};

class FusedAttentionOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input tensor.");
    AddInput("LnScale",
             "(optional) Scale is a 1-dimensional tensor of size "
             "H(`begin_norm_axis` splits the tensor(`X`) to a matrix [N,H])."
             "It is applied to the output.")
        .AsDispensable();
    AddInput("LnBias",
             "(optional) Bias is a 1-dimensional tensor of size "
             "H(`begin_norm_axis` splits the tensor(`X`) to a matrix [N,H])."
             "It is applied to the output.")
        .AsDispensable();
    AddInput("QKVW", "The qkv weight tensor.");
    AddInput("QKVBias", "The qkv bias tensor.");
    AddInput("SrcMask", "(optional) The attention mask tensor in fmha.")
        .AsDispensable();
    AddInput("OutLinearW", "The out_linear weight tensor.");
    AddInput("OutLinearBias", "The out_linear bias tensor.");
#if 1
    AddInput("Ln2Scale",
             "(optional) Scale is a 1-dimensional tensor of size "
             "H(`begin_norm_axis` splits the tensor(`X`) to a matrix [N,H])."
             "It is applied to the output.")
        .AsDispensable();
    AddInput("Ln2Bias",
             "(optional) Bias is a 1-dimensional tensor of size "
             "H(`begin_norm_axis` splits the tensor(`X`) to a matrix [N,H])."
             "It is applied to the output.")
        .AsDispensable();
#endif
#if 1
// todo:
// AddInput("Seed",
//             "The seed of dropout op, it has higher priority than the attr "
//             "fix_seed and seed")
//         .AsDispensable();
#endif
    AddOutput("LnMean", "Mean of the current mini batch.").AsIntermediate();
    AddOutput("LnVariance", "Variance of the current mini batch.")
        .AsIntermediate();
    AddOutput("LnOut", "The output of pre layer_norm.").AsIntermediate();

    AddOutput("QKVOut", "Result after qkv.").AsIntermediate();
    AddOutput("QKVBiasOut", "Result after qkv and bias op.").AsIntermediate();

    // fma
    AddOutput("TransposeOut2", "Result in fmha.").AsIntermediate();
    AddOutput("QKOut", "Result in fmha.").AsIntermediate();
    AddOutput("QKTVOut", "Result in fmha.").AsIntermediate();
    AddOutput("SoftmaxOut", "Result in fmha.").AsIntermediate();
    AddOutput("AttnDropoutMaskOut", "Result in fmha.").AsIntermediate();
    AddOutput("AttnDropoutOut", "Result in fmha.").AsIntermediate();
    AddOutput("SrcMaskOut", "Result in fmha.").AsIntermediate();
    AddOutput("FMHAOut", "Result after fmha.").AsIntermediate();

    AddOutput("OutLinearOut", "Result after out_linear.").AsIntermediate();

#if 1
    AddOutput("DropoutMaskOut", "The random sampled dropout mask.")
        .AsIntermediate();
    AddOutput("Ln2Mean", "Mean of the current mini batch.").AsIntermediate();
    AddOutput("Ln2Variance", "Variance of the current mini batch.")
        .AsIntermediate();
    AddOutput("BiasDropoutResidualOut",
              "Result of residual + dropout(src + bias).")
        .AsIntermediate();
#endif

    AddOutput("Y", "Result after attention.");

    AddAttr<bool>("pre_layer_norm",
                  "if true, the attention op uses pre_layer_norm architecure, "
                  "else, uses post_layer_norm architecuture. "
                  "[default false].")
        .SetDefault(false);
    AddAttr<float>("epsilon",
                   "Constant for numerical stability [default 1e-5].")
        .SetDefault(1e-5)
        .AddCustomChecker([](const float &epsilon) {
          PADDLE_ENFORCE_EQ(epsilon >= 0.0f && epsilon <= 0.001f, true,
                            platform::errors::InvalidArgument(
                                "'epsilon' in Op(LayerNorm) should be between"
                                "0.0 and 0.001, But received [%s].",
                                epsilon));
        });
    // AddAttr<int>("begin_norm_axis",
    //              "the axis of `begin_norm_axis ... Rank(X) - 1` will be "
    //              "normalized. `begin_norm_axis` splits the tensor(`X`) to a "
    //              "matrix [N,H]. [default 1].")
    //     .SetDefault(1)
    //     .AddCustomChecker([](const int &begin_norm_axis) {
    //       PADDLE_ENFORCE_GT(begin_norm_axis, 0,
    //                         platform::errors::InvalidArgument(
    //                             "'begin_norm_axis' in Op(LayerNorm) should
    //                             be"
    //                             "greater than zero. But received [%d].",
    //                             begin_norm_axis));
    //     });

    // for dropout in fmha.
    AddAttr<float>("attn_dropout_prob", "Probability of setting units to zero.")
        .SetDefault(.5f)
        .AddCustomChecker([](const float &drop_p) {
          PADDLE_ENFORCE_EQ(
              drop_p >= 0.0f && drop_p <= 1.0f, true,
              platform::errors::InvalidArgument(
                  "'attn_dropout_prob' must be between 0.0 and 1.0."));
        });
    AddAttr<bool>("is_test1",
                  "(bool, default false) Set to true for inference only, false "
                  "for training. Some layers may run faster when this is true.")
        .SetDefault(false);
    AddAttr<bool>("fix_seed1",
                  "A flag indicating whether to use a fixed seed to generate "
                  "random mask. NOTE: DO NOT set this flag to true in "
                  "training. Setting this flag to true is only useful in "
                  "unittest or for debug that always the same output units "
                  "will be dropped.")
        .SetDefault(true);
    AddAttr<int>("seed1", "Dropout random seed.").SetDefault(0);
    AddAttr<std::string>(
        "dropout_implementation1",
        "[\"downgrade_in_infer\"|\"upscale_in_train\"]"
        "There are two kinds of ways to implement dropout"
        "(the mask below is a tensor have the same shape with input"
        "the value of mask is 0 or 1, the ratio of 0 is dropout_prob)"
        "1. downgrade_in_infer(default), downgrade the outcome at inference "
        "time"
        "   train: out = input * mask"
        "   inference: out = input * (1.0 - dropout_prob)"
        "2. upscale_in_train, upscale the outcome at training time, do nothing "
        "in inference"
        "   train: out = input * mask / ( 1.0 - dropout_prob )"
        "   inference: out = input"
        "   dropout op can be removed from the program. the program will be "
        "efficient")
        .SetDefault("upscale_in_train")
        .AddCustomChecker([](const std::string &type) {
          PADDLE_ENFORCE_EQ(
              type == "downgrade_in_infer" || type == "upscale_in_train", true,
              platform::errors::InvalidArgument(
                  "dropout_implementation can only be downgrade_in_infer or "
                  "upscale_in_train"));
        });

#if 1
    AddAttr<float>("dropout_prob", "Probability of setting units to zero.")
        .SetDefault(.5f)
        .AddCustomChecker([](const float &drop_p) {
          PADDLE_ENFORCE_EQ(drop_p >= 0.0f && drop_p <= 1.0f, true,
                            platform::errors::InvalidArgument(
                                "'dropout_prob' must be between 0.0 and 1.0."));
        });

    AddAttr<bool>("is_test",
                  "(bool, default false) Set to true for inference only, false "
                  "for training. Some layers may run faster when this is true.")
        .SetDefault(false);
    AddAttr<bool>("fix_seed",
                  "A flag indicating whether to use a fixed seed to generate "
                  "random mask. NOTE: DO NOT set this flag to true in "
                  "training. Setting this flag to true is only useful in "
                  "unittest or for debug that always the same output units "
                  "will be dropped.")
        .SetDefault(true);
    AddAttr<int>("seed", "Dropout random seed.").SetDefault(0);
    AddAttr<std::string>(
        "dropout_implementation",
        "[\"downgrade_in_infer\"|\"upscale_in_train\"]"
        "There are two kinds of ways to implement dropout"
        "(the mask below is a tensor have the same shape with input"
        "the value of mask is 0 or 1, the ratio of 0 is dropout_prob)"
        "1. downgrade_in_infer(default), downgrade the outcome at inference "
        "time"
        "   train: out = input * mask"
        "   inference: out = input * (1.0 - dropout_prob)"
        "2. upscale_in_train, upscale the outcome at training time, do nothing "
        "in inference"
        "   train: out = input * mask / ( 1.0 - dropout_prob )"
        "   inference: out = input"
        "   dropout op can be removed from the program. the program will be "
        "efficient")
        .SetDefault("downgrade_in_infer")
        .AddCustomChecker([](const std::string &type) {
          PADDLE_ENFORCE_EQ(
              type == "downgrade_in_infer" || type == "upscale_in_train", true,
              platform::errors::InvalidArgument(
                  "dropout_implementation can only be downgrade_in_infer or "
                  "upscale_in_train"));
        });
    AddAttr<float>("ln2epsilon",
                   "Constant for numerical stability [default 1e-5].")
        .SetDefault(1e-5)
        .AddCustomChecker([](const float &ln2epsilon) {
          PADDLE_ENFORCE_EQ(ln2epsilon >= 0.0f && ln2epsilon <= 0.001f, true,
                            platform::errors::InvalidArgument(
                                "'epsilon' of the second LayerNorm in Fused "
                                "attention op should be between"
                                "0.0 and 0.001, But received [%s].",
                                ln2epsilon));
        });
#endif

    AddComment(R"DOC(
Fused attention: 
if (pre_layernorm)
    layer_norm;
qkv+bias_add;
fmha;
out_linear;
bias_add + dropout + residual + layer_norm;
)DOC");
  }
};

class FusedAttentionGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
// auto x_dim = ctx->GetInputDim("X");
// auto y_dim = ctx->GetInputDim("QKVW");
// std::cout << "x_dim = " << x_dim << std::endl;
// std::cout << "y_dim = " << y_dim << std::endl;
// int batch_size = x_dim[0];
// int seq_len = x_dim[1];
// int embed_dim = x_dim[2];
// std::cout << "batch_size, seq_len, embed_dim= " << batch_size << ", " <<
// seq_len << ", " << embed_dim << std::endl;

#if 1
    PADDLE_ENFORCE_EQ(ctx->Attrs().Get<bool>("is_test"), false,
                      platform::errors::InvalidArgument(
                          "GradOp is only callable when is_test is false"));

    OP_INOUT_CHECK(ctx->HasInput("Ln2Mean"), "Input", "Ln2Mean",
                   "FusedAttentionGrad");
    OP_INOUT_CHECK(ctx->HasInput("Ln2Variance"), "Input", "Ln2Variance",
                   "FusedAttentionGrad");
    if (ctx->HasOutput(framework::GradVarName("Ln2Scale"))) {
      ctx->SetOutputDim(framework::GradVarName("Ln2Scale"),
                        ctx->GetInputDim("Ln2Scale"));
    }
    if (ctx->HasOutput(framework::GradVarName("Ln2Bias"))) {
      ctx->SetOutputDim(framework::GradVarName("Ln2Bias"),
                        ctx->GetInputDim("Ln2Bias"));
    }
#endif
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "FusedAttentionGrad");
    OP_INOUT_CHECK(ctx->HasInput("LnMean"), "Input", "LnMean",
                   "FusedAttentionGrad");
    OP_INOUT_CHECK(ctx->HasInput("LnVariance"), "Input", "LnVariance",
                   "FusedAttentionGrad");
    if (ctx->Attrs().Get<bool>("pre_layer_norm") == true) {
      OP_INOUT_CHECK(ctx->HasInput("LnOut"), "Input", "LnOut",
                     "FusedAttentionGrad");
    }
    OP_INOUT_CHECK(ctx->HasInput("QKVW"), "Input", "QKVW",
                   "FusedAttentionGrad");
    OP_INOUT_CHECK(ctx->HasInput("QKVBias"), "Input", "QKVBias",
                   "FusedAttentionGrad");
    OP_INOUT_CHECK(ctx->HasInput("SrcMask"), "Input", "SrcMask",
                   "FusedAttentionGrad");
    OP_INOUT_CHECK(ctx->HasInput("OutLinearW"), "Input", "OutLinearW",
                   "FusedAttentionGrad");
    OP_INOUT_CHECK(ctx->HasInput("OutLinearBias"), "Input", "OutLinearBias",
                   "FusedAttentionGrad");

    if (ctx->HasOutput(framework::GradVarName("LnScale"))) {
      ctx->SetOutputDim(framework::GradVarName("LnScale"),
                        ctx->GetInputDim("LnScale"));
    }
    if (ctx->HasOutput(framework::GradVarName("LnBias"))) {
      ctx->SetOutputDim(framework::GradVarName("LnBias"),
                        ctx->GetInputDim("LnBias"));
    }
    if (ctx->HasOutput(framework::GradVarName("X"))) {
      ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
    }

    ctx->SetOutputDim(framework::GradVarName("OutLinearBias"),
                      ctx->GetInputDim("OutLinearBias"));
    ctx->SetOutputDim(framework::GradVarName("OutLinearW"),
                      ctx->GetInputDim("OutLinearW"));
    ctx->SetOutputDim(framework::GradVarName("QKVW"), ctx->GetInputDim("QKVW"));
    ctx->SetOutputDim(framework::GradVarName("QKVBias"),
                      ctx->GetInputDim("QKVBias"));

    ctx->SetOutputDim(framework::GradVarName("LnOut"),
                      ctx->GetInputDim("LnOut"));
    ctx->SetOutputDim(framework::GradVarName("FMHAOut"),
                      ctx->GetInputDim("FMHAOut"));
    ctx->SetOutputDim(framework::GradVarName("QKTVOut"),
                      ctx->GetInputDim("QKTVOut"));
    ctx->SetOutputDim(framework::GradVarName("TransposeOut2"),
                      ctx->GetInputDim("TransposeOut2"));
    ctx->SetOutputDim(framework::GradVarName("QKOut"),
                      ctx->GetInputDim("QKOut"));
    ctx->SetOutputDim(framework::GradVarName("SoftmaxOut"),
                      ctx->GetInputDim("SoftmaxOut"));
    ctx->SetOutputDim(framework::GradVarName("AttnDropoutOut"),
                      ctx->GetInputDim("AttnDropoutOut"));
    ctx->SetOutputDim(framework::GradVarName("SrcMaskOut"),
                      ctx->GetInputDim("SrcMaskOut"));
    ctx->SetOutputDim(framework::GradVarName("QKVOut"),
                      ctx->GetInputDim("QKVOut"));
    ctx->SetOutputDim(framework::GradVarName("QKVBiasOut"),
                      ctx->GetInputDim("QKVBiasOut"));
#if 1
    ctx->SetOutputDim(framework::GradVarName("OutLinearOut"),
                      ctx->GetInputDim("OutLinearOut"));
    // ctx->SetOutputDim(framework::GradVarName("DropoutMaskOut"),
    //                   ctx->GetInputDim("DropoutMaskOut"));
    ctx->SetOutputDim(framework::GradVarName("BiasDropoutResidualOut"),
                      ctx->GetInputDim("BiasDropoutResidualOut"));
#endif
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input = ctx.Input<Tensor>("X");
    auto input_data_type = input->type();
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
};

template <typename T>
class FusedAttentionGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("fused_attention_grad");
    op->SetInput(framework::GradVarName("Y"), this->OutputGrad("Y"));

    // inputs x, parameters and their grad.
    op->SetInput("X", this->Input("X"));
    op->SetInput("QKVW", this->Input("QKVW"));
    op->SetInput("QKVBias", this->Input("QKVBias"));
    op->SetInput("SrcMask", this->Input("SrcMask"));
    op->SetInput("OutLinearW", this->Input("OutLinearW"));
    op->SetInput("OutLinearBias", this->Input("OutLinearBias"));
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
#if 1
    if (this->HasInput("Ln2Scale")) {
      op->SetInput("Ln2Scale", this->Input("Ln2Scale"));
      op->SetOutput(framework::GradVarName("Ln2Scale"),
                    this->InputGrad("Ln2Scale"));
    }
    if (this->HasInput("Ln2Bias")) {
      op->SetInput("Ln2Bias", this->Input("Ln2Bias"));
      op->SetOutput(framework::GradVarName("Ln2Bias"),
                    this->InputGrad("Ln2Bias"));
    }
#endif

    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("QKVW"), this->InputGrad("QKVW"));
    op->SetOutput(framework::GradVarName("QKVBias"),
                  this->InputGrad("QKVBias"));
    op->SetOutput(framework::GradVarName("OutLinearBias"),
                  this->InputGrad("OutLinearBias"));
    op->SetOutput(framework::GradVarName("OutLinearW"),
                  this->InputGrad("OutLinearW"));

    // use forward's output as bw's input.
    op->SetInput("LnOut", this->Output("LnOut"));
    op->SetInput("LnMean", this->Output("LnMean"));
    op->SetInput("LnVariance", this->Output("LnVariance"));
    op->SetInput("QKVOut", this->Output("QKVOut"));
    op->SetInput("QKVBiasOut", this->Output("QKVBiasOut"));
    op->SetInput("TransposeOut2", this->Output("TransposeOut2"));
    op->SetInput("QKOut", this->Output("QKOut"));
    op->SetInput("QKTVOut", this->Output("QKTVOut"));
    op->SetInput("SoftmaxOut", this->Output("SoftmaxOut"));
    op->SetInput("AttnDropoutMaskOut", this->Output("AttnDropoutMaskOut"));
    op->SetInput("AttnDropoutOut", this->Output("AttnDropoutOut"));
    op->SetInput("SrcMaskOut", this->Output("SrcMaskOut"));
    op->SetInput("FMHAOut", this->Output("FMHAOut"));
    op->SetInput("OutLinearOut", this->Output("OutLinearOut"));

#if 1
    op->SetInput("Ln2Mean", this->Output("Ln2Mean"));
    op->SetInput("Ln2Variance", this->Output("Ln2Variance"));
    op->SetInput("DropoutMaskOut", this->Output("DropoutMaskOut"));
    op->SetInput("BiasDropoutResidualOut",
                 this->Output("BiasDropoutResidualOut"));
#endif
    // op->SetInput("QKVBiasOut", this->Output("QKVBiasOut"));
    op->SetInput("QKVOut", this->Output("QKVOut"));

    // bw's output: dinput
    op->SetOutput(framework::GradVarName("LnOut"), this->OutputGrad("LnOut"));
    op->SetOutput(framework::GradVarName("QKVOut"), this->OutputGrad("QKVOut"));
    op->SetOutput(framework::GradVarName("QKVBiasOut"),
                  this->OutputGrad("QKVBiasOut"));
    op->SetOutput(framework::GradVarName("QKTVOut"),
                  this->OutputGrad("QKTVOut"));
    op->SetOutput(framework::GradVarName("TransposeOut2"),
                  this->OutputGrad("TransposeOut2"));
    op->SetOutput(framework::GradVarName("QKOut"), this->OutputGrad("QKOut"));
    op->SetOutput(framework::GradVarName("SoftmaxOut"),
                  this->OutputGrad("SoftmaxOut"));
    op->SetOutput(framework::GradVarName("AttnDropoutOut"),
                  this->OutputGrad("AttnDropoutOut"));
    op->SetOutput(framework::GradVarName("SrcMaskOut"),
                  this->OutputGrad("SrcMaskOut"));
    op->SetOutput(framework::GradVarName("FMHAOut"),
                  this->OutputGrad("FMHAOut"));
#if 1
    // op->SetOutput(framework::GradVarName("DropoutMaskOut"),
    //               this->OutputGrad("DropoutMaskOut"));
    op->SetOutput(framework::GradVarName("BiasDropoutResidualOut"),
                  this->OutputGrad("BiasDropoutResidualOut"));
#endif
    op->SetOutput(framework::GradVarName("OutLinearOut"),
                  this->OutputGrad("OutLinearOut"));
    // op->SetOutput(framework::GradVarName("OutLinearBiasOut"),
    // this->OutputGrad("OutLinearBiasOut"));

    op->SetAttrMap(this->Attrs());
  }
};

// DECLARE_NO_NEED_BUFFER_VARS_INFERER(ElementwiseAddLayerNormGradNoNeedBufferVarInferer,
//                                     "Bias");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(fused_attention, ops::FusedAttentionOp,
                  ops::FusedAttentionOpMaker,
                  ops::FusedAttentionGradOpMaker<paddle::framework::OpDesc>,
                  ops::FusedAttentionGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(fused_attention_grad, ops::FusedAttentionGradOp);
// REGISTER_OPERATOR(fused_attention_grad, ops::FusedAttentionGradOp,
//                   ops::FusedAttentionGradNoNeedBufferVarInferer);
