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

    OP_INOUT_CHECK(ctx->HasOutput("LnMean"), "Output", "LnMean",
                   "FusedAttentionOp");
    OP_INOUT_CHECK(ctx->HasOutput("LnVariance"), "Output", "LnVariance",
                   "FusedAttentionOp");
    OP_INOUT_CHECK(ctx->HasOutput("LnOut"), "Output", "LnOut",
                   "FusedAttentionOp");
    // qkv_out: [batch_size, seq_len, 3, num_head, dim_head]
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
    OP_INOUT_CHECK(ctx->HasOutput("Ln2Mean"), "Output", "Ln2Mean",
                   "FusedAttentionOp");
    OP_INOUT_CHECK(ctx->HasOutput("Ln2Variance"), "Output", "Ln2Variance",
                   "FusedAttentionOp");
    OP_INOUT_CHECK(ctx->HasOutput("BiasDropoutResidualOut"), "Output",
                   "BiasDropoutResidualOut", "FusedAttentionOp");
    OP_INOUT_CHECK(ctx->HasOutput("DropoutMaskOut"), "Output", "DropoutMaskOut",
                   "FusedAttentionOp");
    OP_INOUT_CHECK(ctx->HasOutput("Y"), "Output", "Y", "FusedAttentionOp");

    // x: qkv's input [batch_size, seq_len, dim_embed]
    // y: qkv's weight: [3, num_head, dim_head, dim_embed]
    auto x_dim = ctx->GetInputDim("X");
    auto y_dim = ctx->GetInputDim("QKVW");
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
    // [3, batch_size, num_head, seq_len, head_size]
    ctx->SetOutputDim("TransposeOut2",
                      {y_dim[0], x_dim[0], y_dim[1], x_dim[1], y_dim[2]});
    // [batch, num_head, seq_len, seq_len]
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
    // [batch_size, num_heads, seq_len, head_dim]
    ctx->SetOutputDim("QKTVOut", {x_dim[0], y_dim[1], x_dim[1], y_dim[2]});
    // [batch_size, seq_len, number of heads*head size]
    ctx->SetOutputDim("FMHAOut", {x_dim[0], x_dim[1], y_dim[1], y_dim[2]});
    ctx->SetOutputDim("OutLinearOut", ctx->GetInputDim("X"));

    ctx->SetOutputDim("Ln2Mean", {x_dim[0] * x_dim[1]});
    ctx->SetOutputDim("Ln2Variance", {x_dim[0] * x_dim[1]});
    if (ctx->Attrs().Get<bool>("is_test") == false) {
      ctx->SetOutputDim("DropoutMaskOut", ctx->GetInputDim("X"));
    }
    ctx->SetOutputDim("BiasDropoutResidualOut", ctx->GetInputDim("X"));
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
    // AddInput("Seed",
    //             "The seed of dropout op, it has higher priority than the attr
    //             "
    //             "fix_seed and seed")
    //         .AsDispensable();
    AddOutput("LnMean", "Mean of the current mini batch.").AsIntermediate();
    AddOutput("LnVariance", "Variance of the current mini batch.")
        .AsIntermediate();
    AddOutput("LnOut", "The output of pre layer_norm.").AsIntermediate();
    AddOutput("QKVOut", "Result after qkv.").AsIntermediate();
    AddOutput("QKVBiasOut", "Result after qkv and bias op.").AsIntermediate();
    AddOutput("TransposeOut2", "Result in fmha.").AsIntermediate();
    AddOutput("QKOut", "Result in fmha.").AsIntermediate();
    AddOutput("QKTVOut", "Result in fmha.").AsIntermediate();
    AddOutput("SoftmaxOut", "Result in fmha.").AsIntermediate();
    AddOutput("AttnDropoutMaskOut", "Result in fmha.").AsIntermediate();
    AddOutput("AttnDropoutOut", "Result in fmha.").AsIntermediate();
    AddOutput("SrcMaskOut", "Result in fmha.").AsIntermediate();
    AddOutput("FMHAOut", "Result after fmha.").AsIntermediate();
    AddOutput("OutLinearOut", "Result after out_linear.").AsIntermediate();
    AddOutput("DropoutMaskOut", "The random sampled dropout mask.")
        .AsIntermediate();
    AddOutput("Ln2Mean", "Mean of the current mini batch.").AsIntermediate();
    AddOutput("Ln2Variance", "Variance of the current mini batch.")
        .AsIntermediate();
    AddOutput("BiasDropoutResidualOut",
              "Result of residual + dropout(src + bias).")
        .AsIntermediate();
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

    AddComment(R"DOC(
Fused attention op: 
if (pre_layernorm)
    layer_norm;
compute_qkv + bias_add;
fmha;
out_linear;
bias_add + dropout + residual + layer_norm;
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(fused_attention, ops::FusedAttentionOp,
                  ops::FusedAttentionOpMaker);
