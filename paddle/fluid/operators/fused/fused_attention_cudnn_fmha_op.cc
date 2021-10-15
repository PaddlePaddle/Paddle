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

// #include "paddle/fluid/operators/fused/fused_attention_cunn_op.h"

#include <memory>
#include <string>

#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

class FusedAttentionCuDNNFMHAOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
// std::cout << "i am in op infershape\n";
#if CUDNN_VERSION >= 8000
    // mha
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "x", "FusedAttentionOp");
    // OP_INOUT_CHECK(ctx->HasInput("K"), "Input", "K", "FusedAttentionOp");
    // OP_INOUT_CHECK(ctx->HasInput("V"), "Input", "V", "FusedAttentionOp");
    OP_INOUT_CHECK(ctx->HasInput("QO_Seqlen"), "Input", "QO_Seqlen",
                   "FusedAttentionOp");
    OP_INOUT_CHECK(ctx->HasInput("KV_Seqlen"), "Input", "KV_Seqlen",
                   "FusedAttentionOp");
    OP_INOUT_CHECK(ctx->HasInput("OutLinearBias"), "Input", "OutLinearBias",
                   "FusedAttentionOp");

    auto x_dims = ctx->GetInputDim("X");
    // PADDLE_ENFORCE_EQ(x_dims.size(), CUDNN_SEQDATA_DIM_COUNT,
    //                   platform::errors::InvalidArgument(
    //                       "The input tensor X's dimensions of
    //                       FusedAttentionOp "
    //                       "should be equal to %d . But received X's "
    //                       "dimensions = %d.",
    //                       CUDNN_SEQDATA_DIM_COUNT, x_dims.size()));

    auto qo_slen_dims = ctx->GetInputDim("QO_Seqlen");
    PADDLE_ENFORCE_EQ(qo_slen_dims[0], x_dims[0],
                      platform::errors::InvalidArgument(
                          "The number of sequence length should be equal"
                          " to batch size."));

    auto kv_slen_dims = ctx->GetInputDim("KV_Seqlen");
    PADDLE_ENFORCE_EQ(kv_slen_dims[0], x_dims[0],
                      platform::errors::InvalidArgument(
                          "The number of sequence length should be equal"
                          " to batch size."));

    OP_INOUT_CHECK(ctx->HasOutput("LnMean"), "Output", "LnMean",
                   "FusedAttentionOp");
    OP_INOUT_CHECK(ctx->HasOutput("LnVariance"), "Output", "LnVariance",
                   "FusedAttentionOp");
    OP_INOUT_CHECK(ctx->HasOutput("LnOut"), "Output", "LnOut",
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

    auto x_dim = ctx->GetInputDim("X");
    ctx->SetOutputDim("LnMean", {x_dim[0] * x_dim[1]});
    ctx->SetOutputDim("LnVariance", {x_dim[0] * x_dim[1]});
    ctx->SetOutputDim("LnOut", ctx->GetInputDim("X"));
    ctx->SetOutputDim("OutLinearOut", ctx->GetInputDim("X"));

    ctx->SetOutputDim("Ln2Mean", {x_dim[0] * x_dim[1]});
    ctx->SetOutputDim("Ln2Variance", {x_dim[0] * x_dim[1]});
    if (ctx->Attrs().Get<bool>("dropout_is_test") == false) {
      ctx->SetOutputDim("DropoutMaskOut", ctx->GetInputDim("X"));
    }
    ctx->SetOutputDim("BiasDropoutResidualOut", ctx->GetInputDim("X"));

    std::vector<int64_t> output_dims;
    for (int i = 0; i < x_dims.size(); ++i) {
      output_dims.push_back(x_dims[i]);
    }

    ctx->SetOutputDim("Y", framework::make_ddim(output_dims));
    ctx->ShareLoD("X", /*->*/ "Y");
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

class FusedAttentionCuDNNFMHAOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
#if CUDNN_VERSION >= 8000
    // mha
    AddInput("X", "(Tensor), X");
    // AddInput("K", "(Tensor), K");
    // AddInput("V", "(Tensor), V");
    AddInput("W", "(Tensor), W");
    AddInput("QO_Seqlen", "(Tensor), QO_Seqlen");
    AddInput("KV_Seqlen", "(Tensor), KV_Seqlen");

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

    AddOutput("LnMean", "Mean of the current mini batch.").AsIntermediate();
    AddOutput("LnVariance", "Variance of the current mini batch.")
        .AsIntermediate();
    AddOutput("LnOut", "The output of pre layer_norm.").AsIntermediate();

    AddOutput("ReserveSpace",
              "Reserve GPU space for triggering the new semi-persistent "
              "NHWC kernel")
        .AsDispensable()
        .AsExtra();
    AddOutput("OutLinearOut", "Result after out_linear.").AsIntermediate();

    AddOutput("DropoutMaskOut", "The random sampled dropout mask.")
        .AsIntermediate();
    AddOutput("Ln2Mean", "Mean of the current mini batch.").AsIntermediate();
    AddOutput("Ln2Variance", "Variance of the current mini batch.")
        .AsIntermediate();
    AddOutput("BiasDropoutResidualOut",
              "Result of residual + dropout(src + bias).")
        .AsIntermediate();
    AddOutput("Y", "(Tensor), final result of attention.");

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
    // mha
    AddAttr<std::vector<int>>("attn_low_windows", "(Tensor), attn_low_windows");
    AddAttr<std::vector<int>>("attn_high_windows",
                              "(Tensor), attn_high_windows");
    AddAttr<std::vector<int>>("attn_qo_seqlen", "(Tensor), attn_qo_seqlen");
    AddAttr<std::vector<int>>("attn_kv_seqlen", "(Tensor), attn_kv_seqlen");

    AddAttr<float>("attn_dropout_prob", "");
    AddAttr<int>("attn_heads", "");
    //  AddAttr<float>("attn_sm_scaler", "");
    // AddAttr<int>("attn_vec_size", "");
    // AddAttr<int>("attn_q_proj_size", "");
    // AddAttr<int>("attn_k_proj_size", "");
    // AddAttr<int>("attn_v_proj_size", "");
    // AddAttr<int>("attn_o_proj_size", "");
    // AddAttr<int>("attn_max_qo_seq_len", "");
    // AddAttr<int>("attn_max_kv_seq_len", "");
    // AddAttr<int>("attn_beam_size", "");

    AddAttr<float>("dropout_prob", "Probability of setting units to zero.")
        .SetDefault(.5f)
        .AddCustomChecker([](const float &drop_p) {
          PADDLE_ENFORCE_EQ(drop_p >= 0.0f && drop_p <= 1.0f, true,
                            platform::errors::InvalidArgument(
                                "'dropout_prob' must be between 0.0 and 1.0."));
        });

    AddAttr<bool>("dropout_is_test",
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
        "There are two kinds of ways to implement dropout"
        "(the mask below is a tensor have the same shape with input"
        "the value of mask is 0 or 1, the ratio of 0 is dropout_prob)"
        "1. downgrade_in_infer(default), downgrade the outcome at inference "
        "time"
        "   train: out = input * mask"
        "   inference: out = input * (1.0 - dropout_prob)"
        "2. upscale_in_train, upscale the outcome at training time, do "
        "nothing "
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

    AddComment(R"DOC(MHA OP Test)DOC");
#endif
  }
};

class FusedAttentionCuDNNFMHAGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
#if CUDNN_VERSION >= 8000
    PADDLE_ENFORCE_EQ(
        ctx->Attrs().Get<bool>("dropout_is_test"), false,
        platform::errors::InvalidArgument(
            "GradOp is only callable when dropout_is_test is false"));

    // mha
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Y")), "Input",
                   "Y@GRAD", "FusedAttentionCuDNNFMHAGrad");
    OP_INOUT_CHECK(ctx->HasInput("QO_Seqlen"), "Input", "QO_Seqlen",
                   "FusedAttentionCuDNNFMHAGrad");
    OP_INOUT_CHECK(ctx->HasInput("KV_Seqlen"), "Input", "KV_Seqlen",
                   "FusedAttentionCuDNNFMHAGrad");
    OP_INOUT_CHECK(ctx->HasInput("OutLinearBias"), "Input", "OutLinearBias",
                   "FusedAttentionCuDNNFMHAGrad");

    // mha
    // std::string var_names[4] = {"Q", "K", "V", "W"};
    // for (auto s : var_names) {
    //   OP_INOUT_CHECK(ctx->HasInput(s), "Input", s, "FusedAttentionMHAGrad");
    //   auto dims = ctx->GetInputDim(s);
    //   auto grad_name = framework::GradVarName(s);

    //   if (ctx->HasOutput(grad_name)) {
    //     ctx->SetOutputDim(grad_name, dims);
    //   }
    // }

    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X",
                   "FusedAttentionCuDNNFMHAGrad");
    if (ctx->HasOutput(framework::GradVarName("X"))) {
      ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
    }
    ctx->SetOutputDim(framework::GradVarName("W"), ctx->GetInputDim("W"));

    if (ctx->Attrs().Get<bool>("pre_layer_norm") == true) {
      OP_INOUT_CHECK(ctx->HasInput("LnMean"), "Input", "LnMean",
                     "FusedAttentionCuDNNFMHAGrad");
      OP_INOUT_CHECK(ctx->HasInput("LnVariance"), "Input", "LnVariance",
                     "FusedAttentionCuDNNFMHAGrad");
      OP_INOUT_CHECK(ctx->HasInput("LnOut"), "Input", "LnOut",
                     "FusedAttentionCuDNNFMHAGrad");
    }
    if (ctx->HasOutput(framework::GradVarName("LnScale"))) {
      ctx->SetOutputDim(framework::GradVarName("LnScale"),
                        ctx->GetInputDim("LnScale"));
    }
    if (ctx->HasOutput(framework::GradVarName("LnBias"))) {
      ctx->SetOutputDim(framework::GradVarName("LnBias"),
                        ctx->GetInputDim("LnBias"));
    }

    OP_INOUT_CHECK(ctx->HasInput("Ln2Mean"), "Input", "Ln2Mean",
                   "FusedAttentionCuDNNFMHAGrad");
    OP_INOUT_CHECK(ctx->HasInput("Ln2Variance"), "Input", "Ln2Variance",
                   "FusedAttentionCuDNNFMHAGrad");
    if (ctx->HasOutput(framework::GradVarName("Ln2Scale"))) {
      ctx->SetOutputDim(framework::GradVarName("Ln2Scale"),
                        ctx->GetInputDim("Ln2Scale"));
    }
    if (ctx->HasOutput(framework::GradVarName("Ln2Bias"))) {
      ctx->SetOutputDim(framework::GradVarName("Ln2Bias"),
                        ctx->GetInputDim("Ln2Bias"));
    }

    ctx->SetOutputDim(framework::GradVarName("LnOut"),
                      ctx->GetInputDim("LnOut"));
    ctx->SetOutputDim(framework::GradVarName("OutLinearBias"),
                      ctx->GetInputDim("OutLinearBias"));
    ctx->SetOutputDim(framework::GradVarName("BiasDropoutResidualOut"),
                      ctx->GetInputDim("BiasDropoutResidualOut"));
    ctx->SetOutputDim(framework::GradVarName("OutLinearOut"),
                      ctx->GetInputDim("OutLinearOut"));
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
class FusedAttentionCuDNNFMHAGradOpMaker
    : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
#if CUDNN_VERSION >= 8000
    op->SetType("fused_attention_cudnn_fmha_grad");
    op->SetInput(framework::GradVarName("Y"), this->OutputGrad("Y"));

    op->SetInput("X", this->Input("X"));
    // op->SetInput("K", this->Input("K"));
    // op->SetInput("V", this->Input("V"));
    op->SetInput("W", this->Input("W"));
    op->SetInput("QO_Seqlen", this->Input("QO_Seqlen"));
    op->SetInput("KV_Seqlen", this->Input("KV_Seqlen"));

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
    op->SetInput("OutLinearBias", this->Input("OutLinearBias"));

    if (this->HasOutput("ReserveSpace")) {
      op->SetInput("ReserveSpace", this->Output("ReserveSpace"));
    }

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

    op->SetInput("LnOut", this->Output("LnOut"));
    op->SetInput("LnMean", this->Output("LnMean"));
    op->SetInput("LnVariance", this->Output("LnVariance"));

    op->SetInput("OutLinearOut", this->Output("OutLinearOut"));

    op->SetInput("Ln2Mean", this->Output("Ln2Mean"));
    op->SetInput("Ln2Variance", this->Output("Ln2Variance"));
    op->SetInput("DropoutMaskOut", this->Output("DropoutMaskOut"));
    op->SetInput("BiasDropoutResidualOut",
                 this->Output("BiasDropoutResidualOut"));

    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    // op->SetOutput(framework::GradVarName("K"), this->InputGrad("K"));
    // op->SetOutput(framework::GradVarName("V"), this->InputGrad("V"));
    op->SetOutput(framework::GradVarName("W"), this->InputGrad("W"));

    op->SetOutput(framework::GradVarName("LnOut"), this->OutputGrad("LnOut"));

    op->SetOutput(framework::GradVarName("OutLinearBias"),
                  this->InputGrad("OutLinearBias"));
    op->SetOutput(framework::GradVarName("OutLinearOut"),
                  this->OutputGrad("OutLinearOut"));
    op->SetOutput(framework::GradVarName("BiasDropoutResidualOut"),
                  this->OutputGrad("BiasDropoutResidualOut"));

    op->SetAttrMap(this->Attrs());
#endif
  }
};

// DECLARE_NO_NEED_BUFFER_VARS_INFERER(ElementwiseAddLayerNormGradNoNeedBufferVarInferer,
//                                     "Bias");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    fused_attention_cudnn_fmha, ops::FusedAttentionCuDNNFMHAOp,
    ops::FusedAttentionCuDNNFMHAOpMaker,
    ops::FusedAttentionCuDNNFMHAGradOpMaker<paddle::framework::OpDesc>,
    ops::FusedAttentionCuDNNFMHAGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(fused_attention_cudnn_fmha_grad,
                  ops::FusedAttentionCuDNNFMHAGradOp);
