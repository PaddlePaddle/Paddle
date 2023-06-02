// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <string>
#include <vector>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/operators/conv_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_dnn.h"

namespace paddle {
namespace operators {

class FusedDotProductSelfAttentionOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(
        ctx->HasInput("QKV"), "Input", "QKV", "FusedDotProductSelfAttentionOp");
    OP_INOUT_CHECK(ctx->HasInput("ActualSeqlenQ"),
                   "Input",
                   "ActualSeqlenQ",
                   "FusedDotProductSelfAttentionOp");
    OP_INOUT_CHECK(ctx->HasInput("ActualSeqlenKV"),
                   "Input",
                   "ActualSeqlenKV",
                   "FusedDotProductSelfAttentionOp");

    // output: [batch_size, seq_len, num_heads, head_size]
    OP_INOUT_CHECK(ctx->HasOutput("Out"),
                   "Output",
                   "Out",
                   "FusedDotProductSelfAttentionOp");
    OP_INOUT_CHECK(ctx->HasOutput("SoftmaxOut"),
                   "Output",
                   "SoftmaxOut",
                   "FusedDotProductSelfAttentionOp");

    // qkv input shape: [batch_size, seq_len, 3, num_heads, head_size]
    auto qkv_dim = ctx->GetInputDim("QKV");
    // actual seqlen shape: [batch_size]
    auto q_actual_seqlen_dim = ctx->GetInputDim("ActualSeqlenQ");
    auto kv_actual_seqlen_dim = ctx->GetInputDim("ActualSeqlenKV");
    PADDLE_ENFORCE_EQ(qkv_dim.size(),
                      5,
                      platform::errors::InvalidArgument(
                          "The dimensions of qkv must be 5"
                          "(batch_size, seq_len, 3, num_heads, head_size),"
                          "but received dimensions of"
                          "Input is [%d]",
                          qkv_dim.size()));
    PADDLE_ENFORCE_EQ(q_actual_seqlen_dim.size(),
                      1,
                      platform::errors::InvalidArgument(
                          "The dimensions of actual seqlen of Q must be 1"
                          "but received dimensions of"
                          "Input is [%d]",
                          q_actual_seqlen_dim.size()));
    PADDLE_ENFORCE_EQ(kv_actual_seqlen_dim.size(),
                      1,
                      platform::errors::InvalidArgument(
                          "The dimensions of actual seqlen of K must be 1"
                          "but received dimensions of"
                          "Input is [%d]",
                          kv_actual_seqlen_dim.size()));
    PADDLE_ENFORCE_EQ(
        qkv_dim[0],
        q_actual_seqlen_dim[0],
        platform::errors::InvalidArgument(
            "ShapeError: the dimension of qkv_dim[0] and actual_seqlen_dim[0]"
            "must be equal. But received: the shape "
            "of input qkv = [%s], and the shape of "
            "input actual_seqlen = [%s]",
            qkv_dim,
            q_actual_seqlen_dim));
    PADDLE_ENFORCE_EQ(
        qkv_dim[2],
        3,
        platform::errors::InvalidArgument(
            "ShapeError: the dimension of qkv_dim[2] must be 3. But received: "
            "the qkv_dim[2] = [%d]",
            qkv_dim[2]));

    // [batch_size, seq_len, num_head, head_size]
    ctx->SetOutputDim("Out", {qkv_dim[0], qkv_dim[1], qkv_dim[3], qkv_dim[4]});

    ctx->SetOutputDim("SoftmaxOut",
                      {qkv_dim[0], qkv_dim[3], qkv_dim[1], qkv_dim[1]});
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type = OperatorWithKernel::IndicateVarDataType(ctx, "QKV");
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
  framework::OpKernelType GetKernelTypeForVar(
      const std::string &var_name,
      const framework::Tensor &tensor,
      const framework::OpKernelType &expected_kernel_type) const override {
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   expected_kernel_type.place_,
                                   tensor.layout());
  }
};

class FusedDotProductSelfAttentionOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    // inputs
    AddInput("QKV", "The QKV tensor after projection.");
    AddInput("ActualSeqlenQ", "The actual sequence length of Q.");
    AddInput("ActualSeqlenKV", "The actual sequence length of K.");
    AddInput("Bias", "(optional) The bias tensor.").AsDispensable();

    AddOutput("SoftmaxOut", "The softmax output tensor").AsIntermediate();
    AddOutput("Out", "The output tensor of dot product attention.");

    AddAttr<float>("scaling_factor", "The scale value after Q*KT.")
        .SetDefault(.125f);
    AddAttr<float>("attn_dropout_rate", "Probability of setting units to zero.")
        .SetDefault(.5f)
        .AddCustomChecker([](const float &drop_p) {
          PADDLE_ENFORCE_EQ(
              drop_p >= 0.0f && drop_p <= 1.0f,
              true,
              platform::errors::InvalidArgument(
                  "'attn_dropout_rate' must be between 0.0 and 1.0."));
        });
    AddAttr<int>("attn_dropout_seed", "Dropout random seed.").SetDefault(0);
    AddAttr<bool>("is_causal_masking", "Specify if we need causal masking")
        .SetDefault(false);

    AddComment(R"DOC(
  The fused_dot_product_attention operator is the same as following pseudo codes:
  input(qkv): [batch_size, seq_len, 3, num_heads, head_dim]
  ouput: [batch_size, seq_len, num_heads, head_dim]
  {
    out = q * k^t;
    out = attn_mask + out;
    out = softmax(out);
    out = dropout(out);
    out = out * v;
    out = transpose(out, perm=[0, 2, 1, 3]);
  }
    )DOC");
  }
};

class FusedDotProductSelfAttentionGradOp
    : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(
        ctx->HasInput("QKV"), "Input", "QKV", "FusedDotProductSelfAttentionOp");
    OP_INOUT_CHECK(ctx->HasInput("ActualSeqlenQ"),
                   "Input",
                   "ActualSeqlenQ",
                   "FusedDotProductSelfAttentionOp");
    OP_INOUT_CHECK(ctx->HasInput("ActualSeqlenKV"),
                   "Input",
                   "ActualSeqlenKV",
                   "FusedDotProductSelfAttentionOp");
    OP_INOUT_CHECK(ctx->HasInput("SoftmaxOut"),
                   "Input",
                   "SoftmaxOut",
                   "FusedDotProductSelfAttentionOp");

    if (ctx->HasOutput(framework::GradVarName("QKV"))) {
      ctx->SetOutputDim(framework::GradVarName("QKV"), ctx->GetInputDim("QKV"));
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type = OperatorWithKernel::IndicateVarDataType(ctx, "QKV");
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
  framework::OpKernelType GetKernelTypeForVar(
      const std::string &var_name,
      const framework::Tensor &tensor,
      const framework::OpKernelType &expected_kernel_type) const override {
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   expected_kernel_type.place_,
                                   tensor.layout());
  }
};

template <typename T>
class FusedDotProductSelfAttentionGradOpMaker
    : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("fused_dot_product_self_attention_grad");
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetInput("QKV", this->Input("QKV"));
    op->SetInput("ActualSeqlenQ", this->Input("ActualSeqlenQ"));
    op->SetInput("ActualSeqlenKV", this->Input("ActualSeqlenKV"));
    op->SetInput("SoftmaxOut", this->Output("SoftmaxOut"));
    op->SetOutput(framework::GradVarName("QKV"), this->InputGrad("QKV"));
    op->SetAttrMap(this->Attrs());
  }
};

class FusedDotProductCrossAttentionOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(
        ctx->HasInput("Q"), "Input", "Q", "FusedDotProductCrossAttentionOp");
    OP_INOUT_CHECK(
        ctx->HasInput("KV"), "Input", "KV", "FusedDotProductCrossAttentionOp");
    OP_INOUT_CHECK(ctx->HasInput("ActualSeqlenQ"),
                   "Input",
                   "ActualSeqlenQ",
                   "FusedDotProductCrossAttentionOp");
    OP_INOUT_CHECK(ctx->HasInput("ActualSeqlenKV"),
                   "Input",
                   "ActualSeqlenKV",
                   "FusedDotProductCrossAttentionOp");

    // output: [batch_size, seq_len, num_heads, head_size]
    OP_INOUT_CHECK(ctx->HasOutput("Out"),
                   "Output",
                   "Out",
                   "FusedDotProductCrossAttentionOp");
    OP_INOUT_CHECK(ctx->HasOutput("SoftmaxOut"),
                   "Output",
                   "SoftmaxOut",
                   "FusedDotProductCrossAttentionOp");

    // q input shape: [batch_size, q_seq_len, num_heads, head_size]
    // kv input shape: [batch_size, kv_seq_len, 2, num_heads, head_size]
    auto q_dim = ctx->GetInputDim("Q");
    auto kv_dim = ctx->GetInputDim("KV");
    // actual seqlen shape: [batch_size]
    auto q_actual_seqlen_dim = ctx->GetInputDim("ActualSeqlenQ");
    auto kv_actual_seqlen_dim = ctx->GetInputDim("ActualSeqlenKV");
    PADDLE_ENFORCE_EQ(q_dim.size(),
                      4,
                      platform::errors::InvalidArgument(
                          "The dimensions of q must be 4"
                          "(batch_size, seq_len, num_heads, head_size),"
                          "but received dimensions of"
                          "Input is [%d]",
                          q_dim.size()));
    PADDLE_ENFORCE_EQ(kv_dim.size(),
                      5,
                      platform::errors::InvalidArgument(
                          "The dimensions of kv must be 5"
                          "(batch_size, seq_len, 2, num_heads, head_size),"
                          "but received dimensions of"
                          "Input is [%d]",
                          kv_dim.size()));
    PADDLE_ENFORCE_EQ(q_actual_seqlen_dim.size(),
                      1,
                      platform::errors::InvalidArgument(
                          "The dimensions of actual seqlen of Q must be 1"
                          "but received dimensions of"
                          "Input is [%d]",
                          q_actual_seqlen_dim.size()));
    PADDLE_ENFORCE_EQ(kv_actual_seqlen_dim.size(),
                      1,
                      platform::errors::InvalidArgument(
                          "The dimensions of actual seqlen of K must be 1"
                          "but received dimensions of"
                          "Input is [%d]",
                          kv_actual_seqlen_dim.size()));
    PADDLE_ENFORCE_EQ(
        kv_dim[0],
        q_actual_seqlen_dim[0],
        platform::errors::InvalidArgument(
            "ShapeError: the dimension of kv_dim[0] and actual_seqlen_dim[0]"
            "must be equal. But received: the shape "
            "of input kv = [%s], and the shape of "
            "input actual_seqlen = [%s]",
            kv_dim,
            q_actual_seqlen_dim));
    PADDLE_ENFORCE_EQ(
        kv_dim[2],
        2,
        platform::errors::InvalidArgument(
            "ShapeError: the dimension of qkv_dim[2] must be 2. But received: "
            "the qkv_dim[2] = [%d]",
            kv_dim[2]));

    // [batch_size, q_seq_len, num_head, head_size]
    ctx->SetOutputDim("Out", {q_dim[0], q_dim[1], q_dim[2], q_dim[3]});

    // [batch_size, num_head, q_seq_len, kv_seq_len]
    ctx->SetOutputDim("SoftmaxOut", {q_dim[0], q_dim[2], q_dim[1], kv_dim[1]});
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type = OperatorWithKernel::IndicateVarDataType(ctx, "Q");
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
  framework::OpKernelType GetKernelTypeForVar(
      const std::string &var_name,
      const framework::Tensor &tensor,
      const framework::OpKernelType &expected_kernel_type) const override {
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   expected_kernel_type.place_,
                                   tensor.layout());
  }
};

class FusedDotProductCrossAttentionOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    // inputs
    AddInput("Q", "The Q tensor after projection.");
    AddInput("KV", "The KV tensor after projection.");
    AddInput("ActualSeqlenQ", "The actual sequence length of Q.");
    AddInput("ActualSeqlenKV", "The actual sequence length of K.");
    AddInput("Bias", "(optional) The bias tensor.").AsDispensable();

    AddOutput("SoftmaxOut", "The softmax output tensor").AsIntermediate();
    AddOutput("Out", "The output tensor of dot product attention.");

    AddAttr<float>("scaling_factor", "The scale value after Q*KT.")
        .SetDefault(.125f);
    AddAttr<float>("attn_dropout_rate", "Probability of setting units to zero.")
        .SetDefault(.5f)
        .AddCustomChecker([](const float &drop_p) {
          PADDLE_ENFORCE_EQ(
              drop_p >= 0.0f && drop_p <= 1.0f,
              true,
              platform::errors::InvalidArgument(
                  "'attn_dropout_rate' must be between 0.0 and 1.0."));
        });
    AddAttr<int>("attn_dropout_seed", "Dropout random seed.").SetDefault(0);
    AddAttr<bool>("is_causal_masking", "Specify if we need causal masking")
        .SetDefault(false);

    AddComment(R"DOC(
  The fused_dot_product_attention operator is the same as following pseudo codes:
  input(qkv): [batch_size, seq_len, 3, num_heads, head_dim]
  ouput: [batch_size, seq_len, num_heads, head_dim]
  {
    out = q * k^t;
    out = attn_mask + out;
    out = softmax(out);
    out = dropout(out);
    out = out * v;
    out = transpose(out, perm=[0, 2, 1, 3]);
  }
    )DOC");
  }
};

class FusedDotProductCrossAttentionGradOp
    : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(
        ctx->HasInput("Q"), "Input", "Q", "FusedDotProductCrossAttentionOp");
    OP_INOUT_CHECK(
        ctx->HasInput("KV"), "Input", "KV", "FusedDotProductCrossAttentionOp");
    OP_INOUT_CHECK(ctx->HasInput("ActualSeqlenQ"),
                   "Input",
                   "ActualSeqlenQ",
                   "FusedDotProductCrossAttentionOp");
    OP_INOUT_CHECK(ctx->HasInput("ActualSeqlenKV"),
                   "Input",
                   "ActualSeqlenKV",
                   "FusedDotProductCrossAttentionOp");
    OP_INOUT_CHECK(ctx->HasInput("SoftmaxOut"),
                   "Input",
                   "SoftmaxOut",
                   "FusedDotProductCrossAttentionOp");

    if (ctx->HasOutput(framework::GradVarName("Q"))) {
      ctx->SetOutputDim(framework::GradVarName("Q"), ctx->GetInputDim("Q"));
    }
    if (ctx->HasOutput(framework::GradVarName("KV"))) {
      ctx->SetOutputDim(framework::GradVarName("KV"), ctx->GetInputDim("KV"));
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type = OperatorWithKernel::IndicateVarDataType(ctx, "Q");
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
  framework::OpKernelType GetKernelTypeForVar(
      const std::string &var_name,
      const framework::Tensor &tensor,
      const framework::OpKernelType &expected_kernel_type) const override {
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   expected_kernel_type.place_,
                                   tensor.layout());
  }
};

template <typename T>
class FusedDotProductCrossAttentionGradOpMaker
    : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("fused_dot_product_cross_attention_grad");
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetInput("Q", this->Input("Q"));
    op->SetInput("KV", this->Input("KV"));
    op->SetInput("ActualSeqlenQ", this->Input("ActualSeqlenQ"));
    op->SetInput("ActualSeqlenKV", this->Input("ActualSeqlenKV"));
    op->SetInput("SoftmaxOut", this->Output("SoftmaxOut"));
    op->SetOutput(framework::GradVarName("Q"), this->InputGrad("Q"));
    op->SetOutput(framework::GradVarName("KV"), this->InputGrad("KV"));
    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    fused_dot_product_self_attention,
    ops::FusedDotProductSelfAttentionOp,
    ops::FusedDotProductSelfAttentionOpMaker,
    ops::FusedDotProductSelfAttentionGradOpMaker<paddle::framework::OpDesc>,
    ops::FusedDotProductSelfAttentionGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(fused_dot_product_self_attention_grad,
                  ops::FusedDotProductSelfAttentionGradOp);

REGISTER_OPERATOR(
    fused_dot_product_cross_attention,
    ops::FusedDotProductCrossAttentionOp,
    ops::FusedDotProductCrossAttentionOpMaker,
    ops::FusedDotProductCrossAttentionGradOpMaker<paddle::framework::OpDesc>,
    ops::FusedDotProductCrossAttentionGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(fused_dot_product_cross_attention_grad,
                  ops::FusedDotProductCrossAttentionGradOp);
