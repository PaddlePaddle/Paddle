/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/detail/safe_ref.h"

namespace paddle {
namespace operators {

class EmbeddingEltWiseLayerNormOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* context) const override {
    bool has_word_id = context->HasInput("WordId");
    bool has_pos_id = context->HasInput("PosId");
    bool has_sent_id = context->HasInput("SentId");

    PADDLE_ENFORCE_EQ(has_word_id, true,
                      "We can not find the WordId Input of "
                      "EmbeddingEltWiseLayerNormOp for the value of "
                      "HasInput('WordId') is %d.",
                      has_word_id);
    PADDLE_ENFORCE_EQ(has_pos_id, true,
                      "We can not find the PosId Input of "
                      "EmbeddingEltWiseLayerNormOp for the value of "
                      "HasInput('PosId') is %d.",
                      has_pos_id);
    PADDLE_ENFORCE_EQ(has_sent_id, true,
                      "We can not find the SentId Input of "
                      "EmbeddingEltWiseLayerNormOp for the value of "
                      "HasInput('PosId') is %d.",
                      has_sent_id);

    bool has_word_emb = context->HasInput("WordEmb");
    bool has_pos_emb = context->HasInput("PosEmb");
    bool has_sent_emb = context->HasInput("SentEmb");

    PADDLE_ENFORCE_EQ(has_word_emb, true,
                      "We can not find the WordEmb Input of "
                      "EmbeddingEltWiseLayerNormOp for the value of "
                      "HasInput('WordEmb') is %d.",
                      has_word_emb);
    PADDLE_ENFORCE_EQ(has_pos_emb, true,
                      "We can not find the PosEmb Input of "
                      "EmbeddingEltWiseLayerNormOp for the value of "
                      "HasInput('PosEmb') is %d.",
                      has_pos_emb);
    PADDLE_ENFORCE_EQ(has_sent_emb, true,
                      "We can not find the SentEmb Input of "
                      "EmbeddingEltWiseLayerNormOp for the value of "
                      "HasInput('SentEmb') is %d.",
                      has_sent_emb);

    bool has_bias = context->HasInput("Bias");
    bool has_scale = context->HasInput("Scale");
    bool has_out = context->HasOutput("Out");

    PADDLE_ENFORCE_EQ(has_bias, true,
                      "We can not find the Bias Input of "
                      "EmbeddingEltWiseLayerNormOp for the value of "
                      "HasInput('Bias') is %d.",
                      has_bias);
    PADDLE_ENFORCE_EQ(has_scale, true,
                      "We can not find the Scale Input of "
                      "EmbeddingEltWiseLayerNormOp for the value of "
                      "HasInput('Scale') is %d.",
                      has_scale);
    PADDLE_ENFORCE_EQ(has_out, true,
                      "We can not find the 'Out' Output of "
                      "EmbeddingEltWiseLayerNormOp for the value of "
                      "HasOutput('Out') is %d.",
                      has_out);

    // batch * seq_len * 1
    auto dim_word_id = context->GetInputDim("WordId");
    // word_num * hidden
    auto dim_word_emb = context->GetInputDim("WordEmb");
    // hidden
    auto dim_bias = context->GetInputDim("Bias");
    PADDLE_ENFORCE_EQ(
        dim_word_emb[1], dim_bias[0],
        "The second dims (%d) of the Word Embedding should be equal "
        "to the Bias's size(%d).",
        dim_word_emb[1], dim_bias[0]);

    int batch = dim_word_id[0];
    int seq_len = dim_word_id[1];
    int hidden = dim_word_emb[1];
    auto dim_output = framework::make_ddim({batch, seq_len, hidden});
    context->SetOutputDim("Out", dim_output);
    context->ShareLoD("WordId", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "WordEmb");
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

class EmbeddingEltWiseLayerNormOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("WordId", "The word id input of EmbeddingEltWiseLayerNorm op");
    AddInput("PosId", "The position id input of EmbeddingEltWiseLayerNorm op");
    AddInput("SentId", "The sentence id input of EmbeddingEltWiseLayerNorm op");
    AddInput("WordEmb",
             "The Word embedding input of EmbeddingEltWiseLayerNorm op");
    AddInput("PosEmb",
             "The Position embedding input of EmbeddingEltWiseLayerNorm op");
    AddInput("SentEmb",
             "The Sent embedding input of EmbeddingEltWiseLayerNorm op");
    AddInput("Bias", "The LayerNorm Bias of EmbeddingEltWiseLayerNorm op");
    AddInput("Scale", "The LayerNorm Scale of EmbeddingEltWiseLayerNorm op");
    AddOutput("Out", "The output of EmbeddingEltWiseLayerNorm op");
    AddAttr<float>("epsilon",
                   "Constant for numerical stability [default 1e-5].")
        .SetDefault(1e-5)
        .AddCustomChecker([](const float& epsilon) {
          PADDLE_ENFORCE_GE(
              epsilon, 0.0f,
              "'epsilon' is %f, but it should be between 0.0 and 0.001",
              epsilon);
          PADDLE_ENFORCE_LE(
              epsilon, 0.001f,
              "'epsilon' is %f, but it should be between 0.0 and 0.001.",
              epsilon);
        });
    AddComment(R"DOC(
EmbeddingEltWiseLayerNorm Operator.

This op is used for optimize the following structure in ernie model.
wordid -> lookup_table_op -> word
posid -> lookup_table_op -> pos
sentdid -> lookup_table_op -> sent
word + pos + sent -> Y
Y -> layer_norm -> Out

Not suggest to use in other case except has same structure as ernie.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(fused_embedding_eltwise_layernorm,
                             ops::EmbeddingEltWiseLayerNormOp,
                             ops::EmbeddingEltWiseLayerNormOpMaker);
