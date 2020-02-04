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
    PADDLE_ENFORCE_EQ(
        context->HasInput("WordId"), true,
        "Input(WordId) of EmbeddingEltWiseLayerNormOp should not be null.");
    PADDLE_ENFORCE_EQ(
        context->HasInput("PosId"), true,
        "Input(PosId) of EmbeddingEltWiseLayerNormOp should not be null.");
    PADDLE_ENFORCE_EQ(
        context->HasInput("SentId"), true,
        "Input(SentId) of EmbeddingEltWiseLayerNormOp should not be null.");

    PADDLE_ENFORCE_EQ(
        context->HasInput("WordEmb"), true,
        "Input(WordEmb) of EmbeddingEltWiseLayerNormOp should not be null.");
    PADDLE_ENFORCE_EQ(
        context->HasInput("PosEmb"), true,
        "Input(PosEmb) of EmbeddingEltWiseLayerNormOp should not be null.");
    PADDLE_ENFORCE_EQ(
        context->HasInput("SentEmb"), true,
        "Input(SentEmb) of EmbeddingEltWiseLayerNormOp should not be null.");

    PADDLE_ENFORCE_EQ(context->HasInput("Bias"), true,
                      "Input(Bias) of MultiheadOp should not be null.");
    PADDLE_ENFORCE_EQ(context->HasInput("Scale"), true,
                      "Input(Scale) of MultiheadOp should not be null.");
    PADDLE_ENFORCE_EQ(context->HasOutput("Out"), true,
                      "Output(Out) of MatMulOp should not be null.");

    // batch * seq_len * 1
    auto dim_word_id = context->GetInputDim("WordId");
    // word_num * hidden
    auto dim_word_emb = context->GetInputDim("WordEmb");
    // hidden
    auto dim_bias = context->GetInputDim("Bias");
    PADDLE_ENFORCE_EQ(dim_word_emb[1], dim_bias[0],
                      "The second dims of the Word Embedding should be equal "
                      "to the Bias's size.");

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
          PADDLE_ENFORCE_GE(epsilon, 0.0f,
                            "'epsilon' should be between 0.0 and 0.001.");
          PADDLE_ENFORCE_LE(epsilon, 0.001f,
                            "'epsilon' should be between 0.0 and 0.001.");
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
