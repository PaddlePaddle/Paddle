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
#include "paddle/fluid/platform/errors.h"

namespace paddle {
namespace operators {

class EmbeddingEltWiseLayerNormOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* context) const override {
    PADDLE_ENFORCE_EQ(
        context->Inputs("Ids").size(), context->Inputs("Embs").size(),
        platform::errors::InvalidArgument(
            "Two inputs of EmbeddingEltWiseLayerNormOp shoube be "
            "the same size, but received the size of input Ids = %d,"
            " the size of input Embs = %d",
            context->Inputs("Ids").size(), context->Inputs("Embs").size()));
    PADDLE_ENFORCE_GE(context->Inputs("Embs").size(), 2UL,
                      platform::errors::InvalidArgument(
                          "Input Embs of EmbeddingEltWiseLayerNormOp should "
                          "have at least 2 tensors"));
    PADDLE_ENFORCE_GE(context->Inputs("Ids").size(), 2UL,
                      platform::errors::InvalidArgument(
                          "Input Ids of EmbeddingEltWiseLayerNormOp should "
                          "have at least 2 tensors"));

    PADDLE_ENFORCE_EQ(
        context->HasInput("Bias"), true,
        platform::errors::InvalidArgument(
            "Input(Bias) of EmbeddingEltWiseLayerNormOp should not be null."));

    PADDLE_ENFORCE_EQ(
        context->HasInput("Scale"), true,
        platform::errors::InvalidArgument(
            "Input(Scale) of EmbeddingEltWiseLayerNormOp should not be null."));

    PADDLE_ENFORCE_EQ(
        context->HasOutput("Out"), true,
        platform::errors::InvalidArgument(
            "Output(Out) of EmbeddingEltWiseLayerNormOp should not be null."));

    // batch * seq_len * 1
    auto ids_dims = context->GetInputsDim("Ids");
    // word_num * hidden
    auto embs_dims = context->GetInputsDim("Embs");
    // hidden
    auto dims_bias = context->GetInputDim("Bias");
    int batch = ids_dims[0][0];
    int seq_len = ids_dims[0][1];
    int hidden = embs_dims[0][1];
    for (size_t i = 0; i < embs_dims.size(); ++i) {
      PADDLE_ENFORCE_EQ(embs_dims[i].size(), 2,
                        platform::errors::InvalidArgument(
                            "The Emb dim's size shoule be 2, but found %d.",
                            embs_dims[i].size()));
      PADDLE_ENFORCE_EQ(
          embs_dims[i][1], dims_bias[0],
          platform::errors::InvalidArgument(
              "The second dims (%d) of the Embedding should be equal "
              "to the Bias's size(%d).",
              embs_dims[i][1], dims_bias[0]));
      PADDLE_ENFORCE_EQ(
          embs_dims[i][1], hidden,
          platform::errors::InvalidArgument(
              "The second dimension size(%d) of the Embedding should be "
              "equal to the hidden's size(%d)",
              embs_dims[i][1], hidden));
    }

    auto dim_output = phi::make_ddim({batch, seq_len, hidden});
    context->SetOutputDim("Out", dim_output);
    context->ShareLoD("Ids", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto inputs = ctx.MultiInput<framework::Tensor>("Embs");
    auto input_data_type = framework::proto::VarType::Type(0);
    bool flag = 0;
    for (auto* input : inputs) {
      if (input->IsInitialized() && input->numel() > 0) {
        input_data_type = framework::TransToProtoVarType(input->dtype());
        flag = 1;
        break;
      }
    }
    if (flag == 0) {
      PADDLE_THROW(platform::errors::PreconditionNotMet(
          "All Inputs of fused_embedding_eltwise_layernorm OP are Empty!"));
    }
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
};

class EmbeddingEltWiseLayerNormOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Ids", "Input id tensors of EmbeddingEltWiseLayerNorm op")
        .AsDuplicable();
    AddInput("Embs", "Input emb tensors of EmbeddingEltWiseLayerNorm op")
        .AsDuplicable();
    AddInput("Bias", "The LayerNorm Bias of EmbeddingEltWiseLayerNorm op");
    AddInput("Scale", "The LayerNorm Scale of EmbeddingEltWiseLayerNorm op");
    AddOutput("Out", "The output of EmbeddingEltWiseLayerNorm op");
    AddAttr<float>("epsilon",
                   "Constant for numerical stability [default 1e-5].")
        .SetDefault(1e-5)
        .AddCustomChecker([](const float& epsilon) {
          PADDLE_ENFORCE_GE(
              epsilon, 0.0f,
              platform::errors::InvalidArgument(
                  "'epsilon' is %f, but it should be between 0.0 and 0.001",
                  epsilon));
          PADDLE_ENFORCE_LE(
              epsilon, 0.001f,
              platform::errors::InvalidArgument(
                  "'epsilon' is %f, but it should be between 0.0 and 0.001.",
                  epsilon));
        });
    AddComment(R"DOC(
EmbeddingEltWiseLayerNorm Operator.

This op is used for optimize the following structure in ernie model.
id1 -> lookup_table_op -> data1
id2 -> lookup_table_op -> data2
           ...
idn -> lookup_table_op -> data_n
data1 + data2 + ... + data_n -> Y
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
