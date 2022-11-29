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

#include "paddle/fluid/operators/chunk_eval_op.h"

#include <string>
#include <vector>

namespace paddle {
namespace operators {

class ChunkEvalOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(
        ctx->HasInput("Inference"), "Input", "Inference", "chunk_eval");
    OP_INOUT_CHECK(ctx->HasInput("Label"), "Input", "Label", "chunk_eval");

    OP_INOUT_CHECK(
        ctx->HasOutput("Precision"), "Output", "Precision", "chunk_eval");
    OP_INOUT_CHECK(ctx->HasOutput("Recall"), "Output", "Recall", "chunk_eval");
    OP_INOUT_CHECK(
        ctx->HasOutput("F1-Score"), "Output", "F1-Score", "chunk_eval");
    OP_INOUT_CHECK(ctx->HasOutput("NumInferChunks"),
                   "Output",
                   "NumInferChunks",
                   "chunk_eval");
    OP_INOUT_CHECK(ctx->HasOutput("NumLabelChunks"),
                   "Output",
                   "NumLabelChunks",
                   "chunk_eval");
    OP_INOUT_CHECK(ctx->HasOutput("NumCorrectChunks"),
                   "Output",
                   "NumCorrectChunks",
                   "chunk_eval");

    auto inference_dim = ctx->GetInputDim("Inference");
    auto label_dim = ctx->GetInputDim("Label");

    PADDLE_ENFORCE_EQ(
        inference_dim,
        label_dim,
        platform::errors::InvalidArgument(
            "Input(Inference)'s shape must be the same as Input(Label)'s "
            "shape, but received [%s] (Inference) vs [%s] (Label).",
            inference_dim,
            label_dim));

    bool use_padding = ctx->HasInput("SeqLength");
    if (use_padding) {
      PADDLE_ENFORCE_EQ(
          (inference_dim.size() == 3 && inference_dim[2] == 1) ||
              inference_dim.size() == 2,
          true,
          platform::errors::InvalidArgument(
              "when Input(SeqLength) is provided, Input(Inference) "
              "should be of dim 3 (batch_size, bucket, 1) or dim 2 "
              "(batch_size, bucket), but received [%s].",
              inference_dim));
      auto seq_length_dim = ctx->GetInputDim("SeqLength");
      PADDLE_ENFORCE_LE(seq_length_dim.size(),
                        2,
                        platform::errors::InvalidArgument(
                            "Input(SeqLength)'s rank should not be greater "
                            "than 2, but received %d.",
                            seq_length_dim.size()));
    }

    ctx->SetOutputDim("Precision", {1});
    ctx->SetOutputDim("Recall", {1});
    ctx->SetOutputDim("F1-Score", {1});
    ctx->SetOutputDim("NumInferChunks", {1});
    ctx->SetOutputDim("NumLabelChunks", {1});
    ctx->SetOutputDim("NumCorrectChunks", {1});
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(framework::proto::VarType::FP32,
                                   platform::CPUPlace());
  }
};

class ChunkEvalOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Inference",
             "(Tensor, default: Tensor<int64_t>). "
             "Predictions from the network.");
    AddInput("Label",
             "(Tensor, default: Tensor<int64_t>). The true tag sequences.");
    AddInput("SeqLength",
             "(Tensor, default: Tensor<int64_t>). The length of each sequence, "
             "used when Inference and Label are Tensor type .")
        .AsDispensable();
    AddOutput("Precision",
              "(float). The evaluated precision (called positive predictive "
              "value) of chunks on the given mini-batch.");
    AddOutput("Recall",
              "(float). The evaluated recall (true positive rate or "
              "sensitivity) of chunks on the given mini-batch.");
    AddOutput("F1-Score",
              "(float). The evaluated F1-Score on the given mini-batch.");
    AddOutput("NumInferChunks",
              "(int64_t). The number of chunks in Inference on the given "
              "mini-batch.");
    AddOutput(
        "NumLabelChunks",
        "(int64_t). The number of chunks in Label on the given mini-batch.");
    AddOutput(
        "NumCorrectChunks",
        "(int64_t). The number of chunks both in Inference and Label on the "
        "given mini-batch.");
    AddAttr<int>("num_chunk_types",
                 "The number of chunk type. See the description for details.");
    AddAttr<std::string>("chunk_scheme",
                         "The labeling scheme indicating "
                         "how to encode the chunks. Must be IOB, IOE, IOBES or "
                         "plain. See the description"
                         "for details.")
        .SetDefault("IOB");
    AddAttr<std::vector<int>>("excluded_chunk_types",
                              "A list including chunk type ids "
                              "indicating chunk types that are not counted. "
                              "See the description for details.")
        .SetDefault(std::vector<int>{});
    AddComment(R"DOC(
For some basics of chunking, please refer to
'Chunking with Support Vector Machines <https://aclanthology.info/pdf/N/N01/N01-1025.pdf>'.

ChunkEvalOp computes the precision, recall, and F1-score of chunk detection,
and supports IOB, IOE, IOBES and IO (also known as plain) tagging schemes.
Here is a NER example of labeling for these tagging schemes:

          Li     Ming    works  at  Agricultural   Bank   of    China  in  Beijing.
   IO     I-PER  I-PER   O      O   I-ORG          I-ORG  I-ORG I-ORG  O   I-LOC
   IOB    B-PER  I-PER   O      O   B-ORG          I-ORG  I-ORG I-ORG  O   B-LOC
   IOE    I-PER  E-PER   O      O   I-ORG          I-ORG  I-ORG E-ORG  O   E-LOC
   IOBES  B-PER  E-PER   O      O   I-ORG          I-ORG  I-ORG E-ORG  O   S-LOC

There are three chunk types(named entity types) including PER(person), ORG(organization)
and LOC(LOCATION), and we can see that the labels have the form <tag type>-<chunk type>.

Since the calculations actually use label ids rather than labels, extra attention
should be paid when mapping labels to ids to make CheckEvalOp work. The key point
is that the listed equations are satisfied by ids.

   tag_type = label % num_tag_type
   chunk_type = label / num_tag_type

where `num_tag_type` is the num of tag types in the tagging scheme, `num_chunk_type`
is the num of chunk types, and `tag_type` get its value from the following table.

   Scheme Begin Inside End   Single
    plain   0     -      -     -
    IOB     0     1      -     -
    IOE     -     0      1     -
    IOBES   0     1      2     3

Still use NER as example, assuming the tagging scheme is IOB while chunk types are ORG,
PER and LOC. To satisfy the above equations, the label map can be like this:

   B-ORG  0
   I-ORG  1
   B-PER  2
   I-PER  3
   B-LOC  4
   I-LOC  5
   O      6

It's not hard to verify the equations noting that the num of chunk types
is 3 and the num of tag types in IOB scheme is 2. For example, the label
id of I-LOC is 5, the tag type id of I-LOC is 1, and the chunk type id of
I-LOC is 2, which consistent with the results from the equations.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(chunk_eval,
                             ops::ChunkEvalOp,
                             ops::ChunkEvalOpMaker);
REGISTER_OP_CPU_KERNEL(chunk_eval,
                       ops::ChunkEvalKernel<paddle::platform::CPUPlace, float>);
