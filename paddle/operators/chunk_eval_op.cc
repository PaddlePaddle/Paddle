/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/operators/chunk_eval_op.h"

namespace paddle {
namespace operators {

class ChunkEvalOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Inference"),
                   "Input(Inference) of ChunkEvalOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Label"),
                   "Input(Label) of ChunkEvalOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Precision"),
                   "Output(Precision) of ChunkEvalOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Recall"),
                   "Output(Recall) of ChunkEvalOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("F1-Score"),
                   "Output(F1-Score) of ChunkEvalOp should not be null.");

    auto inference_dim = ctx->GetInputDim("Inference");
    auto label_dim = ctx->GetInputDim("Label");

    PADDLE_ENFORCE(inference_dim == label_dim,
                   "Inference's shape must be the same as Label's shape.");

    ctx->SetOutputDim("Precision", {1});
    ctx->SetOutputDim("Recall", {1});
    ctx->SetOutputDim("F1-Score", {1});
  }

  framework::DataType IndicateDataType(
      const framework::ExecutionContext &ctx) const override {
    return framework::DataType::FP32;
  }
};

class ChunkEvalOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  ChunkEvalOpMaker(framework::OpProto *proto,
                   framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("Inference",
             "(Tensor, default: Tensor<int>) Predictions from the network.");
    AddInput("Label", "(Tensor, default: Tensor<int>) Labels of the data.");
    AddOutput(
        "Precision",
        "(float) The precision ratio of the predictions on current data.");
    AddOutput("Recall",
              "(float) The recall ratio of the predictions on current data.");
    AddOutput("F1-Score",
              "(float) The F1-Score of the predictions on current data.");
    AddAttr<int>("num_chunk_types", "(int) The number of chunk type.");
    AddAttr<std::string>("chunk_scheme",
                         "(string, default IOB) The label scheme.")
        .SetDefault("IOB");
    AddAttr<std::vector<int>>(
        "excluded_chunk_types",
        "(list<int>) A list<int> indicating chunk types not to be counted.")
        .SetDefault(std::vector<int>{});
    AddComment(R"DOC(
Chunk evaluator is used to evaluate segment labelling accuracy for a
sequence. It calculates precision, recall and F1 scores for the chunk detection.
To use chunk evaluator, several concepts need to be clarified firstly.
[Chunk type] is the type of the whole chunk and a chunk consists of one or several words.  (For example in NER, ORG for organization name, PER for person name etc.)
[Tag type] indicates the position of a word in a chunk. (B for begin, I for inside, E for end, S for single)
We can name a label by combining tag type and chunk type. (ie. B-ORG for begining of an organization name)
The construction of label dictionary should obey the following rules:
- Use one of the listed labelling schemes. These schemes differ in ways indicating chunk boundry.

    Scheme    Description
    plain    Use the same label for the whole chunk.
    IOB      Two labels for chunk type X, B-X for chunk begining and I-X for chunk inside.
    IOE      Two labels for chunk type X, E-X for chunk ending and I-X for chunk inside.
    IOBES    Four labels for chunk type X, B-X for chunk begining, I-X for chunk inside, E-X for chunk end and S-X for single word chunk.

To make it clear, let's illustrate by an NER example.
Assuming that there are three named entity types including ORG, PER and LOC which are called 'chunk type' here,
if 'IOB' scheme were used, the label set will be extended to a set including B-ORG, I-ORG, B-PER, I-PER, B-LOC, I-LOC and O,
in which B-ORG for begining of ORG and I-ORG for inside of ORG.
Prefixes which are called 'tag type' here are added to chunk types and there are two tag types including B and I.
Of course, the training data should be labeled accordingly.
- Mapping is done correctly by the listed equations and assigning protocol.
The following table are equations to extract tag type and chunk type from a label.

    tagType = label % numTagType
    chunkType = label / numTagType
    otherChunkType = numChunkTypes

The following table shows the mapping rule between tagType and tag type in each scheme.

    Scheme Begin Inside End   Single
    plain  0     -      -     -
    IOB    0     1      -     -
    IOE    -     0      1     -
    IOBES  0     1      2     3

Continue the NER example, and the label dict should look like this to satify above equations:

    B-ORG  0
    I-ORG  1
    B-PER  2
    I-PER  3
    B-LOC  4
    I-LOC  5
    O      6

In this example, chunkType has three values: 0 for ORG, 1 for PER, 2 for LOC, because the scheme is
"IOB" so tagType has two values: 0 for B and 1 for I.
Here we will use I-LOC to explain the above mapping rules in detail.
For I-LOC, the label id is 5, so we can get tagType=1 and chunkType=2, which means I-LOC is a part of NER chunk LOC
and the tag is I.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(chunk_eval, ops::ChunkEvalOp,
                             ops::ChunkEvalOpMaker);
REGISTER_OP_CPU_KERNEL(chunk_eval,
                       ops::ChunkEvalKernel<paddle::platform::CPUPlace, float>);
