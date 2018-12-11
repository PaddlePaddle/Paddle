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

#include "paddle/fluid/operators/beam_search_op.h"

#include <string>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

class BeamSearchOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    // inputs and outputs stored in proto
    AddInput("pre_ids",
             "(LoDTensor) The LoDTensor containing the selected ids at the "
             "previous step. It should be a tensor with shape (batch_size, 1) "
             "and lod `[[0, 1, ... , batch_size], [0, 1, ..., batch_size]]` at "
             "thefirst step.");
    AddInput("pre_scores",
             "(LoDTensor) The LoDTensor containing the accumulated "
             "scores corresponding to the selected ids at the previous step.");
    AddInput("ids",
             "(LoDTensor) The LoDTensor containing the candidates ids. Its "
             "shape should be (batch_size * beam_size, K), where K supposed to "
             "be beam_size.");
    AddInput("scores",
             "(LoDTensor) The LodTensor containing the accumulated scores "
             "corresponding to Input(ids) and its shape is the same as the "
             "shape of Input(ids).");
    AddOutput("selected_ids",
              "A LodTensor that stores the IDs selected by beam search.");
    AddOutput("selected_scores",
              "A LoDTensor containing the accumulated scores corresponding to "
              "Output(selected_ids).");

    // Attributes stored in AttributeMap
    AddAttr<int>("level", "the level of LoDTensor");
    AddAttr<int>("beam_size", "beam size for beam search");
    AddAttr<int>("end_id",
                 "the token id which indicates the end of a sequence");

    AddComment(R"DOC(
This operator does the search in beams for one time step. 
Specifically, it selects the top-K candidate word ids of current step from
Input(ids) according to their Input(scores) for all source sentences,
where K is Attr(beam_size) and Input(ids), Input(scores) are predicted results
from the computation cell. Additionally, Input(pre_ids) and Input(pre_scores)
are the output of beam_search at previous step, they are needed for special use
to handle ended candidate translations. The paths linking prefixes and selected
candidates are organized and reserved in lod.

Note that the Input(scores) passed in should be accumulated scores, and
length penalty should be done with extra operators before calculating the
accumulated scores if needed, also suggest finding top-K before it and
using the top-K candidates following.
)DOC");
  }
};

class BeamSearchOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *ctx) const override {
    for (const std::string &arg :
         std::vector<std::string>({"pre_ids", "ids", "scores"})) {
      PADDLE_ENFORCE(ctx->HasInput(arg), "BeamSearch need input argument '%s'",
                     arg);
    }
    for (const std::string &arg :
         std::vector<std::string>({"selected_ids", "selected_scores"})) {
      PADDLE_ENFORCE(ctx->HasOutput(arg),
                     "BeamSearch need output argument '%s'", arg);
    }
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    framework::OpKernelType kt = framework::OpKernelType(
        framework::ToDataType(
            ctx.Input<framework::LoDTensor>("pre_ids")->type()),
        platform::CPUPlace());
    return kt;
  }
};

class BeamSearchInferVarType : public framework::VarTypeInference {
 public:
  void operator()(const framework::OpDesc &op_desc,
                  framework::BlockDesc *block) const override {
    for (auto &o : op_desc.Output("selected_ids")) {
      auto &selected_ids = block->FindRecursiveOrCreateVar(o);
      selected_ids.SetType(framework::proto::VarType::LOD_TENSOR);
    }
    for (auto &o : op_desc.Output("selected_scores")) {
      auto &selected_scores = block->FindRecursiveOrCreateVar(o);
      selected_scores.SetType(framework::proto::VarType::LOD_TENSOR);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(beam_search, ops::BeamSearchOp, ops::BeamSearchOpMaker,
                  ops::BeamSearchInferVarType);
REGISTER_OP_CPU_KERNEL(
    beam_search,
    ops::BeamSearchOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::BeamSearchOpKernel<paddle::platform::CPUDeviceContext, double>,
    ops::BeamSearchOpKernel<paddle::platform::CPUDeviceContext, int>,
    ops::BeamSearchOpKernel<paddle::platform::CPUDeviceContext, int64_t>);
