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

#include "paddle/operators/trieconcat_op.h"

namespace paddle {
namespace operators {

using value_type = float;
using LoDTensor = framework::LoDTensor;

void PackTwoLoDTensor(const LoDTensor &pre, const LoDTensor &cur,
                      std::vector<std::vector<int64_t>> *ended_result,
                      std::vector<std::vector<int64_t>> *extending_result) {
  PADDLE_ENFORCE_EQ(pre.lod()[0].size(), cur.lod()[0].size(),
                    "the number of source sentences size should be the same");
  PADDLE_ENFORCE_EQ(
      pre.lod()[1].size(), cur.lod()[1].size(),
      "the number of prefix and it's Candidate words should be the same");
  const size_t batch_size = pre.lod()[0].size() - 1;
  for (size_t i = 0; i < batch_size; ++i) {
    int64_t source_start = pre.lod()[0][i];
    int64_t source_end = pre.lod()[0][i + 1];
  }
}

class TrieConcatOp : public framework::OperatorBase {
 public:
  TrieConcatOp(const std::string &type,
               const framework::VariableNameMap &inputs,
               const framework::VariableNameMap &outputs,
               const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}
  void Run(const framework::Scope &scope,
           const platform::DeviceContext &dev_ctx) const override {
    // TODO(qiao) trie concat a vector of LodTensors to a LodTensor
    framework::ExecutionContext ctx(*this, scope, dev_ctx);
    const std::vector<LoDTensor> *input =
        ctx.Input<std::vector<LoDTensor>>("X");
    const size_t step_num = input->size();
    PADDLE_ENFORCE_LT(step_num, 0, "beam search stop should be larger than 0");
    for (auto &in : *input) {
      PADDLE_ENFORCE_EQ(in.lod().size(), 2UL, "Level of LodTensor should be 2");
    }
    // prepare output
    auto *output = ctx.Output<LoDTensor>("Out");
    output->mutable_data<value_type>(ctx.GetPlace());

    const size_t batch_size = input->at(0).lod()[0].size() - 1;

    std::vector<std::vector<int64_t>> ended_result;
    ended_result.reserve(batch_size);
    std::vector<std::vector<int64_t>> extending_result;
    extending_result.reserve(batch_size);
  }
};

class TrieConcatOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  TrieConcatOpProtoMaker(framework::OpProto *proto,
                         framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "(vector<LodTensor>)The input vector of tensors");
    AddOutput("Out", "(Tensor)The output tensor");
    AddComment(R"DOC(
The Tensor will be permuted according to the axis values given.
)DOC");
  }
};

class TrieConcatInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *context) const override {
    PADDLE_ENFORCE(context->HasInput("X"), "TrieConcatOp must has input X");
    PADDLE_ENFORCE(context->HasOutput("out"),
                   "TrieConcatOp must has output Out");
  }
};

class TrieConcatInferVarType : public framework::VarTypeInference {
 public:
  void operator()(const framework::OpDescBind &op_desc,
                  framework::BlockDescBind *block) const override {
    for (auto &o : op_desc.Output("Out")) {
      block->Var(o)->SetType(framework::VarDesc::LOD_TENSOR);
    }
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(trie_concat, paddle::operators::TrieConcatOp,
                  paddle::operators::TrieConcatOpProtoMaker,
                  paddle::operators::TrieConcatInferShape,
                  paddle::operators::TrieConcatInferVarType,
                  paddle::framework::EmptyGradOpMaker);
