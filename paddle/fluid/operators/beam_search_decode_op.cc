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

#include "paddle/fluid/operators/beam_search_decode_op.h"
#include <string>
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace operators {

struct BeamSearchDecodeFunctor {
  BeamSearchDecodeFunctor(const LoDTensorArray& step_ids,
                          const LoDTensorArray& step_scores,
                          LoDTensor* id_tensor, LoDTensor* score_tensor)
      : step_ids_origin_(step_ids),
        step_scores_origin_(step_scores),
        id_tensor_(id_tensor),
        score_tensor_(score_tensor) {
    tensor_on_gpu_ = false;
    // First make a copy of GPU data on CPU
    if (platform::is_gpu_place(step_ids_origin_[0].place())) {
      tensor_on_gpu_ = true;
      platform::DeviceContextPool& pool =
          platform::DeviceContextPool::Instance();
      auto* dev_ctx = pool.Get(step_ids_origin_[0].place());
      // Copy all tensors in the input tensor array
      for (auto& step_id : step_ids_origin_) {
        framework::LoDTensor out;
        dev_ctx->Wait();
        framework::TensorCopy(step_id, platform::CPUPlace(), *dev_ctx, &out);
        dev_ctx->Wait();

        out.set_lod(step_id.lod());
        step_ids_.push_back(out);
      }
    }
    if (platform::is_gpu_place(step_scores_origin_[0].place())) {
      tensor_on_gpu_ = true;
      platform::DeviceContextPool& pool =
          platform::DeviceContextPool::Instance();
      auto* dev_ctx = pool.Get(step_scores_origin_[0].place());
      // Copy all tensors in the input tensor array
      for (auto& step_score : step_scores_origin_) {
        framework::LoDTensor out;
        dev_ctx->Wait();
        framework::TensorCopy(step_score, platform::CPUPlace(), *dev_ctx, &out);
        dev_ctx->Wait();

        out.set_lod(step_score.lod());
        step_scores_.push_back(out);
      }
    }
  }

  template <typename T>
  void operator()() const;

  bool tensor_on_gpu_;
  const LoDTensorArray& step_ids_origin_;
  const LoDTensorArray& step_scores_origin_;
  LoDTensorArray step_ids_ = LoDTensorArray();
  LoDTensorArray step_scores_ = LoDTensorArray();
  LoDTensor* id_tensor_;
  LoDTensor* score_tensor_;
};

template <typename T>
void BeamSearchDecodeFunctor::operator()() const {
  BeamSearchDecoder<T> beam_search_decoder;
  // Check if the tensor is on GPU. If so, use the CPU copy instead
  if (tensor_on_gpu_) {
    beam_search_decoder.PackAllSteps(step_ids_, step_scores_, id_tensor_,
                                     score_tensor_);
  } else {
    beam_search_decoder.PackAllSteps(step_ids_origin_, step_scores_origin_,
                                     id_tensor_, score_tensor_);
  }
}

template <>
void BeamSearchDecodeFunctor::operator()<bool>() const {
  PADDLE_THROW("beam search decode op does not support bool!");
}

class BeamSearchDecodeOp : public framework::OperatorBase {
 public:
  BeamSearchDecodeOp(const std::string& type,
                     const framework::VariableNameMap& inputs,
                     const framework::VariableNameMap& outputs,
                     const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope& scope,
               const platform::Place& dev_place) const override {
    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    auto& dev_ctx = *pool.Get(dev_place);

    framework::ExecutionContext ctx(*this, scope, dev_ctx);

    const LoDTensorArray* ids = ctx.Input<LoDTensorArray>("Ids");
    const LoDTensorArray* scores = ctx.Input<LoDTensorArray>("Scores");
    const size_t step_num = ids->size();
    PADDLE_ENFORCE_GT(step_num, 0UL,
                      "beam search steps should be larger than 0");
    const size_t source_num = ids->at(0).lod().at(0).size() - 1;
    PADDLE_ENFORCE_GT(source_num, 0UL, "source num should be larger than 0");

    for (size_t i = 0; i < step_num; ++i) {
      PADDLE_ENFORCE_EQ(ids->at(i).lod().size(), 2UL,
                        "Level of LodTensor should be 2");
    }

    // prepare output
    LoDTensor* sentenceIds = ctx.Output<LoDTensor>("SentenceIds");
    LoDTensor* sentenceScores = ctx.Output<LoDTensor>("SentenceScores");

    framework::VisitDataType(
        framework::ToDataType(scores->at(0).type()),
        BeamSearchDecodeFunctor(*ids, *scores, sentenceIds, sentenceScores));
  }
};

class BeamSearchDecodeOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Ids",
             "(LodTensorArray)"
             "score of the candidate words in each step");
    AddInput("Scores",
             "(LodTensorArray)"
             "score of the candidate words in each step");
    AddOutput("SentenceIds",
              "(LodTensor)"
              "All possible result sentences of word ids");
    AddOutput("SentenceScores",
              "(LodTensor)"
              "All possible result sentences of word scores");
    AddComment(R"DOC(
Pack the result of Beam search op into SentenceIds and SentenceScores.
)DOC");
  }
};

class BeamSearchDecodeInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext* context) const override {
    PADDLE_ENFORCE(context->HasInput("Ids"),
                   "BeamSearchDecodeOp must has input Ids");
    PADDLE_ENFORCE(context->HasInput("Scores"),
                   "BeamSearchDecodeOp must has input Scores");
    PADDLE_ENFORCE(context->HasOutput("SentenceIds"),
                   "BeamSearchDecodeOp must has output SentenceIds");
    PADDLE_ENFORCE(context->HasOutput("SentenceScores"),
                   "BeamSearchDecodeOp must has output SentenceScores");
  }
};

class BeamSearchDecodeInferVarType : public framework::VarTypeInference {
 public:
  void operator()(const framework::OpDesc& op_desc,
                  framework::BlockDesc* block) const override {
    for (auto& o : op_desc.Output("SentenceIds")) {
      block->Var(o)->SetType(framework::proto::VarType::LOD_TENSOR);
    }
    for (auto& o : op_desc.Output("SentenceScores")) {
      block->Var(o)->SetType(framework::proto::VarType::LOD_TENSOR);
    }
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(beam_search_decode, paddle::operators::BeamSearchDecodeOp,
                  paddle::operators::BeamSearchDecodeOpProtoMaker,
                  paddle::operators::BeamSearchDecodeInferShape,
                  paddle::operators::BeamSearchDecodeInferVarType,
                  paddle::framework::EmptyGradOpMaker);
