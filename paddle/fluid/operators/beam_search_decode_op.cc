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

#include <algorithm>
#include <string>

#include "paddle/fluid/operators/beam_search_decode_op.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace operators {

struct BeamSearchDecodeFunctor {
  BeamSearchDecodeFunctor(const LoDTensorArray& step_ids,
                          const LoDTensorArray& step_scores,
                          LoDTensor* id_tensor, LoDTensor* score_tensor,
                          size_t beam_size, int end_id)
      : beam_size_(beam_size),
        end_id_(end_id),
        step_ids_origin_(step_ids),
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
        if (step_id.numel() > 0) {
          dev_ctx->Wait();
          framework::TensorCopy(step_id, platform::CPUPlace(), *dev_ctx, &out);
          dev_ctx->Wait();
        }

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
        if (step_score.numel() > 0) {
          dev_ctx->Wait();
          framework::TensorCopy(step_score, platform::CPUPlace(), *dev_ctx,
                                &out);
          dev_ctx->Wait();
        }

        out.set_lod(step_score.lod());
        step_scores_.push_back(out);
      }
    }
  }

  template <typename T>
  void apply() const;

  bool tensor_on_gpu_;
  size_t beam_size_;
  int end_id_;
  // TODO(Superjomn) Here might result serious performance issue in the
  // concurrency
  // scenarios.
  const LoDTensorArray& step_ids_origin_;
  const LoDTensorArray& step_scores_origin_;
  LoDTensorArray step_ids_ = LoDTensorArray();
  LoDTensorArray step_scores_ = LoDTensorArray();
  LoDTensor* id_tensor_;
  LoDTensor* score_tensor_;
};

template <typename T>
void BeamSearchDecodeFunctor::apply() const {
  BeamSearchDecoder<T> beam_search_decoder(beam_size_, end_id_);
  // Check if the tensor is on GPU. If so, use the CPU copy instead
  if (tensor_on_gpu_) {
    beam_search_decoder.Backtrace(step_ids_, step_scores_, id_tensor_,
                                  score_tensor_);
  } else {
    beam_search_decoder.Backtrace(step_ids_origin_, step_scores_origin_,
                                  id_tensor_, score_tensor_);
  }
}

template <>
void BeamSearchDecodeFunctor::apply<bool>() const {
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

    framework::RuntimeContext run_ctx(Inputs(), Outputs(), scope);
    framework::ExecutionContext ctx(*this, scope, dev_ctx, run_ctx);

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

    size_t beam_size = ctx.Attr<int>("beam_size");
    int end_id = ctx.Attr<int>("end_id");

    // prepare output
    LoDTensor* sentenceIds = ctx.Output<LoDTensor>("SentenceIds");
    LoDTensor* sentenceScores = ctx.Output<LoDTensor>("SentenceScores");

    framework::VisitDataType(
        scores->at(0).type(),
        BeamSearchDecodeFunctor(*ids, *scores, sentenceIds, sentenceScores,
                                beam_size, end_id));
  }
};

class BeamSearchDecodeOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Ids",
             "(LodTensorArray)"
             "The LodTensorArray containing the selected ids of all steps");
    AddInput("Scores",
             "(LodTensorArray)"
             "The LodTensorArray containing the selected scores of all steps");
    AddOutput(
        "SentenceIds",
        "(LodTensor)"
        "An LodTensor containing all generated id sequences for all source "
        "sentences");
    AddOutput(
        "SentenceScores",
        "(LodTensor)"
        "An LodTensor containing scores corresponding to Output(SentenceIds)");
    AddAttr<int>("beam_size", "beam size for beam search");
    AddAttr<int>("end_id",
                 "the token id which indicates the end of a sequence");
    AddComment(R"DOC(
Beam Search Decode Operator. This Operator constructs the full hypotheses for
each source sentence by walking back along the LoDTensorArray Input(ids)
whose lods can be used to restore the path in the beam search tree.

The Output(SentenceIds) and Output(SentenceScores) separately contain the
generated id sequences and the corresponding scores. The shapes and lods of the
two LodTensor are same. The lod level is 2 and the two levels separately
indicate how many hypotheses each source sentence has and how many ids each
hypothesis has.
)DOC");
  }
};

class BeamSearchDecodeInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext* context) const override {
    PADDLE_ENFORCE(context->HasInput("Ids"),
                   "BeamSearchDecodeOp must have input Ids");
    PADDLE_ENFORCE(context->HasInput("Scores"),
                   "BeamSearchDecodeOp must have input Scores");
    PADDLE_ENFORCE(context->HasOutput("SentenceIds"),
                   "BeamSearchDecodeOp must have output SentenceIds");
    PADDLE_ENFORCE(context->HasOutput("SentenceScores"),
                   "BeamSearchDecodeOp must have output SentenceScores");
  }
};

class BeamSearchDecodeInferVarType : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext* ctx) const override {
    ctx->SetOutputType("SentenceIds", framework::proto::VarType::LOD_TENSOR,
                       framework::ALL_ELEMENTS);
    ctx->SetOutputType("SentenceScores", framework::proto::VarType::LOD_TENSOR,
                       framework::ALL_ELEMENTS);
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(
    beam_search_decode, paddle::operators::BeamSearchDecodeOp,
    paddle::operators::BeamSearchDecodeOpProtoMaker,
    paddle::operators::BeamSearchDecodeInferShape,
    paddle::operators::BeamSearchDecodeInferVarType,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
