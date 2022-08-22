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

#pragma once

#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/beam_search_decode_op_def.h"

namespace paddle {
namespace operators {

struct BeamSearchDecodeFunctor {
  BeamSearchDecodeFunctor(const LoDTensorArray& step_ids,
                          const LoDTensorArray& step_scores,
                          LoDTensor* id_tensor,
                          LoDTensor* score_tensor,
                          size_t beam_size,
                          int end_id)
      : beam_size_(beam_size),
        end_id_(end_id),
        step_ids_origin_(step_ids),
        step_scores_origin_(step_scores),
        id_tensor_(id_tensor),
        score_tensor_(score_tensor) {
    tensor_on_gpu_ = false;
    tensor_on_npu_ = false;
    // First make a copy of GPU data on CPU
    if (platform::is_gpu_place(step_ids_origin_[0].place()) ||
        platform::is_npu_place(step_ids_origin_[0].place())) {
      if (platform::is_gpu_place(step_ids_origin_[0].place())) {
        tensor_on_gpu_ = true;
      } else {
        tensor_on_npu_ = true;
      }
      platform::DeviceContextPool& pool =
          platform::DeviceContextPool::Instance();
      auto* dev_ctx = pool.Get(step_ids_origin_[0].place());
      // Copy all tensors in the input tensor array
      for (auto& step_id : step_ids_origin_) {
        framework::LoDTensor out;
        if (step_id.numel() > 0) {
          if (tensor_on_gpu_) {
            dev_ctx->Wait();
          }
          framework::TensorCopy(step_id, platform::CPUPlace(), *dev_ctx, &out);
          dev_ctx->Wait();
        }

        out.set_lod(step_id.lod());
        step_ids_.push_back(out);
      }
    }
    if (platform::is_gpu_place(step_scores_origin_[0].place()) ||
        platform::is_npu_place(step_scores_origin_[0].place())) {
      if (platform::is_gpu_place(step_scores_origin_[0].place())) {
        tensor_on_gpu_ = true;
      } else {
        tensor_on_npu_ = true;
      }
      platform::DeviceContextPool& pool =
          platform::DeviceContextPool::Instance();
      auto* dev_ctx = pool.Get(step_scores_origin_[0].place());
      // Copy all tensors in the input tensor array
      for (auto& step_score : step_scores_origin_) {
        framework::LoDTensor out;
        if (step_score.numel() > 0) {
          if (tensor_on_gpu_) {
            dev_ctx->Wait();
          }
          framework::TensorCopy(
              step_score, platform::CPUPlace(), *dev_ctx, &out);
          dev_ctx->Wait();
        }

        out.set_lod(step_score.lod());
        step_scores_.push_back(out);
      }
    }
  }

  template <typename T>
  void apply_mix() const {
    if (std::is_same<bool, T>::value) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "beam search decode op does not support bool!"));

    } else {
      BeamSearchDecoder<T> beam_search_decoder(beam_size_, end_id_);
      // Check if the tensor is on GPU or NPU. If so, use the CPU copy instead
      if (tensor_on_gpu_ || tensor_on_npu_) {
        beam_search_decoder.Backtrace(
            step_ids_, step_scores_, id_tensor_, score_tensor_);
      } else {
        beam_search_decoder.Backtrace(
            step_ids_origin_, step_scores_origin_, id_tensor_, score_tensor_);
      }
    }
  }

  bool tensor_on_gpu_;
  bool tensor_on_npu_;
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

template <typename DeviceContext, typename T>
class BeamSearchDecodeOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const LoDTensorArray* ids = context.Input<LoDTensorArray>("Ids");
    const LoDTensorArray* scores = context.Input<LoDTensorArray>("Scores");
    const size_t step_num = ids->size();
    PADDLE_ENFORCE_GT(
        step_num,
        0UL,
        platform::errors::InvalidArgument(
            "beam search steps, which is the"
            "size of Input(Ids) LoDTensorArray. beam search steps should "
            "be larger than 0, but received %d. ",
            step_num));
    const size_t source_num = ids->at(0).lod().at(0).size() - 1;
    PADDLE_ENFORCE_GT(
        source_num,
        0UL,
        platform::errors::InvalidArgument(
            "source_num is the sequence number of the"
            "first decoding step, indicating by Input(Ids)[0].lod[0].size. "
            "The number of source_num should be larger than"
            "0, but received %d. ",
            source_num));

    for (size_t i = 0; i < step_num; ++i) {
      PADDLE_ENFORCE_EQ(
          ids->at(i).lod().size(),
          2UL,
          platform::errors::InvalidArgument(
              "For the i step in beam search steps,"
              "the size of Input(Ids)[i].lod() should larger than 2,"
              "but received %d. ",
              ids->at(i).lod().size()));
    }

    size_t beam_size = context.Attr<int>("beam_size");
    int end_id = context.Attr<int>("end_id");

    // prepare output
    LoDTensor* sentenceIds = context.Output<LoDTensor>("SentenceIds");
    LoDTensor* sentenceScores = context.Output<LoDTensor>("SentenceScores");

    BeamSearchDecodeFunctor bs(
        *ids, *scores, sentenceIds, sentenceScores, beam_size, end_id);
    bs.apply_mix<T>();
  }
};

}  // namespace operators
}  // namespace paddle
