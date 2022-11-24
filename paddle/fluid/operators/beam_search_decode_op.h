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
        phi::DenseTensor out;
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
        phi::DenseTensor out;
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

<<<<<<< HEAD
template <typename T>
using SentenceVector = std::vector<Sentence<T>>;

template <typename T>
struct BeamSearchDecoder {
  BeamSearchDecoder(size_t beam_size, int end_id)
      : beam_size_(beam_size), end_id_(end_id) {}

  /**
   * convert the result sentence_vector for each source sentence into two
   * LodTensor.
   * One is all candidate sentences with word id, one is all candidate sentences
   * with word score.
   * Param:
   *  sentence_vector_list: sentence_vector for each source sentence.
   *  id_tensor: result LoDTensor for sentences of id.
   *  score_tensor: result LoDTensor for sentences of score.
   *  reverse: whether ids of sentence in sentence_vector_list is reversed
   *  sort_by_score: whether to sort hypotheses of each sentence by scores.
   */
  void ConvertSentenceVectorToLodTensor(
      std::vector<SentenceVector<T>> sentence_vector_list,
      LoDTensor* id_tensor,
      LoDTensor* score_tensor,
      bool reverse = true,
      bool sort_by_score = true) const;

  /**
   * Gather the hypotheses for each source sentence by backtrace though the
   * LoDTensorArray step_ids whose lods reserve the path in the tree.
   */
  void Backtrace(const LoDTensorArray& step_ids,
                 const LoDTensorArray& step_scores,
                 LoDTensor* id_tensor,
                 LoDTensor* score_tensor) const;
=======
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
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f

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

<<<<<<< HEAD
template <typename T>
void BeamSearchDecoder<T>::ConvertSentenceVectorToLodTensor(
    std::vector<SentenceVector<T>> sentence_vector_list,
    LoDTensor* id_tensor,
    LoDTensor* score_tensor,
    bool reverse,
    bool sort_by_score) const {
  size_t src_num = sentence_vector_list.size();

  PADDLE_ENFORCE_NE(
      src_num,
      0,
      platform::errors::InvalidArgument(
          "src_num is the sequence number of the first decoding step"
          ", indicating by Input(Ids)[0].lod[0].size."
          "src_num has wrong value."
          "src_num should not be 0,"
          "But received %d.",
          src_num));

  std::vector<size_t> source_level_lod = {0};
  std::vector<size_t> sentence_level_lod = {0};
  std::vector<int64_t> id_data;
  std::vector<T> score_data;

  for (size_t src_idx = 0; src_idx < src_num; ++src_idx) {
    if (sort_by_score) {
      sort(sentence_vector_list[src_idx].begin(),
           sentence_vector_list[src_idx].end(),
           [reverse](const Sentence<T>& a, const Sentence<T>& b) {
             if (reverse)
               return a.scores.front() > b.scores.front();
             else
               return a.scores.back() > b.scores.back();
           });
    }
    for (Sentence<T>& sentence : sentence_vector_list[src_idx]) {
      if (reverse) {
        id_data.insert(id_data.end(),
                       sentence.word_ids.rbegin(),
                       sentence.word_ids.rend());
        score_data.insert(
            score_data.end(), sentence.scores.rbegin(), sentence.scores.rend());
      } else {
        id_data.insert(
            id_data.end(), sentence.word_ids.begin(), sentence.word_ids.end());
        score_data.insert(
            score_data.end(), sentence.scores.begin(), sentence.scores.end());
      }
=======
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
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f

    size_t beam_size = context.Attr<int>("beam_size");
    int end_id = context.Attr<int>("end_id");

<<<<<<< HEAD
  auto cpu_place = std::unique_ptr<paddle::platform::CPUPlace>(
      new paddle::platform::CPUPlace());
  phi::CPUContext cpu_ctx(*cpu_place);

  framework::LoD lod;
  lod.push_back(source_level_lod);
  lod.push_back(sentence_level_lod);

  id_tensor->set_lod(lod);
  id_tensor->Resize({static_cast<int64_t>(id_data.size())});
  id_tensor->mutable_data<int64_t>(paddle::platform::CPUPlace());
  framework::TensorFromVector<int64_t>(id_data, cpu_ctx, id_tensor);

  score_tensor->set_lod(lod);
  score_tensor->Resize({static_cast<int64_t>(score_data.size())});
  score_tensor->mutable_data<T>(paddle::platform::CPUPlace());
  framework::TensorFromVector<T>(score_data, cpu_ctx, score_tensor);
}

template <typename T>
void BeamSearchDecoder<T>::Backtrace(const LoDTensorArray& step_ids,
                                     const LoDTensorArray& step_scores,
                                     LoDTensor* id_tensor,
                                     LoDTensor* score_tensor) const {
  PADDLE_ENFORCE_NE(
      step_ids.empty(),
      true,
      platform::errors::InvalidArgument("Input(Ids) should not be empty."
                                        "But the Input(Ids) is empty."));
  PADDLE_ENFORCE_EQ(
      step_ids.size(),
      step_scores.size(),
      platform::errors::InvalidArgument(
          "The size of Input(Ids) and Input(Scores) should be "
          "the same. But the size of Input(Ids) and Input(Scores) "
          "are not equal."));
  const size_t step_num = step_ids.size();
  const size_t src_num = step_ids.at(0).lod().at(kSourceLevel).size() - 1;
  std::vector<SentenceVector<T>> sentence_vector_list(
      src_num, SentenceVector<T>(beam_size_));
  std::vector<std::vector<size_t>> prefix_idx_vector_list(src_num);
  for (int step_id = step_num - 1; step_id >= 0; --step_id) {
    auto& cur_ids = step_ids.at(step_id);
    auto& cur_scores = step_scores.at(step_id);
    for (size_t src_idx = 0; src_idx < src_num; ++src_idx) {
      // for each source sentence
      auto& sentence_vector = sentence_vector_list.at(src_idx);
      auto& prefix_idx_vector = prefix_idx_vector_list.at(src_idx);
      size_t src_prefix_start = cur_ids.lod().at(kSourceLevel)[src_idx];
      size_t src_prefix_end = cur_ids.lod().at(kSourceLevel)[src_idx + 1];
      if (prefix_idx_vector.empty()) {  // be finished and pruned at this step
                                        // or the last time step
        for (size_t prefix_idx = src_prefix_start; prefix_idx < src_prefix_end;
             ++prefix_idx) {
          size_t candidate_start = cur_ids.lod().at(kSentenceLevel)[prefix_idx];
          size_t candidate_end =
              cur_ids.lod().at(kSentenceLevel)[prefix_idx + 1];
          for (size_t candidate_idx = candidate_start;
               candidate_idx < candidate_end;
               ++candidate_idx) {
            prefix_idx_vector.push_back(prefix_idx);
            size_t idx = prefix_idx_vector.size() - 1;
            auto cur_id = cur_ids.data<int64_t>()[candidate_idx];
            auto cur_score = cur_scores.data<T>()[candidate_idx];
            sentence_vector.at(idx).word_ids.push_back(cur_id);
            sentence_vector.at(idx).scores.push_back(cur_score);
          }
        }
      } else {  // use prefix_idx_vector to backtrace
        size_t src_candidate_start =
            cur_ids.lod().at(kSentenceLevel)[src_prefix_start];
        size_t prefix_idx = src_prefix_start;
        size_t candidate_num =
            cur_ids.lod().at(kSentenceLevel)[prefix_idx + 1] -
            cur_ids.lod().at(kSentenceLevel)[prefix_idx];
        for (size_t idx = 0; idx < prefix_idx_vector.size(); ++idx) {
          auto candidate_idx = prefix_idx_vector.at(idx);
          auto cur_id = cur_ids.data<int64_t>()[candidate_idx];
          auto cur_score = cur_scores.data<T>()[candidate_idx];
          if (cur_id != end_id_ || sentence_vector.at(idx).word_ids.empty()) {
            // to skip redundant end tokens
            sentence_vector.at(idx).word_ids.push_back(cur_id);
            sentence_vector.at(idx).scores.push_back(cur_score);
          }
=======
    // prepare output
    LoDTensor* sentenceIds = context.Output<LoDTensor>("SentenceIds");
    LoDTensor* sentenceScores = context.Output<LoDTensor>("SentenceScores");
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f

    BeamSearchDecodeFunctor bs(
        *ids, *scores, sentenceIds, sentenceScores, beam_size, end_id);
    bs.apply_mix<T>();
  }
<<<<<<< HEAD

  ConvertSentenceVectorToLodTensor(
      sentence_vector_list, id_tensor, score_tensor, true, true);
}
=======
};
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f

}  // namespace operators
}  // namespace paddle
