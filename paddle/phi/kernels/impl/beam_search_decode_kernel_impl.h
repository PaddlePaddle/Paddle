// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/beam_search_decode.h"

namespace phi {

struct BeamSearchDecodeFunctor {
  BeamSearchDecodeFunctor(const TensorArray& step_ids,
                          const TensorArray& step_scores,
                          phi::DenseTensor* id_tensor,
                          phi::DenseTensor* score_tensor,
                          size_t beam_size,
                          int end_id)
      : beam_size_(beam_size),
        end_id_(end_id),
        step_ids_origin_(step_ids),
        step_scores_origin_(step_scores),
        id_tensor_(id_tensor),
        score_tensor_(score_tensor) {
    tensor_on_gpu_ = false;
    // First make a copy of GPU data on CPU
    if (step_ids_origin_[0].place().GetType() == phi::AllocationType::GPU) {
      if (step_ids_origin_[0].place().GetType() == phi::AllocationType::GPU) {
        tensor_on_gpu_ = true;
      }
      phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
      auto* dev_ctx = pool.Get(step_ids_origin_[0].place());
      // Copy all tensors in the input tensor array
      for (auto& step_id : step_ids_origin_) {
        phi::DenseTensor out;
        if (step_id.numel() > 0) {
          if (tensor_on_gpu_) {
            dev_ctx->Wait();
          }
          phi::Copy(*dev_ctx, step_id, phi::CPUPlace(), false, &out);
          dev_ctx->Wait();
        }

        out.set_lod(step_id.lod());
        step_ids_.push_back(out);
      }
    }
    if (step_scores_origin_[0].place().GetType() == phi::AllocationType::GPU) {
      if (step_scores_origin_[0].place().GetType() ==
          phi::AllocationType::GPU) {
        tensor_on_gpu_ = true;
      }
      phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
      auto* dev_ctx = pool.Get(step_scores_origin_[0].place());
      // Copy all tensors in the input tensor array
      for (auto& step_score : step_scores_origin_) {
        phi::DenseTensor out;
        if (step_score.numel() > 0) {
          if (tensor_on_gpu_) {
            dev_ctx->Wait();
          }
          phi::Copy(*dev_ctx, step_score, phi::CPUPlace(), false, &out);
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
      PADDLE_THROW(common::errors::InvalidArgument(
          "beam search decode op does not support bool!"));

    } else {
      phi::funcs::BeamSearchDecoder<T> beam_search_decoder(beam_size_, end_id_);
      // Check if the tensor is on GPU. If so, use the CPU copy instead
      if (tensor_on_gpu_) {
        beam_search_decoder.Backtrace(
            step_ids_, step_scores_, id_tensor_, score_tensor_);
      } else {
        beam_search_decoder.Backtrace(
            step_ids_origin_, step_scores_origin_, id_tensor_, score_tensor_);
      }
    }
  }

  bool tensor_on_gpu_;
  size_t beam_size_;
  int end_id_;
  // TODO(Superjomn) Here might result serious performance issue in the
  // concurrency
  // scenarios.
  const TensorArray& step_ids_origin_;
  const TensorArray& step_scores_origin_;
  TensorArray step_ids_ = TensorArray();
  TensorArray step_scores_ = TensorArray();
  phi::DenseTensor* id_tensor_;
  phi::DenseTensor* score_tensor_;
};

template <typename T, typename Context>
void BeamSearchDecodeOpKernel(const Context& dev_ctx,
                              const TensorArray& ids_in,
                              const TensorArray& scores_in,
                              int beam_size,
                              int end_id,
                              DenseTensor* sentence_ids,
                              DenseTensor* sentence_scores) {
  const TensorArray* ids = &ids_in;
  const TensorArray* scores = &scores_in;
  const size_t step_num = ids->size();
  PADDLE_ENFORCE_GT(
      step_num,
      0UL,
      common::errors::InvalidArgument(
          "beam search steps, which is the"
          "size of Input(Ids) TensorArray. beam search steps should "
          "be larger than 0, but received %d. ",
          step_num));
  const size_t source_num = ids->at(0).lod().at(0).size() - 1;
  PADDLE_ENFORCE_GT(
      source_num,
      0UL,
      common::errors::InvalidArgument(
          "source_num is the sequence number of the"
          "first decoding step, indicating by Input(Ids)[0].lod[0].size. "
          "The number of source_num should be larger than"
          "0, but received %d. ",
          source_num));

  for (size_t i = 0; i < step_num; ++i) {
    size_t tmp = ids->at(i).lod().size();
    PADDLE_ENFORCE_EQ(
        tmp,
        2UL,
        common::errors::InvalidArgument(
            "For the i step in beam search steps,"
            "the size of Input(Ids)[i].lod() should larger than 2,"
            "but received %d. ",
            tmp));
  }

  // prepare output
  phi::DenseTensor* sentenceIds = sentence_ids;
  phi::DenseTensor* sentenceScores = sentence_scores;
  BeamSearchDecodeFunctor bs(
      *ids, *scores, sentenceIds, sentenceScores, beam_size, end_id);
  bs.apply_mix<T>();
}
}  // namespace phi
