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
  BeamSearchDecodeFunctor(const phi::TensorArray& step_ids,
                          const phi::TensorArray& step_scores,
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
          framework::TensorCopy(step_id, phi::CPUPlace(), *dev_ctx, &out);
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
          framework::TensorCopy(step_score, phi::CPUPlace(), *dev_ctx, &out);
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
      BeamSearchDecoder<T> beam_search_decoder(beam_size_, end_id_);
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
  const phi::TensorArray& step_ids_origin_;
  const phi::TensorArray& step_scores_origin_;
  phi::TensorArray step_ids_ = phi::TensorArray();
  phi::TensorArray step_scores_ = phi::TensorArray();
  phi::DenseTensor* id_tensor_;
  phi::DenseTensor* score_tensor_;
};

}  // namespace operators
}  // namespace paddle
