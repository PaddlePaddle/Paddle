// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/eager/grad_node_info.h"

namespace egr {

/**
 * Input Buffer is designed for backward grad accumulate.
 * Since we will have one output used by multi preceding ops in forward pass,
 * we will meet a problem that we need to accumulate multiple grads into one.
 *
 * GradTensorHolder should have as same format as forward output **/
class GradTensorHolder {
 public:
  explicit GradTensorHolder(
      const std::vector<std::vector<GradSlotMeta>>& metas) {
    VLOG(7) << "Init GradTensorHolder with meta size: " << metas.size();
    buffer_.resize(metas.size());
    for (size_t i = 0; i < buffer_.size(); i++) {
      VLOG(7) << "Init GradTensorHolder with meta rank: " << metas[i].size();
      buffer_[i].resize(metas[i].size());
    }
  }

  GradTensorHolder(const GradTensorHolder& other) = default;

  explicit GradTensorHolder(
      std::vector<std::vector<paddle::experimental::Tensor>>&& inputs)
      : buffer_(std::move(inputs)) {}

  GradTensorHolder& operator=(const GradTensorHolder& other) = default;

  // Create new tensor and copy tensor->impl
  void add(size_t slot_id, size_t rank, const paddle::experimental::Tensor& t,
           bool fill_one = false);

  const std::vector<paddle::experimental::Tensor>& operator[](
      const size_t& pos) {
    return buffer_[pos];
  }

  std::vector<std::vector<paddle::experimental::Tensor>>& Buffers() {
    return buffer_;
  }

  void SetBufferSlotRankZeros(size_t slot_id, size_t rank);

 private:
  std::vector<std::vector<paddle::experimental::Tensor>> buffer_;
};

}  // namespace egr
