// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace paddle {
namespace distributed {

struct PullSparseValue {
  explicit PullSparseValue(int numel, int dim)
      : numel_(numel),
        dim_(dim),
        is_training_(true),
        feasigns_(nullptr),
        frequencies_(nullptr) {}

  explicit PullSparseValue(std::vector<uint64_t>& feasigns,     // NOLINT
                           std::vector<uint32_t>& frequencies,  // NOLINT
                           int dim) {
    numel_ = feasigns.size();
    dim_ = dim;
    is_training_ = true;
    feasigns_ = feasigns.data();
    frequencies_ = frequencies.data();
  }

  void DeserializeFromBytes(void* bytes) {
    /*
    |---isTraining--------------|
    |---8*{num}B(keysData)------|
    |---4*{num}B(Frequencies)---|
    */
    auto* begin = reinterpret_cast<char*>(bytes);
    is_training_ = reinterpret_cast<bool*>(begin)[0];
    feasigns_ = reinterpret_cast<uint64_t*>(begin + sizeof(bool));
    frequencies_ = reinterpret_cast<uint32_t*>(begin + sizeof(bool) +
                                               sizeof(uint64_t) * numel_);
  }

  void Fission(const int shard_id, const int shard_num,
               std::vector<int>* offset_shard) const {
    offset_shard->reserve(numel_ / shard_num + 1);
    for (int x = 0; x < numel_; ++x) {
      if (int(feasigns_[x] % shard_num) == shard_id) {
        offset_shard->push_back(x);
      }
    }
  }

  int numel_;
  int dim_;
  bool is_training_;
  uint64_t* feasigns_;
  uint32_t* frequencies_;
};

}  // namespace distributed
}  // namespace paddle
