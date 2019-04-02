// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include <algorithm>
#include <vector>
#include "memory.h"

namespace paddle {
namespace lite {

template <TargetType Target>
class EventTree {
 public:
  using event_t = Event<Target>;

  void AddChild(const event_t& event) { children_.push_back(event); }

  void Sync() {
    for (auto& event : children_) {
      TargetWrapper<Target>::SyncEvent(event);
    }
  }

 private:
  std::vector<event_t> children_;
};

using DDim = std::vector<int64_t>;
static DDim SliceDims(const DDim& dims, int begin, int end) {
  return DDim(dims.begin() + begin, dims.begin() + end - 1);
}

static int product(const DDim& dims) {
  return std::accumulate(dims.begin(), dims.end(), 1,
                         [](int a, int b) { return a * b; });
}

static DDim flatten_to_2d(const DDim& dims, int col) {
  return DDim({product(SliceDims(dims, 0, col)),
               product(SliceDims(dims, col, dims.size()))});
}

using LoD = std::vector<std::vector<size_t>>;

// A light-weight tensor implementation.
class Tensor {
 public:
  Tensor() = default;

  template <typename T>
  const T* data() const {
    return static_cast<const T*>(buffer_.data());
  }

  void Resize(const DDim& ddim) { dims_ = ddim; }

  const DDim& dims() const { return dims_; }

  const LoD& lod() { return lod_; }
  LoD* mutable_lod() { return &lod_; }

  template <typename T>
  T* mutable_data() {
    buffer_.ResetLazy(target_, product(dims_));
    return static_cast<T*>(buffer_.data());
  }

  bool IsInitialized() const { return buffer_.data(); }

 private:
  TargetType target_{TargetType::kHost};
  DDim dims_;
  Buffer buffer_;
  LoD lod_;
};

}  // namespace lite
}  // namespace paddle
