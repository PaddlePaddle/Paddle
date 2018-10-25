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
#include <vector>
#include "paddle/fluid/framework/lod_tensor.h"

namespace paddle {
namespace framework {

// NOTE The vector<LoDTensor> can't be replaced with the class LoDTensorArray
// directly, because there are many vector<LoDTensor> used accross the project,
// and some of them are treated as LoDTensorArray.
#if !defined(PADDLE_ON_INFERENCE)

using LoDTensorArray = std::vector<LoDTensor>;

#else  // !PADDLE_ON_INFERENCE

#pragma message "LoDTensorArray is replaced with the inference one."
/*
 * A LoDTensorArray which will not deallocate buffer when resized, fix the data
 * diff in inference, and more performance friendly in the concurrency
 * scenerios.
 */
class LoDTensorArray {
 public:
  LoDTensorArray() = default;

  using iterator = std::vector<LoDTensor>::iterator;
  using const_iterator = std::vector<LoDTensor>::const_iterator;

  const_iterator begin() const { return array_.begin(); }
  const_iterator end() const { return array_.begin() + size_; }
  iterator begin() { return array_.begin(); }
  iterator end() { return array_.begin() + size_; }

  void push_back(const LoDTensor& x) {
    if (size_ < array_.size()) {
      array_[size_++] = x;
    } else {
      array_.push_back(x);
      ++size_;
    }
  }
  void resize(size_t size) {
    if (array_.size() < size) {
      array_.resize(size);
    }
    size_ = size;
  }

  void emplace_back() { array_.emplace_back(); }

  void emplace_back(LoDTensor&& x) { array_.emplace_back(std::move(x)); }

  LoDTensor& back() { return array_.back(); }

  size_t space() const { return array_.size(); }

  void reserve(size_t size) {
    // Naive warning to tell user this array might be to large. The memory and
    // buffer used by this TensorArray will not be deleted during the training
    // and inference phase, so attention not to make it expand too long.
    if (size > 800UL) {
      LOG(WARNING) << "TensorArray has more than 800 items";
    }
    array_.reserve(size);
  }

  bool empty() const { return size_ == 0UL; }
  void clear() { size_ = 0UL; }

  LoDTensor& operator[](size_t id) { return array_[id]; }
  const LoDTensor& operator[](size_t id) const { return array_[id]; }
  LoDTensor& at(size_t id) { return array_.at(id); }
  const LoDTensor& at(size_t id) const { return array_.at(id); }

  size_t size() const { return size_; }

 private:
  size_t size_{0};
  std::vector<LoDTensor> array_;
};
#endif  // !PADDLE_ON_INFERENCE

}  // namespace framework
}  // namespace paddle
