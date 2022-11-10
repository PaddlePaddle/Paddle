/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/phi/core/extended_tensor.h"

namespace phi {

class Vocab : public ExtendedTensor, public TypeInfoTraits<TensorBase, Vocab> {
 public:
  Vocab() = default;

  Vocab(Vocab&& other) = default;

  Vocab(const Vocab& other) = default;

  Vocab& operator=(const Vocab& other) = default;

  Vocab& operator=(Vocab&& other) = default;

  /// \brief Destroy the Vocab and release exclusive resources.
  virtual ~Vocab() = default;

 public:
  /// \brief Returns the name of the class for type traits.
  /// \return The name of the class.
  static const char* name() { return "Vocab"; }

  size_t size() const { return data_.size(); }

  void clear() { data_.clear(); }

  void emplace(const std::wstring& key, std::int32_t value) {
    data_.emplace(key, value);
  }

  std::int32_t at(const std::wstring& key) { return data_.at(key); }

  std::int32_t at(const std::wstring& key) const { return data_.at(key); }

  std::unordered_map<std::wstring, std::int32_t>::iterator find(
      const std::wstring& key) {
    return data_.find(key);
  }

  std::unordered_map<std::wstring, std::int32_t>::const_iterator find(
      const std::wstring& key) const {
    return data_.find(key);
  }

  std::unordered_map<std::wstring, std::int32_t>::iterator begin() {
    return data_.begin();
  }

  std::unordered_map<std::wstring, std::int32_t>::const_iterator begin() const {
    return data_.begin();
  }

  std::unordered_map<std::wstring, std::int32_t>::iterator end() {
    return data_.end();
  }

  std::unordered_map<std::wstring, std::int32_t>::const_iterator end() const {
    return data_.end();
  }

 private:
  std::unordered_map<std::wstring, std::int32_t> data_;
};

}  // namespace phi
