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

#include <unordered_map>

#include "paddle/phi/core/extended_tensor.h"

namespace phi {

class CPlusString : public ExtendedTensor,
                    public TypeInfoTraits<TensorBase, CPlusString> {
 public:
  CPlusString() = default;

  CPlusString(CPlusString&& other) = default;

  CPlusString(const CPlusString& other) = default;

  CPlusString& operator=(const CPlusString& other) = default;

  CPlusString& operator=(const std::string& other) {
    this->data_ = other;
    return *this;
  }

  CPlusString& operator=(std::string&& other) {
    this->data_ = other;
    return *this;
  }

  CPlusString& operator=(CPlusString&& other) = default;

  /// \brief Destroy the CPlusString and release exclusive resources.
  virtual ~CPlusString() = default;

 public:
  /// \brief Returns the name of the class for type traits.
  /// \return The name of the class.
  static const char* name() { return "CPlusString"; }

  size_t size() const { return data_.size(); }

  void clear() { data_.clear(); }

  std::string::iterator begin() { return data_.begin(); }

  std::string* Get() { return &data_; }

  const std::string& Get() const { return data_; }

  std::string::const_iterator begin() const { return data_.begin(); }

  std::string::iterator end() { return data_.end(); }

  std::string::const_iterator end() const { return data_.end(); }

 private:
  std::string data_;
};

}  // namespace phi
