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

#include <codecvt>
#include <iostream>
#include <locale>
#include <string>
#include <unordered_map>
#include <vector>
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/extended_tensor.h"
#include "paddle/phi/core/vocab/phi_tensor_base_vector.h"

namespace phi {
template <>
struct PhiVectorType<std::string> {
  const char* type_name = "PhiVectorString";
};

// Note(YuanRisheng): Vocab is mainly used for faster_tokenizer_op and we don't
// recommend widely use it. Because faster_tokenizer_op may be deleted in the
// future and this class will be deleted.

class Vocab : public phi::ExtendedTensor,
              public phi::TypeInfoTraits<phi::TensorBase, Vocab> {
 public:
  Vocab() = default;

  Vocab(Vocab&& other) = default;

  Vocab(const Vocab& other) = default;

  Vocab& operator=(const Vocab& other) = default;

  Vocab& operator=(Vocab&& other) = default;

  Vocab& operator=(
      const std::unordered_map<std::wstring, std::int32_t>& other) {
    this->data_ = other;
    return *this;
  }

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

// Note(YuanRisheng): PhiVector is essentially a vector that only used for PHI
// Kernel. It can be used when you define a non-tensor type that needs to be
// stored in a vector as PHI kernel argument.

using String = std::string;
using Strings = PhiVector<std::string>;

// Convert the std::string type to the std::string type.
bool ConvertStrToWstr(const std::string& src, std::wstring* res);
// Convert the std::wstring type to the std::string type.
void ConvertWstrToStr(const std::wstring& src, std::string* res);
// Normalization Form Canonical Decomposition.
void NFD(const std::string& s, std::string* ret);

// Write the data which is type of
// std::unordered_map<td::string, int32_t> to ostream.
void StringMapToStream(std::ostream& os,
                       const std::unordered_map<std::string, int32_t>& data);

// Read the data which is type of
// std::unordered_map<td::string, int32_t> from istream.
void StringMapFromStream(std::istream& is,
                         std::unordered_map<std::string, int32_t>* data);
}  // namespace phi

namespace paddle {
namespace framework {
using Vocab = phi::Vocab;
using Strings = phi::Strings;
using String = phi::String;
using phi::ConvertStrToWstr;
using phi::ConvertWstrToStr;
using phi::NFD;
using phi::StringMapFromStream;
using phi::StringMapToStream;
}  // namespace framework
}  // namespace paddle
