/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include <codecvt>
#include <iostream>
#include <locale>
#include <string>
#include <unordered_map>
#include <vector>
#include "paddle/phi/core/extended_tensor.h"

namespace paddle {
namespace framework {

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

template <typename T>
struct PhiVectorType;

template <>
struct PhiVectorType<std::string> {
  const char* type_name = "PhiVectorString";
};

template <typename T>
class PhiVector : public phi::ExtendedTensor,
                  public phi::TypeInfoTraits<phi::TensorBase, PhiVector<T>> {
 public:
  PhiVector() = default;

  explicit PhiVector(const std::vector<T>& init_data) : data_(init_data) {}

  PhiVector(PhiVector&& other) = default;

  PhiVector(const PhiVector& other) = default;

  PhiVector& operator=(const PhiVector& other) = default;

  PhiVector& operator=(const std::vector<T>& other) {
    data_ = other;
    return *this;
  }

  PhiVector& operator=(PhiVector&& other) = default;

  /// \brief Destroy the PhiVector and release exclusive resources.
  virtual ~PhiVector() = default;

 public:
  /// \brief Returns the name of the class for type traits.
  /// \return The name of the class.
  static const char* name() { return PhiVectorType<T>().type_name; }

  size_t size() const { return data_.size(); }

  void resize(size_t size) { data_.resize(size); }

  void clear() { data_.clear(); }

  void emplace_back(const T& feed_data) { data_.emplace_back(feed_data); }

  const T& operator[](size_t index) const { return data_[index]; }

  T& operator[](size_t index) { return data_[index]; }

  T& at(size_t index) { return data_.at(index); }

  const T& at(size_t index) const { return data_.at(index); }

  typename std::vector<T>::iterator begin() { return data_.begin(); }

  typename std::vector<T>::const_iterator begin() const {
    return data_.begin();
  }

  typename std::vector<T>::iterator end() { return data_.end(); }

  typename std::vector<T>::const_iterator end() const { return data_.end(); }

 private:
  std::vector<T> data_;
  static std::string name_;
};

template <typename T>
std::string PhiVector<T>::name_ = "";  // NOLINT

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
}  // namespace framework
}  // namespace paddle
