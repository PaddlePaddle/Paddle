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

#include <string>
#include <unordered_map>
#include <vector>
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/extended_tensor.h"

namespace phi {

template <typename T>
struct PhiVectorType;

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

  bool empty() const { return data_.empty(); }

  const T& back() const { return data_.back(); }

  T& back() { return data_.back(); }

  void resize(size_t size) { data_.resize(size); }

  void clear() { data_.clear(); }

  void emplace_back(const T& feed_data) { data_.emplace_back(feed_data); }

  void emplace_back() { data_.emplace_back(); }

  void push_back(const T& feed_data) { data_.push_back(feed_data); }

  void pop_back() { data_.pop_back(); }

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
};

}  // namespace phi

namespace paddle {
namespace framework {
template <typename T>
using PhiVector = phi::PhiVector<T>;

template <typename T>
using PhiVectorType = phi::PhiVectorType<T>;
}  // namespace framework
}  // namespace paddle
