// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

// #The file has been adapted from pytorch project
// #Licensed under  BSD-style license -
// https://github.com/pytorch/pytorch/blob/main/LICENSE

#pragma once

#include <vector>

namespace c10 {

// c10::List is a type-safe wrapper around std::vector for PyTorch compatibility
template <typename T>
class List {
 public:
  using value_type = T;
  using size_type = typename std::vector<T>::size_type;
  using iterator = typename std::vector<T>::iterator;
  using const_iterator = typename std::vector<T>::const_iterator;
  using reference = typename std::vector<T>::reference;
  using const_reference = typename std::vector<T>::const_reference;

  List() = default;
  List(std::initializer_list<T> init) : vec_(init) {}
  explicit List(size_type count) : vec_(count) {}
  List(size_type count, const T& value) : vec_(count, value) {}
  template <typename InputIt>
  List(InputIt first, InputIt last) : vec_(first, last) {}
  List(const std::vector<T>& vec) : vec_(vec) {}        // NOLINT
  List(std::vector<T>&& vec) : vec_(std::move(vec)) {}  // NOLINT

  // Conversion to std::vector
  const std::vector<T>& vec() const { return vec_; }
  std::vector<T>& vec() { return vec_; }
  operator const std::vector<T>&() const { return vec_; }  // NOLINT
  operator std::vector<T>&() { return vec_; }              // NOLINT

  // Standard vector-like interface
  size_type size() const { return vec_.size(); }
  bool empty() const { return vec_.empty(); }
  void clear() { vec_.clear(); }
  void reserve(size_type new_cap) { vec_.reserve(new_cap); }
  size_type capacity() const { return vec_.capacity(); }

  reference operator[](size_type pos) { return vec_[pos]; }
  const_reference operator[](size_type pos) const { return vec_[pos]; }
  reference at(size_type pos) { return vec_.at(pos); }
  const_reference at(size_type pos) const { return vec_.at(pos); }
  reference front() { return vec_.front(); }
  const_reference front() const { return vec_.front(); }
  reference back() { return vec_.back(); }
  const_reference back() const { return vec_.back(); }

  void push_back(const T& value) { vec_.push_back(value); }
  void push_back(T&& value) { vec_.push_back(std::move(value)); }
  template <typename... Args>
  reference emplace_back(Args&&... args) {
    return vec_.emplace_back(std::forward<Args>(args)...);
  }
  void pop_back() { vec_.pop_back(); }

  iterator begin() { return vec_.begin(); }
  const_iterator begin() const { return vec_.begin(); }
  const_iterator cbegin() const { return vec_.cbegin(); }
  iterator end() { return vec_.end(); }
  const_iterator end() const { return vec_.end(); }
  const_iterator cend() const { return vec_.cend(); }

  void resize(size_type count) { vec_.resize(count); }
  void resize(size_type count, const T& value) { vec_.resize(count, value); }

  bool operator==(const List<T>& other) const { return vec_ == other.vec_; }
  bool operator!=(const List<T>& other) const { return vec_ != other.vec_; }

 private:
  std::vector<T> vec_;
};

}  // namespace c10
