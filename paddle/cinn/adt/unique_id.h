// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <atomic>
#include <cstddef>
#include <functional>

namespace cinn::adt {

class UniqueId final {
 public:
  UniqueId() : unique_id_(NewSeqNumber()) {}
  UniqueId(const UniqueId&) = default;
  UniqueId(UniqueId&&) = default;
  UniqueId& operator=(const UniqueId&) = default;
  UniqueId& operator=(UniqueId&&) = default;

  static UniqueId New() { return UniqueId{NewSeqNumber()}; }

  bool operator==(const UniqueId& other) const {
    return this->unique_id_ == other.unique_id_;
  }

  bool operator!=(const UniqueId& other) const {
    return !this->operator==(other);
  }

  bool operator<(const UniqueId& other) const {
    return this->unique_id_ < other.unique_id_;
  }

  // For unit test only
  static void ResetSeqNumber(std::size_t init) { *MutSeqNumber() = init; }

  std::size_t unique_id() const { return unique_id_; }

 private:
  static std::size_t NewSeqNumber() { return ++*MutSeqNumber(); }

  static std::atomic<std::size_t>* MutSeqNumber() {
    static std::atomic<std::size_t> seq_number{0};
    return &seq_number;
  }

  explicit UniqueId(std::size_t unique_id) : unique_id_(unique_id) {}
  std::size_t unique_id_;
};

}  // namespace cinn::adt

namespace std {

template <>
struct hash<cinn::adt::UniqueId> final {
  std::size_t operator()(const cinn::adt::UniqueId& unique_id) const {
    return unique_id.unique_id();
  }
};

}  // namespace std
