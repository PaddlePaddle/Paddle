// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
#include <c10/core/SymInt.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace at {
class Tensor;
}

namespace at::indexing {

constexpr int64_t INDEX_MIN = std::numeric_limits<int64_t>::min();
constexpr int64_t INDEX_MAX = std::numeric_limits<int64_t>::max();

enum class TensorIndexType { None, Ellipsis, SymInt, Boolean, Slice, Tensor };

constexpr std::nullopt_t None = std::nullopt;

struct EllipsisIndexType final {
  EllipsisIndexType() = default;
};

const EllipsisIndexType Ellipsis = EllipsisIndexType();

struct Slice final {
 public:
  Slice(std::optional<c10::SymInt> start_index = std::nullopt,
        std::optional<c10::SymInt> stop_index = std::nullopt,
        std::optional<c10::SymInt> step_index = std::nullopt) {
    if (!step_index.has_value()) {
      step_ = c10::SymInt(1);
    } else {
      step_ = std::move(step_index).value();
    }

    if (!start_index.has_value()) {
      start_ = c10::SymInt(step_ < 0 ? INDEX_MAX : 0);
    } else {
      start_ = std::move(start_index).value();
    }

    if (!stop_index.has_value()) {
      stop_ = c10::SymInt(step_ < 0 ? INDEX_MIN : INDEX_MAX);
    } else {
      stop_ = std::move(stop_index).value();
    }
  }

  inline c10::SymInt start() const { return start_; }
  inline c10::SymInt stop() const { return stop_; }
  inline c10::SymInt step() const { return step_; }

 private:
  c10::SymInt start_;
  c10::SymInt stop_;
  c10::SymInt step_;
};

struct TensorIndex final {
  TensorIndex(std::nullopt_t /*unused*/) : type_(TensorIndexType::None) {}

  TensorIndex(at::indexing::EllipsisIndexType /*unused*/)
      : type_(TensorIndexType::Ellipsis) {}
  TensorIndex(const char* str) : TensorIndex(at::indexing::Ellipsis) {
    if (std::strcmp(str, "...") != 0) {
      throw std::invalid_argument(
          "Expected \"...\" to represent an ellipsis index.");
    }
  }

  TensorIndex(c10::SymInt integer)
      : integer_(std::move(integer)), type_(TensorIndexType::SymInt) {}
  TensorIndex(int64_t integer) : TensorIndex(c10::SymInt(integer)) {}
  TensorIndex(int integer) : TensorIndex(c10::SymInt(integer)) {}

  template <class T, class = std::enable_if_t<std::is_same_v<bool, T>>>
  TensorIndex(T boolean) : boolean_(boolean), type_(TensorIndexType::Boolean) {}

  TensorIndex(Slice slice)
      : slice_(std::move(slice)), type_(TensorIndexType::Slice) {}

  TensorIndex(const at::Tensor& tensor);

  inline bool is_none() const { return type_ == TensorIndexType::None; }
  inline bool is_ellipsis() const { return type_ == TensorIndexType::Ellipsis; }
  inline bool is_integer() const { return type_ == TensorIndexType::SymInt; }
  inline c10::SymInt integer() const { return integer_; }
  inline bool is_boolean() const { return type_ == TensorIndexType::Boolean; }
  inline bool boolean() const { return boolean_; }
  inline bool is_slice() const { return type_ == TensorIndexType::Slice; }
  inline const Slice& slice() const { return slice_; }
  inline bool is_tensor() const { return type_ == TensorIndexType::Tensor; }
  const at::Tensor& tensor() const;

 private:
  c10::SymInt integer_ = 0;
  bool boolean_ = false;
  Slice slice_;
  std::shared_ptr<at::Tensor> tensor_;
  TensorIndexType type_;
};

}  // namespace at::indexing
