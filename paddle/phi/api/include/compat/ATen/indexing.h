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
#include <optional>

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

}  // namespace at::indexing
