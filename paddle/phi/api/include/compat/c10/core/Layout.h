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

// #The file has been adapted from pytorch project
// #Licensed under  BSD-style license -
// https://github.com/pytorch/pytorch/blob/main/LICENSE

#pragma once

#include <c10/util/Exception.h>

#include <cstdint>
#include <ostream>

namespace c10 {
enum class Layout : int8_t {
  Strided,
  Sparse,
  SparseCsr,
  Mkldnn,
  SparseCsc,
  SparseBsr,
  SparseBsc,
  Jagged,
  NumOptions
};

constexpr auto kStrided = Layout::Strided;
constexpr auto kSparse = Layout::Sparse;
constexpr auto kSparseCsr = Layout::SparseCsr;
constexpr auto kMkldnn = Layout::Mkldnn;
constexpr auto kSparseCsc = Layout::SparseCsc;
constexpr auto kSparseBsr = Layout::SparseBsr;
constexpr auto kSparseBsc = Layout::SparseBsc;
constexpr auto kJagged = Layout::Jagged;

inline std::ostream& operator<<(std::ostream& stream, c10::Layout layout) {
  switch (layout) {
    case c10::kStrided:
      return stream << "Strided";
    case c10::kSparse:
      return stream << "Sparse";
    case c10::kSparseCsr:
      return stream << "SparseCsr";
    case c10::kSparseCsc:
      return stream << "SparseCsc";
    case c10::kSparseBsr:
      return stream << "SparseBsr";
    case c10::kSparseBsc:
      return stream << "SparseBsc";
    case c10::kMkldnn:
      return stream << "Mkldnn";
    case c10::kJagged:
      return stream << "Jagged";
    default:
      TORCH_CHECK(false, "Unknown layout");
  }
}

}  // namespace c10

namespace at {
using c10::kJagged;
using c10::kMkldnn;
using c10::kSparse;
using c10::kSparseBsc;
using c10::kSparseBsr;
using c10::kSparseCsc;
using c10::kSparseCsr;
using c10::kStrided;

using c10::Layout;
}  // namespace at
namespace torch {
using c10::kJagged;
using c10::kMkldnn;
using c10::kSparse;
using c10::kSparseBsc;
using c10::kSparseBsr;
using c10::kSparseCsc;
using c10::kSparseCsr;
using c10::kStrided;

using c10::Layout;
}  // namespace torch
