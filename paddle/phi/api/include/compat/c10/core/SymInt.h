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
#include <c10/util/accumulate.h>
#include <cstdint>
#include <optional>

namespace c10 {

class SymInt {
 public:
  SymInt() : data_(0) {}
  /*implicit*/ SymInt(int64_t d) : data_(d) {}  // NOLINT
  /*implicit*/ operator int64_t() const { return data_; }

  int64_t guard_int(const char* file, int64_t line) const {
    (void)file;
    (void)line;
    return data_;
  }

  std::optional<int64_t> maybe_as_int() const { return data_; }

 private:
  int64_t data_;
};

}  // namespace c10
