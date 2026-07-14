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

// The file has been adapted from pytorch project
// Licensed under BSD-style license -
// https://github.com/pytorch/pytorch/blob/main/LICENSE

#pragma once

#include <cstdint>

namespace c10 {

struct Float8_e4m3fnuz {
  constexpr Float8_e4m3fnuz() = default;
  explicit constexpr Float8_e4m3fnuz(uint8_t value) : x(value) {}

  uint8_t x{0};
};

}  // namespace c10

namespace at {
using c10::Float8_e4m3fnuz;
}  // namespace at

namespace torch {
using c10::Float8_e4m3fnuz;
}  // namespace torch
