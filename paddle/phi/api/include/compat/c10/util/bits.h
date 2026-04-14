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

struct bits1x8 {
  constexpr bits1x8() = default;
  explicit constexpr bits1x8(uint8_t value) : val_(value) {}

  uint8_t val_{0};
};

struct bits2x4 {
  constexpr bits2x4() = default;
  explicit constexpr bits2x4(uint8_t value) : val_(value) {}

  uint8_t val_{0};
};

struct bits4x2 {
  constexpr bits4x2() = default;
  explicit constexpr bits4x2(uint8_t value) : val_(value) {}

  uint8_t val_{0};
};

struct bits8 {
  constexpr bits8() = default;
  explicit constexpr bits8(uint8_t value) : val_(value) {}

  uint8_t val_{0};
};

struct bits16 {
  constexpr bits16() = default;
  explicit constexpr bits16(uint16_t value) : val_(value) {}

  uint16_t val_{0};
};

}  // namespace c10

namespace at {
using c10::bits16;
using c10::bits1x8;
using c10::bits2x4;
using c10::bits4x2;
using c10::bits8;
}  // namespace at

namespace torch {
using c10::bits16;
using c10::bits1x8;
using c10::bits2x4;
using c10::bits4x2;
using c10::bits8;
}  // namespace torch
