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

/// Defines the Float4_e2m1fn_x2 type (4-bit floating-point, two elements packed
/// into one byte). This is the FP4 dtype from the OCP MX format spec
/// (https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf,
/// Section 5.3.3)
///
/// Given two high precision values val0 and val1, here is the
/// binary configuration of their packed representation, from MSB to LSB:
///
///   original value             | val1 : val0
///   ========================================
///   bit index (MSB==7, LSB==0) | 7654 : 3210
///   sign/exponent/mantissa     | seem : seem

struct alignas(1) Float4_e2m1fn_x2 {
  uint8_t val_;
  Float4_e2m1fn_x2() = default;
  explicit constexpr Float4_e2m1fn_x2(uint8_t val) : val_(val) {}
};

/// Comparison operators
inline bool operator==(const Float4_e2m1fn_x2& a, const Float4_e2m1fn_x2& b) {
  return a.val_ == b.val_;
}

inline bool operator!=(const Float4_e2m1fn_x2& a, const Float4_e2m1fn_x2& b) {
  return a.val_ != b.val_;
}

}  // namespace c10

namespace at {
using c10::Float4_e2m1fn_x2;
using c10::operator!=;
using c10::operator==;
}  // namespace at

namespace torch {
using c10::Float4_e2m1fn_x2;
using c10::operator!=;
using c10::operator==;
}  // namespace torch
