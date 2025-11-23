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

#include <ATen/ATen.h>
#include <ATen/Functions.h>
#include <ATen/core/TensorBody.h>
#include <c10/core/ScalarType.h>
#include <c10/util/OptionalArrayRef.h>

namespace torch {

using namespace at;  // NOLINT

using std::nullopt;   // NOLINT
using std::optional;  // NOLINT

using Dtype = at::ScalarType;

constexpr auto kUInt8 = at::kByte;
constexpr auto kInt8 = at::kChar;
constexpr auto kInt16 = at::kShort;
constexpr auto kInt32 = at::kInt;
constexpr auto kInt64 = at::kLong;
constexpr auto kUInt16 = at::kUInt16;
constexpr auto kUInt32 = at::kUInt32;

constexpr auto kFloat16 = at::kHalf;
constexpr auto kFloat32 = at::kFloat;
constexpr auto kFloat64 = at::kDouble;
constexpr auto kBFloat16 = at::kBFloat16;

constexpr auto kU8 = kUInt8;
constexpr auto kU16 = kUInt16;
constexpr auto kU32 = kUInt32;
constexpr auto kI8 = kInt8;
constexpr auto kI16 = kInt16;
constexpr auto kI32 = kInt32;
constexpr auto kI64 = kInt64;
constexpr auto kF16 = kFloat16;
constexpr auto kF32 = kFloat32;
constexpr auto kF64 = kFloat64;

}  // namespace torch
