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

#include <cstddef>
#include <cstdint>
#include <functional>

namespace c10 {

enum class DeviceType : int8_t {
  CPU = 0,
  CUDA = 1,
  XPU = 12,
  IPU = 18,
  CUSTOM = 20,
  PrivateUse1 = CUSTOM,
};

constexpr DeviceType kCUDA = DeviceType::CUDA;
constexpr DeviceType kCPU = DeviceType::CPU;
constexpr DeviceType kCUSTOM = DeviceType::CUSTOM;
constexpr DeviceType kXPU = DeviceType::XPU;
constexpr DeviceType kIPU = DeviceType::IPU;
constexpr DeviceType kPrivateUse1 = DeviceType::PrivateUse1;

}  // namespace c10

namespace std {
template <>
struct hash<c10::DeviceType> {
  std::size_t operator()(c10::DeviceType k) const noexcept {
    return std::hash<int>()(static_cast<int>(k));
  }
};
}  // namespace std

namespace at {
using c10::DeviceType;
using c10::kCPU;
using c10::kCUDA;
using c10::kCUSTOM;
using c10::kIPU;
using c10::kPrivateUse1;
using c10::kXPU;
}  // namespace at
