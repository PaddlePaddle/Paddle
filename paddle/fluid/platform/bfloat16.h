// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#ifdef _WIN32
#define GLOG_NO_ABBREVIATED_SEVERITIES  // msvc conflict logging with windows.h
#ifndef NOMINMAX
#define NOMINMAX  // msvc max/min macro conflict with std::min/max
#endif
#endif

#include "paddle/utils/Core"

namespace paddle {
namespace platform {
using bfloat16 = Eigen::bfloat16;
}  // namespace platform
}  // namespace paddle

inline void* memset(paddle::platform::bfloat16* ptr, int value, size_t num) {
  return memset(reinterpret_cast<void*>(ptr), value, num);
}

namespace std {
template <>
struct is_pod<paddle::platform::bfloat16> {
  static const bool value = true;
};

template <>
struct is_floating_point<paddle::platform::bfloat16>
    : std::integral_constant<
          bool, std::is_same<paddle::platform::bfloat16,
                             typename std::remove_cv<
                                 paddle::platform::bfloat16>::type>::value> {};

template <>
struct is_signed<paddle::platform::bfloat16> {
  static const bool value = true;
};

template <>
struct is_unsigned<paddle::platform::bfloat16> {
  static const bool value = false;
};
}  // namespace std
