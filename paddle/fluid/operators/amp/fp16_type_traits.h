/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {
namespace details {

template <typename T>
class MPTypeTrait {
 public:
  using Type = T;
};

template <>
class MPTypeTrait<platform::float16> {
 public:
  using Type = float;
};

template <typename T, int N>
struct GetVecType;

template <typename T>
struct GetVecType<T, 1> {
  using Type = T;
};

template <>
struct GetVecType<paddle::platform::float16, 2> {
  using Type = half2;
};

template <>
struct GetVecType<paddle::platform::float16, 4> {
  using Type = float2;
};

template <>
struct GetVecType<float, 2> {
  using Type = float2;
};

template <>
struct GetVecType<float, 4> {
  using Type = float4;
};

template <>
struct GetVecType<double, 2> {
  using Type = double2;
};

template <>
struct GetVecType<double, 4> {
  using Type = double4;
};

}  // namespace details
}  // namespace operators
}  // namespace paddle
