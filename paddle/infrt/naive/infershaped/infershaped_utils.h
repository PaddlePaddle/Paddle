// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <type_traits>
#include "paddle/infrt/tensor/dense_host_tensor.h"

namespace infrt {
namespace naive {
namespace infershaped {

using KeyType = const tensor::DenseHostTensor&;
using CountType = uint8_t;

constexpr CountType value(std::true_type) { return 1; }

constexpr CountType value(std::false_type) { return 0; }

template <typename T>
constexpr CountType value() {
  return value(std::integral_constant<bool, std::is_same<T, KeyType>::value>{});
}

template <typename FirstArg>
constexpr CountType count(CountType num) {
  return num;
}

template <typename FirstArg>
constexpr CountType count() {
  return 0;
}

template <>
constexpr CountType count<KeyType>(CountType num) {
  return num + 1;
}

template <>
constexpr CountType count<KeyType>() {
  return 1;
}

template <typename FirstArg, typename SecondArg, typename... RestOfArgs>
constexpr CountType count(CountType num) {
  return count<SecondArg, RestOfArgs...>(num + value<FirstArg>());
}

template <typename FirstArg, typename SecondArg, typename... RestOfArgs>
constexpr CountType count() {
  return count<SecondArg, RestOfArgs...>(value<FirstArg>());
}

}  // namespace infershaped

template <typename F>
struct InferShapeHelper;

template <typename Return, typename... Args>
struct InferShapeHelper<Return (*)(Args...)> {
  static constexpr int count = infershaped::count<Args...>();
};

}  // namespace naive
}  // namespace infrt
