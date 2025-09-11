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

#include <ATen/Tensor.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>

namespace at {

inline std::optional<Device> device_of(const Tensor& t) {
  if (t.defined()) {
    return t.device();
  } else {
    return std::nullopt;
  }
}

inline std::optional<Device> device_of(const std::optional<Tensor>& t) {
  return t.has_value() ? device_of(t.value()) : std::nullopt;
}

}  // namespace at
