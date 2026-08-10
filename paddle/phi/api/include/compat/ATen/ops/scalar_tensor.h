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

#pragma once

#include <ATen/ops/full.h>
#include <c10/core/Scalar.h>
#include <c10/core/TensorOptions.h>

#include <optional>

namespace at {

inline at::Tensor scalar_tensor(const at::Scalar& scalar,
                                at::TensorOptions options = {}) {
  return at::full({}, scalar, options);
}

inline at::Tensor scalar_tensor(const at::Scalar& scalar,
                                ::std::optional<at::ScalarType> dtype,
                                ::std::optional<at::Layout> layout,
                                ::std::optional<at::Device> device,
                                ::std::optional<bool> pin_memory) {
  auto options =
      at::TensorOptions()
          .dtype(dtype.value_or(c10::get_default_dtype_as_scalartype()))
          .layout(layout)
          .device(device.value_or(at::kCPU))
          .pinned_memory(pin_memory);
  return scalar_tensor(scalar, options);
}

}  // namespace at
