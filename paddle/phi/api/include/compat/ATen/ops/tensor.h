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
// #Licensed under   BSD-style license -
// https://github.com/pytorch/pytorch/blob/main/LICENSE

#pragma once
#include <ATen/core/Tensor.h>
#include <c10/core/ScalarType.h>

namespace at {

#define TENSOR(T, S)                                               \
  Tensor tensor(ArrayRef<T> values, const TensorOptions& options); \
  inline Tensor tensor(std::initializer_list<T> values,            \
                       const TensorOptions& options) {             \
    return at::tensor(ArrayRef<T>(values), options);               \
  }                                                                \
  inline Tensor tensor(T value, const TensorOptions& options) {    \
    return at::tensor(ArrayRef<T>(value), options);                \
  }                                                                \
  inline Tensor tensor(ArrayRef<T> values) {                       \
    return at::tensor(std::move(values), at::dtype(k##S));         \
  }                                                                \
  inline Tensor tensor(std::initializer_list<T> values) {          \
    return at::tensor(ArrayRef<T>(values));                        \
  }                                                                \
  inline Tensor tensor(T value) { return at::tensor(ArrayRef<T>(value)); }
AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, TENSOR)
AT_FORALL_COMPLEX_TYPES(TENSOR)
#undef TENSOR

}  // namespace at
