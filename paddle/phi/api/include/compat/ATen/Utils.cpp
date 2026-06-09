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

#include <ATen/Utils.h>

#include <ATen/ops/empty.h>
#include <ATen/ops/to.h>
#include <c10/core/Layout.h>
#include <c10/core/ScalarType.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Exception.h>
#include <c10/util/accumulate.h>

#include <algorithm>

#include "paddle/common/macros.h"
#include "paddle/phi/api/include/sparse_api.h"
#include "paddle/phi/api/include/tensor.h"

namespace at {
namespace detail {

template <typename T>
Tensor tensor_cpu(ArrayRef<T> values, const TensorOptions& options) {
  constexpr auto native_scalar_type = c10::CppTypeToScalarType<T>::value;
  auto result = at::empty(values.size(), options.dtype(native_scalar_type));
  PD_CHECK(result.is_contiguous());
  std::copy(values.begin(), values.end(), result.template data_ptr<T>());
  if (options.dtype() != native_scalar_type) {
    return result.to(at::TensorOptions().dtype(options.dtype()));
  }
  return result;
}

template <typename T>
Tensor tensor_backend(ArrayRef<T> values, const TensorOptions& options) {
  auto cpu_tensor =
      tensor_cpu(values, options.device(c10::Device(c10::DeviceType::CPU)));
  return cpu_tensor.to(options.device());
}

template <typename T>
Tensor tensor_complex_cpu(ArrayRef<T> values, const TensorOptions& options) {
  constexpr auto native_scalar_type = c10::CppTypeToScalarType<T>::value;
  auto result = at::empty(values.size(), options.dtype(native_scalar_type));
  PD_CHECK(result.is_contiguous());
  std::copy(values.begin(), values.end(), result.template data_ptr<T>());
  if (options.dtype() != native_scalar_type) {
    return result.to(at::TensorOptions().dtype(options.dtype()));
  }
  return result;
}

template <typename T>
Tensor tensor_complex_backend(ArrayRef<T> values,
                              const TensorOptions& options) {
  auto cpu_tensor = tensor_complex_cpu(
      values, options.device(c10::Device(c10::DeviceType::CPU)));
  return cpu_tensor.to(options.device());
}

}  // namespace detail

#define TENSOR(T, _1)                                                          \
  PADDLE_API Tensor tensor(ArrayRef<T> values, const TensorOptions& options) { \
    if (options.device().type() != c10::DeviceType::CPU) {                     \
      return at::detail::tensor_backend(values, options);                      \
    } else {                                                                   \
      return at::detail::tensor_cpu(values, options);                          \
    }                                                                          \
  }
AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, TENSOR)
#undef TENSOR

#define TENSOR(T, _1)                                                          \
  PADDLE_API Tensor tensor(ArrayRef<T> values, const TensorOptions& options) { \
    if (options.device().type() != c10::DeviceType::CPU) {                     \
      return at::detail::tensor_complex_backend(values, options);              \
    } else {                                                                   \
      return at::detail::tensor_complex_cpu(values, options);                  \
    }                                                                          \
  }
AT_FORALL_COMPLEX_TYPES(TENSOR)
#undef TENSOR

}  // namespace at
