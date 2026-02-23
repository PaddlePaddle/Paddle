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

#include <ATen/core/Tensor.h>
#include <ATen/core/TensorBase.h>
#include <c10/core/Scalar.h>
#include <c10/core/ScalarType.h>
#include <limits>
#include <optional>
#include "paddle/phi/api/include/tensor.h"

namespace at {

// Helper function implementations
namespace detail {
inline at::Scalar get_default_min_value(c10::ScalarType dtype) {
  switch (dtype) {
    case c10::ScalarType::Byte:
      return at::Scalar(static_cast<uint8_t>(0));
    case c10::ScalarType::Char:
      return at::Scalar(std::numeric_limits<int8_t>::lowest());
    case c10::ScalarType::Short:
      return at::Scalar(std::numeric_limits<int16_t>::lowest());
    case c10::ScalarType::Int:
      return at::Scalar(std::numeric_limits<int32_t>::lowest());
    case c10::ScalarType::Long:
      return at::Scalar(std::numeric_limits<int64_t>::lowest());
    case c10::ScalarType::UInt16:
      return at::Scalar(static_cast<uint16_t>(0));
    case c10::ScalarType::UInt32:
      return at::Scalar(static_cast<uint32_t>(0));
    case c10::ScalarType::UInt64:
      return at::Scalar(static_cast<uint64_t>(0));
    case c10::ScalarType::Half:
      return at::Scalar(-std::numeric_limits<float>::infinity());
    case c10::ScalarType::Float:
      return at::Scalar(-std::numeric_limits<float>::infinity());
    case c10::ScalarType::Double:
      return at::Scalar(-std::numeric_limits<double>::infinity());
    case c10::ScalarType::BFloat16:
      return at::Scalar(-std::numeric_limits<float>::infinity());
    case c10::ScalarType::Bool:
      return at::Scalar(false);
    default:
      return at::Scalar(-std::numeric_limits<double>::infinity());
  }
}

inline at::Scalar get_default_max_value(c10::ScalarType dtype) {
  switch (dtype) {
    case c10::ScalarType::Byte:
      return at::Scalar(std::numeric_limits<uint8_t>::max());
    case c10::ScalarType::Char:
      return at::Scalar(std::numeric_limits<int8_t>::max());
    case c10::ScalarType::Short:
      return at::Scalar(std::numeric_limits<int16_t>::max());
    case c10::ScalarType::Int:
      return at::Scalar(std::numeric_limits<int32_t>::max());
    case c10::ScalarType::Long:
      return at::Scalar(std::numeric_limits<int64_t>::max());
    case c10::ScalarType::UInt16:
      return at::Scalar(std::numeric_limits<uint16_t>::max());
    case c10::ScalarType::UInt32:
      return at::Scalar(std::numeric_limits<uint32_t>::max());
    case c10::ScalarType::UInt64:
      return at::Scalar(std::numeric_limits<uint64_t>::max());
    case c10::ScalarType::Half:
      return at::Scalar(std::numeric_limits<float>::infinity());
    case c10::ScalarType::Float:
      return at::Scalar(std::numeric_limits<float>::infinity());
    case c10::ScalarType::Double:
      return at::Scalar(std::numeric_limits<double>::infinity());
    case c10::ScalarType::BFloat16:
      return at::Scalar(std::numeric_limits<float>::infinity());
    case c10::ScalarType::Bool:
      return at::Scalar(true);
    default:
      return at::Scalar(std::numeric_limits<double>::infinity());
  }
}
}  // namespace detail

}  // namespace at

namespace at {

inline at::Tensor Tensor::clamp(const ::std::optional<at::Scalar>& min,
                                const ::std::optional<at::Scalar>& max) const {
  c10::ScalarType dtype = this->dtype();
  at::Scalar min_scalar =
      min.has_value() ? min.value() : detail::get_default_min_value(dtype);
  at::Scalar max_scalar =
      max.has_value() ? max.value() : detail::get_default_max_value(dtype);
  return Tensor(paddle::experimental::clip(tensor_, min_scalar, max_scalar));
}

inline at::Tensor Tensor::clamp(const ::std::optional<at::Tensor>& min,
                                const ::std::optional<at::Tensor>& max) const {
  PaddleTensor result = tensor_;
  if (min.has_value()) {
    result = paddle::experimental::maximum(result, min.value()._PD_GetInner());
  }
  if (max.has_value()) {
    result = paddle::experimental::minimum(result, max.value()._PD_GetInner());
  }
  return Tensor(result);
}

inline at::Tensor& Tensor::clamp_(
    const ::std::optional<at::Scalar>& min,
    const ::std::optional<at::Scalar>& max) const {
  c10::ScalarType dtype = this->dtype();
  at::Scalar min_scalar =
      min.has_value() ? min.value() : detail::get_default_min_value(dtype);
  at::Scalar max_scalar =
      max.has_value() ? max.value() : detail::get_default_max_value(dtype);
  paddle::experimental::clip_(
      const_cast<PaddleTensor&>(tensor_), min_scalar, max_scalar);
  return const_cast<at::Tensor&>(*this);
}

inline at::Tensor& Tensor::clamp_(
    const ::std::optional<at::Tensor>& min,
    const ::std::optional<at::Tensor>& max) const {
  if (min.has_value()) {
    PaddleTensor temp =
        paddle::experimental::maximum(tensor_, min.value()._PD_GetInner());
    const_cast<PaddleTensor&>(tensor_) = temp;
  }
  if (max.has_value()) {
    PaddleTensor temp =
        paddle::experimental::minimum(tensor_, max.value()._PD_GetInner());
    const_cast<PaddleTensor&>(tensor_) = temp;
  }
  return const_cast<at::Tensor&>(*this);
}

inline at::Tensor Tensor::clamp_max(const at::Scalar& max) const {
  return clamp(::std::nullopt, max);
}

inline at::Tensor Tensor::clamp_max(const at::Tensor& max) const {
  return Tensor(paddle::experimental::minimum(tensor_, max._PD_GetInner()));
}

inline at::Tensor& Tensor::clamp_max_(const at::Scalar& max) const {
  return clamp_(::std::nullopt, max);
}

inline at::Tensor& Tensor::clamp_max_(const at::Tensor& max) const {
  PaddleTensor temp =
      paddle::experimental::minimum(tensor_, max._PD_GetInner());
  const_cast<PaddleTensor&>(tensor_) = temp;
  return const_cast<at::Tensor&>(*this);
}

inline at::Tensor Tensor::clamp_min(const at::Scalar& min) const {
  return clamp(min, ::std::nullopt);
}

inline at::Tensor Tensor::clamp_min(const at::Tensor& min) const {
  return Tensor(paddle::experimental::maximum(tensor_, min._PD_GetInner()));
}

inline at::Tensor& Tensor::clamp_min_(const at::Scalar& min) const {
  return clamp_(min, ::std::nullopt);
}

inline at::Tensor& Tensor::clamp_min_(const at::Tensor& min) const {
  PaddleTensor temp =
      paddle::experimental::maximum(tensor_, min._PD_GetInner());
  const_cast<PaddleTensor&>(tensor_) = temp;
  return const_cast<at::Tensor&>(*this);
}

inline at::Tensor Tensor::operator[](int64_t index) const {
  return paddle::experimental::slice(tensor_,
                                     /*axes=*/{0},
                                     /*starts=*/{index},
                                     /*ends=*/{index + 1},
                                     /*infer_flags=*/{1},
                                     /*decrease_axis=*/{0});
}

}  // namespace at
