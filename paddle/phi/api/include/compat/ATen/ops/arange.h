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

// The file has been adapted from pytorch project
// Licensed under BSD-style license -
// https://github.com/pytorch/pytorch/blob/main/LICENSE

#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/native/RangeUtils.h>
#include <c10/core/TensorOptions.h>
#include <utils/pinned_place.h>
#include <optional>

#include "paddle/phi/api/include/api.h"
#include "paddle/phi/common/place.h"

namespace at {

namespace detail {

inline bool _PD_IsIntegralArangeScalar(const at::Scalar& scalar) {
  switch (scalar.dtype()) {
    case phi::DataType::BOOL:
    case phi::DataType::UINT8:
    case phi::DataType::INT8:
    case phi::DataType::UINT16:
    case phi::DataType::INT16:
    case phi::DataType::UINT32:
    case phi::DataType::INT32:
    case phi::DataType::UINT64:
    case phi::DataType::INT64:
      return true;
    default:
      return false;
  }
}

inline at::ScalarType _PD_ResolveArangeDtype(const at::Scalar& start,
                                             const at::Scalar& end,
                                             const at::Scalar& step,
                                             const at::TensorOptions& options) {
  if (options.has_dtype()) {
    return options.dtype().toScalarType();
  }
  if (_PD_IsIntegralArangeScalar(start) && _PD_IsIntegralArangeScalar(end) &&
      _PD_IsIntegralArangeScalar(step)) {
    return at::kLong;
  }
  return c10::get_default_dtype_as_scalartype();
}

inline paddle::Tensor _PD_MakeArangeScalarTensor(const at::Scalar& scalar,
                                                 phi::DataType dtype,
                                                 const phi::Place& place) {
  return paddle::experimental::full({}, scalar, dtype, place);
}

}  // namespace detail

inline at::Tensor arange(const at::Scalar& start,
                         const at::Scalar& end,
                         const at::Scalar& step,
                         at::TensorOptions options = {}) {
  // Match PyTorch: step must be non-zero and consistent with (end - start).
  at::native::arange_check_bounds(start, end, step);
  auto dtype = detail::_PD_ResolveArangeDtype(start, end, step, options);
  auto pd_dtype = compat::_PD_AtenScalarTypeToPhiDataType(dtype);
  if (options.pinned_memory()) {
    // Pinning memory is only supported for CPU tensors
    if (options.has_device() && !options.device().is_cpu()) {
      PD_THROW(
          "pin_memory=true requires device to be CPU, but got non-CPU device");
    }
    phi::Place base_place = options._PD_GetPlace();
    phi::Place pinned_place = compat::_PD_GetCreatePinnedPlace(base_place);
    auto dense = paddle::experimental::arange(
        detail::_PD_MakeArangeScalarTensor(start, pd_dtype, phi::CPUPlace()),
        detail::_PD_MakeArangeScalarTensor(end, pd_dtype, phi::CPUPlace()),
        detail::_PD_MakeArangeScalarTensor(step, pd_dtype, phi::CPUPlace()),
        pd_dtype,
        phi::CPUPlace());
    return dense.copy_to(pinned_place, /*blocking=*/true);
  }
  return paddle::experimental::arange(
      detail::_PD_MakeArangeScalarTensor(
          start, pd_dtype, options._PD_GetPlace()),
      detail::_PD_MakeArangeScalarTensor(end, pd_dtype, options._PD_GetPlace()),
      detail::_PD_MakeArangeScalarTensor(
          step, pd_dtype, options._PD_GetPlace()),
      pd_dtype,
      options._PD_GetPlace());
}

inline at::Tensor arange(const at::Scalar& end,
                         at::TensorOptions options = {}) {
  return arange(/*start=*/0, end, /*step=*/1, options);
}

inline at::Tensor arange(const at::Scalar& end,
                         ::std::optional<at::ScalarType> dtype,
                         ::std::optional<at::Layout> layout,
                         ::std::optional<at::Device> device,
                         ::std::optional<bool> pin_memory) {
  auto options = at::TensorOptions()
                     .dtype(dtype)
                     .layout(layout)
                     .device(device)
                     .pinned_memory(pin_memory);
  return arange(/*start=*/0, end, /*step=*/1, options);
}

inline at::Tensor arange(const at::Scalar& start,
                         const at::Scalar& end,
                         at::TensorOptions options = {}) {
  return arange(start, end, /*step=*/1, options);
}

inline at::Tensor arange(const at::Scalar& start,
                         const at::Scalar& end,
                         ::std::optional<at::ScalarType> dtype,
                         ::std::optional<at::Layout> layout,
                         ::std::optional<at::Device> device,
                         ::std::optional<bool> pin_memory) {
  auto options = at::TensorOptions()
                     .dtype(dtype)
                     .layout(layout)
                     .device(device)
                     .pinned_memory(pin_memory);
  return arange(start, end, /*step=*/1, options);
}

inline at::Tensor arange(const at::Scalar& start,
                         const at::Scalar& end,
                         const at::Scalar& step,
                         ::std::optional<at::ScalarType> dtype,
                         ::std::optional<at::Layout> layout,
                         ::std::optional<at::Device> device,
                         ::std::optional<bool> pin_memory) {
  auto options = at::TensorOptions()
                     .dtype(dtype)
                     .layout(layout)
                     .device(device)
                     .pinned_memory(pin_memory);
  return arange(start, end, step, options);
}

}  // namespace at
