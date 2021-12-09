/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/pten/core/tensor_meta.h"

namespace pten {

DenseTensorMeta::DenseTensorMeta(DataType dtype, const DDim& dims)
    : dims(dims), dtype(dtype) {}

DenseTensorMeta::DenseTensorMeta(DataType dtype,
                                 const DDim& dims,
                                 DataLayout layout)
    : dims(dims), dtype(dtype), layout(layout) {}

DenseTensorMeta::DenseTensorMeta(DataType dtype,
                                 const DDim& dims,
                                 DataLayout layout,
                                 const std::vector<std::vector<size_t>>& lod)
    : dims(dims), dtype(dtype), layout(layout), lod(lod) {}

bool DenseTensorMeta::valid() const noexcept {
  bool valid{true};
  valid = valid && (dtype != DataType::UNDEFINED);
  valid = valid && (layout != DataLayout::UNDEFINED);
  valid = valid && (is_scalar || product(dims) >= 0);
  return valid;
}

bool operator==(const DenseTensorMeta& lhs, const DenseTensorMeta& rhs) {
  bool ret = true;
  return ret && (lhs.is_scalar == rhs.is_scalar) && (lhs.dims == rhs.dims) &&
         (lhs.dtype == rhs.dtype) && (lhs.layout == rhs.layout) &&
         (lhs.lod == rhs.lod) && (lhs.offset == rhs.offset);
}
}  // namespace pten
