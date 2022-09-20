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

#include "paddle/phi/core/tensor_meta.h"

namespace phi {

DenseTensorMeta::DenseTensorMeta(DataType dtype, const DDim& dims)
    : dims(dims), dtype(dtype) {}

DenseTensorMeta::DenseTensorMeta(DataType dtype,
                                 const DDim& dims,
                                 DataLayout layout,
                                 size_t offset)
    : dims(dims), dtype(dtype), layout(layout), offset(offset) {}

DenseTensorMeta::DenseTensorMeta(DataType dtype,
                                 const DDim& dims,
                                 DataLayout layout,
                                 const LoD& lod,
                                 size_t offset)
    : dims(dims), dtype(dtype), layout(layout), lod(lod), offset(offset) {}

bool DenseTensorMeta::valid() const noexcept {
  bool valid{true};
  valid = valid && (dtype != DataType::UNDEFINED);
  valid = valid && (layout != DataLayout::UNDEFINED);
  valid = valid && (is_scalar || product(dims) >= 0);
  return valid;
}

StringTensorMeta::StringTensorMeta(const DDim& dims) : dims(dims) {}

bool StringTensorMeta::valid() const noexcept {
  bool valid{true};
  valid = valid && (is_scalar || product(dims) >= 0);
  return valid;
}

SparseTensorMeta::SparseTensorMeta(const DDim& dims) : dims(dims) {}

SparseTensorMeta::SparseTensorMeta(const DDim& dims, const DataLayout& layout)
    : dims(dims), layout(layout) {}

bool SparseTensorMeta::valid() const noexcept {
  bool valid{true};
  valid = valid && (layout != DataLayout::UNDEFINED);
  valid = valid && (product(dims) >= 0);
  return valid;
}

}  // namespace phi
