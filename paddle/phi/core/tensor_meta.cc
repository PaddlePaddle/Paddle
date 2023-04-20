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

DDim calc_strides(const DDim& dims, DataLayout layout) {
  if (product(dims) <= 0) {
    return dims;
  }

  DDim strides(dims);

  if (dims.size() == 4 && layout == DataLayout::NHWC) {
    strides[1] = 1;
    strides[3] = dims[1];
    strides[2] = strides[3] * dims[3];
    strides[0] = strides[2] * dims[2];
  } else if (dims.size() == 5 && layout == DataLayout::NDHWC) {
    strides[1] = 1;
    strides[4] = dims[1];
    strides[3] = strides[4] * dims[4];
    strides[2] = strides[3] * dims[3];
    strides[0] = strides[2] * dims[2];
  } else {
    strides = ::phi::stride(dims);
  }

  return strides;
}

bool is_contiguous(const DDim& dims, const DDim& strides, DataLayout layout) {
  // TODO(liudongxue01): optimize this
  return strides == calc_strides(dims, layout);
}

void DenseTensorMeta::sync_strides() {
  this->strides = calc_strides(dims, layout);
}

void DenseTensorMeta::update(DDim dims) {
  this->dims = dims;
  sync_strides();
}

void DenseTensorMeta::update(DataLayout layout) {
  this->layout = layout;
  sync_strides();
}

void DenseTensorMeta::update(DDim dims, DataLayout layout) {
  this->dims = dims;
  this->layout = layout;
  sync_strides();
}

void DenseTensorMeta::update(DDim dims, DDim strides, DataLayout layout) {
  this->dims = dims;
  this->strides = strides;
  this->layout = layout;
  if (strides.size() != dims.size()) {
    sync_strides();
  }
}

DenseTensorMeta::DenseTensorMeta() { use_gpudnn = true; }

DenseTensorMeta::DenseTensorMeta(DataType dtype, const DDim& dims)
    : dtype(dtype) {
  update(dims);
  use_gpudnn = true;
}

DenseTensorMeta::DenseTensorMeta(DataType dtype,
                                 const DDim& dims,
                                 const DDim& strides)
    : dims(dims), dtype(dtype), strides(strides) {
  use_gpudnn = true;
}

DenseTensorMeta::DenseTensorMeta(DataType dtype,
                                 const DDim& dims,
                                 DataLayout layout,
                                 size_t offset)
    : dtype(dtype), offset(offset) {
  update(dims, layout);
  use_gpudnn = true;
}

DenseTensorMeta::DenseTensorMeta(DataType dtype,
                                 const DDim& dims,
                                 DataLayout layout,
                                 const LoD& lod,
                                 size_t offset)
    : dtype(dtype), lod(lod), offset(offset) {
  update(dims, layout);
  use_gpudnn = true;
}

DenseTensorMeta::DenseTensorMeta(const DenseTensorMeta& other) {
  is_scalar = other.is_scalar;
  use_gpudnn = other.use_gpudnn;
  dtype = other.dtype;
  lod = other.lod;
  offset = other.offset;
  update(other.dims, other.strides, other.layout);
}

DenseTensorMeta& DenseTensorMeta::operator=(const DenseTensorMeta& other) {
  is_scalar = other.is_scalar;
  use_gpudnn = other.use_gpudnn;
  dtype = other.dtype;
  lod = other.lod;
  offset = other.offset;
  update(other.dims, other.strides, other.layout);
  return *this;
}

DenseTensorMeta& DenseTensorMeta::operator=(DenseTensorMeta&& other) {
  is_scalar = other.is_scalar;
  use_gpudnn = other.use_gpudnn;
  dtype = other.dtype;
  lod = std::move(other.lod);
  offset = other.offset;
  update(other.dims, other.strides, other.layout);
  return *this;
}

bool DenseTensorMeta::valid() const noexcept {
  bool valid{true};
  valid = valid && (dtype != DataType::UNDEFINED);
  valid = valid && (layout != DataLayout::UNDEFINED);
  valid = valid && (is_scalar || product(dims) >= 0);
  return valid;
}

bool DenseTensorMeta::is_contiguous() const noexcept {
  return phi::is_contiguous(dims, strides, layout);
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
