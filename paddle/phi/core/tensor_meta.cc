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

DDim DenseTensorMeta::calc_strides(const DDim& dims, DataLayout layout) {
  DDim strides(dims);
  if (layout == DataLayout::NCHW) {
    strides[dims.size() - 1] = 1;
    for (int i = dims.size() - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * dims[i + 1];
    }
  } else if (layout == DataLayout::NHWC) {
    strides[1] = 1;
    strides[3] = dims[1];
    strides[2] = strides[3] * dims[3];
    strides[0] = strides[2] * dims[2];
  } else if (layout == DataLayout::NCDHW) {
    // TODO(wanghuancoder)
  } else if (layout == DataLayout::NDHWC) {
    // TODO(wanghuancoder)
  }
  return strides;
}

DenseTensorMeta::DenseTensorMeta() { use_gpudnn = true; }

DenseTensorMeta::DenseTensorMeta(DataType dtype, const DDim& dims)
    : dims(dims), dtype(dtype) {
  strides = calc_strides(dims);
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
    : dims(dims), dtype(dtype), layout(layout), offset(offset) {
  strides = calc_strides(dims, layout);
  use_gpudnn = true;
}

DenseTensorMeta::DenseTensorMeta(DataType dtype,
                                 const DDim& dims,
                                 DataLayout layout,
                                 const LoD& lod,
                                 size_t offset)
    : dims(dims), dtype(dtype), layout(layout), lod(lod), offset(offset) {
  strides = calc_strides(dims, layout);
  use_gpudnn = true;
}

DenseTensorMeta::DenseTensorMeta(const DenseTensorMeta& other) {
  is_scalar = other.is_scalar;
  use_gpudnn = other.use_gpudnn;
  dims = other.dims;
  dtype = other.dtype;
  layout = other.layout;
  lod = other.lod;
  offset = other.offset;
  strides = other.strides;
  if (strides.size() == 1 && strides[0] == 0) {
    strides == calc_strides(dims, layout);
  }
}

DenseTensorMeta& DenseTensorMeta::operator=(const DenseTensorMeta& other) {
  is_scalar = other.is_scalar;
  use_gpudnn = other.use_gpudnn;
  dims = other.dims;
  dtype = other.dtype;
  layout = other.layout;
  lod = other.lod;
  offset = other.offset;
  strides = other.strides;
  if (strides.size() == 1 && strides[0] == 0) {
    strides == calc_strides(dims, layout);
  }
  return *this;
}

DenseTensorMeta& DenseTensorMeta::operator=(DenseTensorMeta&& other) {
  is_scalar = other.is_scalar;
  use_gpudnn = other.use_gpudnn;
  dims = std::move(other.dims);
  dtype = other.dtype;
  layout = other.layout;
  lod = std::move(other.lod);
  offset = other.offset;
  strides = std::move(other.strides);
  if (strides.size() == 1 && strides[0] == 0) {
    strides == calc_strides(dims, layout);
  }
  return *this;
}

bool DenseTensorMeta::valid() const noexcept {
  bool valid{true};
  valid = valid && (dtype != DataType::UNDEFINED);
  valid = valid && (layout != DataLayout::UNDEFINED);
  valid = valid && (is_scalar || product(dims) >= 0);
  return valid;
}

bool DenseTensorMeta::is_contiguous(DataLayout exp_layout) const noexcept {
  return strides == calc_strides(dims, exp_layout);
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
