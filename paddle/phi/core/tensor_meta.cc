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

DDim DenseTensorMeta::calc_stride(const DDim& dims, DataLayout layout) {
  if (product(dims) <= 0) {
    return dims;
  }

  DDim stride(dims);

  if (dims.size() == 4 && layout == DataLayout::NHWC) {
    stride[1] = 1;
    stride[3] = dims[1];
    stride[2] = stride[3] * dims[3];
    stride[0] = stride[2] * dims[2];
  } else if (dims.size() == 5 && layout == DataLayout::NDHWC) {
    stride[1] = 1;
    stride[4] = dims[1];
    stride[3] = stride[4] * dims[4];
    stride[2] = stride[3] * dims[3];
    stride[0] = stride[2] * dims[2];
  } else {
    stride[dims.size() - 1] = 1;
    for (int i = dims.size() - 2; i >= 0; --i) {
      stride[i] = stride[i + 1] * dims[i + 1];
    }
  }

  return stride;
}

DenseTensorMeta::DenseTensorMeta() { use_gpudnn = true; }

DenseTensorMeta::DenseTensorMeta(DataType dtype, const DDim& dims)
    : dims(dims), dtype(dtype) {
  stride = calc_stride(dims);
  use_gpudnn = true;
}

DenseTensorMeta::DenseTensorMeta(DataType dtype,
                                 const DDim& dims,
                                 const DDim& stride)
    : dims(dims), dtype(dtype), stride(stride) {
  use_gpudnn = true;
}

DenseTensorMeta::DenseTensorMeta(DataType dtype,
                                 const DDim& dims,
                                 DataLayout layout,
                                 size_t offset)
    : dims(dims), dtype(dtype), layout(layout), offset(offset) {
  stride = calc_stride(dims, layout);
  use_gpudnn = true;
}

DenseTensorMeta::DenseTensorMeta(DataType dtype,
                                 const DDim& dims,
                                 DataLayout layout,
                                 const LoD& lod,
                                 size_t offset)
    : dims(dims), dtype(dtype), layout(layout), lod(lod), offset(offset) {
  stride = calc_stride(dims, layout);
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
  if (product(other.stride) <= 0) {
    stride == calc_stride(dims, layout);
  } else {
    stride = other.stride;
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
  if (product(other.stride) <= 0) {
    stride == calc_stride(dims, layout);
  } else {
    stride = other.stride;
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
  if (product(other.stride) <= 0) {
    stride == calc_stride(dims, layout);
  } else {
    stride = std::move(other.stride);
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
  return stride == calc_stride(dims, exp_layout);
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
