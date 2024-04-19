// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <algorithm>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <string>
#include <vector>
#include "paddle/phi/backends/gpu/musa/mudnn_helper.h"
#include "paddle/phi/core/utils/data_type.h"

namespace phi {
namespace backends {
namespace gpu {

template <typename T>
inline std::vector<T> TransformDimOrder(const std::vector<T>& dims) {
  std::vector<T> transformed_dims(dims.begin(), dims.end());
  if (dims.size() < 4) {
    return transformed_dims;
  }
  T H, W, D, C;
  if (dims.size() == 4) {
    H = dims[1];
    W = dims[2];
    C = dims[3];
    transformed_dims[1] = C;
    transformed_dims[2] = H;
    transformed_dims[3] = W;
  } else {
    D = dims[1];
    H = dims[2];
    W = dims[3];
    C = dims[4];
    transformed_dims[1] = C;
    transformed_dims[2] = D;
    transformed_dims[3] = H;
    transformed_dims[4] = W;
  }
  return transformed_dims;
}

inline dynload::Tensor::Type ToCudnnDataType(const phi::DataType& t) {
  dynload::Tensor::Type type = dynload::Tensor::Type::FLOAT;
  switch (t) {
    case phi::DataType::FLOAT16:
      type = dynload::Tensor::Type::HALF;
      break;
    case phi::DataType::FLOAT32:
      type = dynload::Tensor::Type::FLOAT;
      break;
    case phi::DataType::FLOAT64:
      type = dynload::Tensor::Type::DOUBLE;
      break;
    default:
      PD_THROW("Don't support this data type ", t);
  }
  return type;
}

class TensorDescriptor {
 public:
  using T = dynload::Tensor;
  TensorDescriptor() : desc_(std::make_unique<T>()) {}
  T* desc() { return desc_.get(); }
  T* desc() const { return desc_.get(); }
  void set(const phi::DenseTensor& tensor, const int groups = 1) {
    auto dims = phi::vectorize<int64_t>(tensor.dims());
    std::vector<int64_t> strides(dims.size());
    strides[dims.size() - 1] = 1;
    for (int i = dims.size() - 2; i >= 0; i--) {
      strides[i] = dims[i + 1] * strides[i + 1];
    }
    desc_->SetType(ToCudnnDataType(tensor.dtype()));
    desc_->SetNdInfo(static_cast<int>(dims.size()), dims.data(), strides.data());
    desc_->SetAddr(tensor.data());
  }

  template <typename Type>
  void set(const phi::DenseTensor& tensor, const Type* data) {
    auto dims = phi::vectorize<int64_t>(tensor.dims());
    std::vector<int64_t> strides(dims.size());
    strides[dims.size() - 1] = 1;
    for (int i = dims.size() - 2; i >= 0; i--) {
      strides[i] = dims[i + 1] * strides[i + 1];
    }
    desc_->SetType(ToCudnnDataType(tensor.dtype()));
    desc_->SetNdInfo(static_cast<int>(dims.size()), dims.data(), strides.data());
    desc_->SetAddr(data);
  }

  void set(const std::vector<int>& dims,
           const dynload::Tensor::Format format,
           const dynload::Tensor::Type dtype) {
    std::vector<int64_t> transformed_dims;
    std::vector<int64_t> dims_64(dims.begin(), dims.end());
    if (format == dynload::Tensor::Format::NHWC) {
      transformed_dims = TransformDimOrder(dims_64);
    } else {
      transformed_dims = dims_64;
    }
    desc_->SetFormat(format);
    desc_->SetType(dtype);
    desc_->SetNdInfo(static_cast<int>(transformed_dims.size()), transformed_dims.data());
  }

  void set(const phi::DenseTensor& tensor,
           const dynload::Tensor::Format format) {
    auto dims = phi::vectorize<int>(tensor.dims());
    auto dtype = ToCudnnDataType(tensor.dtype());
    set(dims, format, dtype);
    desc_->SetAddr(tensor.data());
  }

 private:
  std::unique_ptr<T> desc_;
};

class FilterDescriptor {
 public:
  using T = phi::dynload::Tensor;
  FilterDescriptor() : desc_(std::make_unique<T>()) {}
  T* desc() { return desc_.get(); }
  T* desc() const { return desc_.get(); }

  void set(const std::vector<int>& dims,
           const dynload::Tensor::Format format,
           const dynload::Tensor::Type dtype,
           const int groups = 1) {
    std::vector<int64_t> transformed_dims;
    std::vector<int64_t> dims_64(dims.begin(), dims.end());
    if (format == dynload::Tensor::Format::NHWC) {
      transformed_dims = TransformDimOrder(dims_64);
    } else {
      transformed_dims = dims_64;
    }
    if (groups > 1) {
      transformed_dims[1] = transformed_dims[1] / groups;
    }
    desc_->SetFormat(format);
    desc_->SetType(dtype);
    desc_->SetNdInfo(static_cast<int>(transformed_dims.size()), transformed_dims.data());
  }

  void set(const phi::DenseTensor& tensor,
           const dynload::Tensor::Format format,
           const int groups = 1) {
    auto dims = phi::vectorize<int>(tensor.dims());
    auto dtype = ToCudnnDataType(tensor.dtype());
    set(dims, format, dtype, groups);
    desc_->SetAddr(tensor.data());
  }

 private:
  std::unique_ptr<T> desc_;
};

class ConvolutionDescriptor {
 public:
  using T = dynload::Convolution;
  ConvolutionDescriptor() : desc_(std::make_unique<T>()) {}
  T* desc() { return desc_.get(); }
  T* desc() const { return desc_.get(); }

  void set(dynload::Tensor::Type dtype,
           const std::vector<int>& pads,
           const std::vector<int>& strides,
           const std::vector<int>& dilations,
           bool allow_tf32,
           const int groups = 1) {
    allow_tf32_ = allow_tf32;
    desc_->SetNdInfo(
        pads.size(), pads.data(), strides.data(), dilations.data());
    desc_->SetComputeMode(dynload::Convolution::ComputeMode::TENSOR);
    desc_->SetGroups(groups);
  }

  bool allow_tf32_;

 private:
  std::unique_ptr<T> desc_;
};

}  // namespace gpu
}  // namespace backends
}  // namespace phi
