// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/miopen_helper.h"

namespace paddle {
namespace framework {
class Tensor;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace platform {
using framework::Tensor;

template <typename T>
inline miopenDataType_t ToMIOpenDataType(const T& t) {
  auto type = framework::ToDataType(t);
  return ToMIOpenDataType(type);
}

inline std::vector<int> TransformDimOrder(const std::vector<int>& dims) {
  std::vector<int> transformed_dims(dims.begin(), dims.end());
  int H, W, D, C;
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

template <>
inline miopenDataType_t ToMIOpenDataType(
    const framework::proto::VarType::Type& t) {
  miopenDataType_t type = miopenFloat;
  switch (t) {
    case framework::proto::VarType::FP16:
      type = miopenHalf;
      break;
    case framework::proto::VarType::FP32:
      type = miopenFloat;
      break;
    default:
      break;
  }
  return type;
}

class ActivationDescriptor {
 public:
  ActivationDescriptor() {
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::miopenCreateActivationDescriptor(&desc_));
  }
  ~ActivationDescriptor() {
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::miopenDestroyActivationDescriptor(desc_));
  }
  template <typename T>
  void set(miopenActivationMode_t mode, const T& coef) {
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::miopenSetActivationDescriptor(
        desc_, mode, static_cast<double>(coef), 0.0, 0.0));
  }

  miopenActivationDescriptor_t desc() { return desc_; }
  miopenActivationDescriptor_t desc() const { return desc_; }

 private:
  miopenActivationDescriptor_t desc_;
};

class TensorDescriptor {
 public:
  TensorDescriptor() {
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::miopenCreateTensorDescriptor(&desc_));
  }
  ~TensorDescriptor() {
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::miopenDestroyTensorDescriptor(desc_));
  }
  miopenTensorDescriptor_t desc() { return desc_; }
  miopenTensorDescriptor_t desc() const { return desc_; }

  void set(const Tensor& tensor, const int groups = 1) {
    auto dims = framework::vectorize<int>(tensor.dims());
    std::vector<int> strides(dims.size());
    strides[dims.size() - 1] = 1;
    for (int i = dims.size() - 2; i >= 0; i--) {
      strides[i] = dims[i + 1] * strides[i + 1];
    }
    std::vector<int> dims_with_group(dims.begin(), dims.end());
    if (groups > 1) {
      dims_with_group[1] = dims_with_group[1] / groups;
    }
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::miopenSetTensorDescriptor(
        desc_, ToMIOpenDataType(tensor.type()),
        static_cast<int>(dims_with_group.size()),
        const_cast<int*>(dims_with_group.data()),
        const_cast<int*>(strides.data())));
  }

  void set(const Tensor& tensor, const miopenTensorFormat_t format) {
    const int groups = 1;
    auto dims = framework::vectorize<int>(tensor.dims());
    std::vector<int> strides(dims.size());
    strides[dims.size() - 1] = 1;
    for (int i = dims.size() - 2; i >= 0; i--) {
      strides[i] = dims[i + 1] * strides[i + 1];
    }
    std::vector<int> dims_with_group(dims.begin(), dims.end());
    if (groups > 1) {
      dims_with_group[1] = dims_with_group[1] / groups;
    }
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::miopenSetTensorDescriptor(
        desc_, ToMIOpenDataType(tensor.type()),
        static_cast<int>(dims_with_group.size()),
        const_cast<int*>(dims_with_group.data()),
        const_cast<int*>(strides.data())));
  }

 private:
  miopenTensorDescriptor_t desc_;
};

class FilterDescriptor {
 public:
  FilterDescriptor() {
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::miopenCreateTensorDescriptor(&desc_));
  }
  ~FilterDescriptor() {
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::miopenDestroyTensorDescriptor(desc_));
  }
  miopenTensorDescriptor_t desc() { return desc_; }
  miopenTensorDescriptor_t desc() const { return desc_; }

  void set(const Tensor& tensor, const miopenTensorFormat_t format,
           const int groups = 1) {
    auto dims = framework::vectorize<int>(tensor.dims());
    std::vector<int> transformed_dims;
    PADDLE_ENFORCE_EQ(format, MIOPEN_TENSOR_NCHW,
                      platform::errors::InvalidArgument(
                          "format should ONLY be NCHW in MIOPEN."));
    transformed_dims = dims;
    if (groups > 1) {
      transformed_dims[1] = transformed_dims[1] / groups;
    }
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::miopenSetTensorDescriptor(
        desc_, ToMIOpenDataType(tensor.type()),
        static_cast<int>(transformed_dims.size()),
        const_cast<int*>(transformed_dims.data()), nullptr));
  }

 private:
  miopenTensorDescriptor_t desc_;
};

class ConvolutionDescriptor {
 public:
  ConvolutionDescriptor() {
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::miopenCreateConvolutionDescriptor(&desc_));
  }
  ~ConvolutionDescriptor() {
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::miopenDestroyConvolutionDescriptor(desc_));
  }
  miopenConvolutionDescriptor_t desc() { return desc_; }
  miopenConvolutionDescriptor_t desc() const { return desc_; }

  void set(miopenDataType_t dtype, const std::vector<int>& pads,
           const std::vector<int>& strides, const std::vector<int>& dilations,
           bool allow_tf32, const int groups = 1) {
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::miopenInitConvolutionNdDescriptor(
        desc_, static_cast<int>(pads.size()), const_cast<int*>(pads.data()),
        const_cast<int*>(strides.data()), const_cast<int*>(dilations.data()),
        miopenConvolution));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::miopenSetConvolutionGroupCount(desc_, groups));
  }

 private:
  miopenConvolutionDescriptor_t desc_;
};

}  // namespace platform
}  // namespace paddle
