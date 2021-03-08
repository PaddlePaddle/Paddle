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
inline miopenDataType_t ToCudnnDataType(const T& t) {
  auto type = framework::ToDataType(t);
  return ToCudnnDataType(type);
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
inline miopenDataType_t ToCudnnDataType(
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
  using T = miopenActivationDescriptor;
  struct Deleter {
    void operator()(T* t) {
      if (t != nullptr) {
        PADDLE_ENFORCE_CUDA_SUCCESS(
            dynload::miopenDestroyActivationDescriptor(t));
        t = nullptr;
      }
    }
  };
  ActivationDescriptor() {
    T* raw_ptr;
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::miopenCreateActivationDescriptor(&raw_ptr));
    desc_.reset(raw_ptr);
  }
  template <typename T>
  void set(miopenActivationMode_t mode, const T& coef) {
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::miopenSetActivationDescriptor(
        desc_.get(), mode, static_cast<double>(coef), 0.0, 0.0));
  }

  T* desc() { return desc_.get(); }
  T* desc() const { return desc_.get(); }

 private:
  std::unique_ptr<T, Deleter> desc_;
};

class TensorDescriptor {
 public:
  using T = miopenTensorDescriptor;
  struct Deleter {
    void operator()(T* t) {
      if (t != nullptr) {
        PADDLE_ENFORCE_CUDA_SUCCESS(dynload::miopenDestroyTensorDescriptor(t));
        t = nullptr;
      }
    }
  };
  TensorDescriptor() {
    T* raw_ptr;
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::miopenCreateTensorDescriptor(&raw_ptr));
    desc_.reset(raw_ptr);
  }
  T* desc() { return desc_.get(); }
  T* desc() const { return desc_.get(); }

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
        (miopenTensorDescriptor_t)(desc_.get()), ToCudnnDataType(tensor.type()),
        static_cast<int>(dims_with_group.size()),
        const_cast<int*>(dims_with_group.data()),
        const_cast<int*>(strides.data())));
  }

  void set(const Tensor& tensor, const miopenTensorFormat_t format) {
    const int groups = 1;
    PADDLE_ENFORCE_EQ(format, MIOPEN_TENSOR_NCHW,
                      platform::errors::InvalidArgument(
                          "format should ONLY be NCHW in MIOPEN."));
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
        (miopenTensorDescriptor_t)(desc_.get()), ToCudnnDataType(tensor.type()),
        static_cast<int>(dims_with_group.size()),
        const_cast<int*>(dims_with_group.data()),
        const_cast<int*>(strides.data())));
  }

 private:
  std::unique_ptr<T, Deleter> desc_;
};

class FilterDescriptor {
 public:
  using T = miopenTensorDescriptor;
  struct Deleter {
    void operator()(T* t) {
      if (t != nullptr) {
        PADDLE_ENFORCE_CUDA_SUCCESS(dynload::miopenDestroyTensorDescriptor(t));
        t = nullptr;
      }
    }
  };
  FilterDescriptor() {
    T* raw_ptr;
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::miopenCreateTensorDescriptor(&raw_ptr));
    desc_.reset(raw_ptr);
  }
  T* desc() { return desc_.get(); }
  T* desc() const { return desc_.get(); }

  void set(const Tensor& tensor, const miopenTensorFormat_t format,
           const int groups = 1) {
    auto dims = framework::vectorize<int>(tensor.dims());
    std::vector<int> transformed_dims;
    PADDLE_ENFORCE_EQ(format, MIOPEN_TENSOR_NCHW,
                      platform::errors::InvalidArgument(
                          "format should ONLY be NCHW in MIOPEN."));
    transformed_dims = dims;
    // if (groups > 1) {
    //   transformed_dims[1] = transformed_dims[1] / groups;
    // }
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::miopenSet4dTensorDescriptor(
        (miopenTensorDescriptor_t)desc_.get(), ToCudnnDataType(tensor.type()),
        transformed_dims[0], transformed_dims[1], transformed_dims[2],
        transformed_dims[3]));
  }

 private:
  std::unique_ptr<T, Deleter> desc_;
};

class ConvolutionDescriptor {
 public:
  using T = miopenConvolutionDescriptor;
  struct Deleter {
    void operator()(T* t) {
      if (t != nullptr) {
        PADDLE_ENFORCE_CUDA_SUCCESS(
            dynload::miopenDestroyConvolutionDescriptor(t));
        t = nullptr;
      }
    }
  };
  ConvolutionDescriptor() {
    T* raw_ptr;
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::miopenCreateConvolutionDescriptor(&raw_ptr));
    desc_.reset(raw_ptr);
  }
  T* desc() { return desc_.get(); }
  T* desc() const { return desc_.get(); }

  void set(miopenDataType_t dtype, const std::vector<int>& pads,
           const std::vector<int>& strides, const std::vector<int>& dilations,
           bool allow_tf32, const int groups = 1) {
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::miopenInitConvolutionNdDescriptor(
        (miopenConvolutionDescriptor_t)desc_.get(),
        static_cast<int>(pads.size()), const_cast<int*>(pads.data()),
        const_cast<int*>(strides.data()), const_cast<int*>(dilations.data()),
        miopenConvolution));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::miopenSetConvolutionGroupCount(
            (miopenConvolutionDescriptor_t)desc_.get(), groups));
  }

 private:
  std::unique_ptr<T, Deleter> desc_;
};

}  // namespace platform
}  // namespace paddle
