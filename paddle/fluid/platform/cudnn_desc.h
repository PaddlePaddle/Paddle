// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include <iterator>
#include <memory>
#include <numeric>
#include <string>
#include <vector>
#include "paddle/fluid/platform/cudnn_helper.h"

namespace paddle {
namespace platform {
using framework::Tensor;

template <typename T>
cudnnDataType_t ToCudnnDataType(const T& t) {
  auto type = framework::ToDataType(t);
  return ToCudnnDataType(type);
}

template <>
cudnnDataType_t ToCudnnDataType(const framework::proto::VarType::Type& t) {
  cudnnDataType_t type = CUDNN_DATA_FLOAT;
  switch (t) {
    case framework::proto::VarType::FP16:
      type = CUDNN_DATA_HALF;
      break;
    case framework::proto::VarType::FP32:
      type = CUDNN_DATA_FLOAT;
      break;
    case framework::proto::VarType::FP64:
      type = CUDNN_DATA_DOUBLE;
      break;
    default:
      break;
  }
  return type;
}

class ActivationDescriptor {
 public:
  using T = cudnnActivationDescriptor_t;
  struct Deleter {
    void operator()(T* t) {
      if (t != nullptr) {
        PADDLE_ENFORCE(dynload::cudnnDestroyActivationDescriptor(*t));
      }
    }
  };
  ActivationDescriptor() {
    T raw_ptr;
    PADDLE_ENFORCE(dynload::cudnnCreateActivationDescriptor(&raw_ptr));
    desc_.reset(&raw_ptr);
  }
  template <typename T>
  void set(cudnnActivationMode_t mode, const T& coef) {
    double relu_ceiling = 0.0;
    ActivationMode activation_mode = StringToActivationMode(act);
    cudnnActivationMode_t mode;
    switch (activation_mode) {
      case ActivationMode::kRelu6:
        relu_ceiling = 6.0;
        mode = CUDNN_ACTIVATION_CLIPPED_RELU;
        break;
      case ActivationMode::kReluX:
        relu_ceiling = value_max;
        mode = CUDNN_ACTIVATION_CLIPPED_RELU;
        break;
      case ActivationMode::kRelu:
        mode = CUDNN_ACTIVATION_RELU;
        break;
      case ActivationMode::kSigmoid:
        mode = CUDNN_ACTIVATION_SIGMOID;
        break;
      case ActivationMode::kTanh:
        mode = CUDNN_ACTIVATION_TANH;
        break;
      default:
        PADDLE_THROW("unrecognized activation mode: %d .",
                     static_cast<int>(activation_mode));
    }
    CUDNN_ENFORCE(dynload::cudnnSetActivationDescriptor(
        desc_, mode, CUDNN_NOT_PROPAGATE_NAN, relu_ceiling));
  }

  T desc() { return *desc_; }
  T desc() const { return *desc_; }

 private:
  std::unique_ptr<T, Deleter> desc_;
};

struct CudnnActivationFunctor {
  void operator()() {}
};

struct CudnnActivationGradFunctor {};

class TensorDescriptor {
 public:
  using T = cudnnTensorDescriptor_t;
  struct Deleter {
    void operator()(T* t) {
      if (t != nullptr) {
        PADDLE_ENFORCE(dynload::cudnnDestroyTensorDescriptor(*t));
      }
    }
  };
  TensorDescriptor() {
    T raw_ptr;
    PADDLE_ENFORCE(dynload::cudnnCreateTensorDescriptor(&raw_ptr));
    desc_.reset(&raw_ptr);
  }

  T desc() { return *desc_; }
  T desc() const { return *desc_; }
  void set(const Tensor& tensor, const int groups = 1) {
    auto dims = framework::vectorize2int(tensor.dims());
    std::vector<int> strides(dims.size());
    strides[dims.size() - 1] = 1;
    for (int i = dims.size() - 2; i >= 0; i--) {
      strides[i] = dims[i + 1] * strides[i + 1];
    }
    // Update tensor descriptor dims setting if groups > 1
    // NOTE: Assume using NCHW or NCDHW order
    std::vector<int> dims_with_group(dims.begin(), dims.end());  // copy
    if (groups > 1) {
      dims_with_group[1] = dims_with_group[1] / groups;
    }
    PADDLE_ENFORCE(dynload::cudnnSetTensorNdDescriptor(
        *desc_, ToCudnnDataType(tensor.type()), dims_with_group.size(),
        dims_with_group.data(), strides.data()));
  }

 private:
  std::unique_ptr<T> desc_;
};

}  // namespace platform
}  // namespace paddle
