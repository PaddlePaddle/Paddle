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

#include "paddle/phi/backends/gpu/cuda/cudnn_helper.h"
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

inline cudnnDataType_t ToCudnnDataType(const phi::DataType& t) {
  cudnnDataType_t type = CUDNN_DATA_FLOAT;
  switch (t) {
    case phi::DataType::FLOAT16:
      type = CUDNN_DATA_HALF;
      break;
    case phi::DataType::FLOAT32:
      type = CUDNN_DATA_FLOAT;
      break;
    case phi::DataType::FLOAT64:
      type = CUDNN_DATA_DOUBLE;
      break;
#if CUDNN_VERSION_MIN(8, 1, 0)
    case phi::DataType::BFLOAT16:
      type = CUDNN_DATA_BFLOAT16;
      break;
#endif
    default:
      break;
  }
  return type;
}

class ActivationDescriptor {
 public:
  using T = cudnnActivationStruct;
  struct Deleter {
    void operator()(T* t) {
      if (t != nullptr) {
        PADDLE_ENFORCE_GPU_SUCCESS(
            phi::dynload::cudnnDestroyActivationDescriptor(t));
        t = nullptr;
      }
    }
  };
  ActivationDescriptor() {
    T* raw_ptr;
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnCreateActivationDescriptor(&raw_ptr));
    desc_.reset(raw_ptr);
  }
  template <typename T>
  void set(cudnnActivationMode_t mode, const T& coef) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnSetActivationDescriptor(
        desc_.get(), mode, CUDNN_NOT_PROPAGATE_NAN, static_cast<double>(coef)));
  }

  T* desc() { return desc_.get(); }
  T* desc() const { return desc_.get(); }

 private:
  std::unique_ptr<T, Deleter> desc_;
};

class TensorDescriptor {
 public:
  using T = cudnnTensorStruct;
  struct Deleter {
    void operator()(T* t) {
      if (t != nullptr) {
        PADDLE_ENFORCE_GPU_SUCCESS(
            phi::dynload::cudnnDestroyTensorDescriptor(t));
        t = nullptr;
      }
    }
  };
  TensorDescriptor() {
    T* raw_ptr;
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnCreateTensorDescriptor(&raw_ptr));
    desc_.reset(raw_ptr);
  }
  T* desc() { return desc_.get(); }
  T* desc() const { return desc_.get(); }
  void set(const phi::DenseTensor& tensor, const int groups = 1) {
    auto dims = phi::vectorize<int>(tensor.dims());
    std::vector<int> strides(dims.size());
    strides[dims.size() - 1] = 1;
    for (int i = dims.size() - 2; i >= 0; i--) {
      strides[i] = dims[i + 1] * strides[i + 1];
    }
    std::vector<int> dims_with_group(dims.begin(), dims.end());
    if (groups > 1) {
      dims_with_group[1] = dims_with_group[1] / groups;
    }
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnSetTensorNdDescriptor(
        desc_.get(),
        ToCudnnDataType(tensor.dtype()),
        dims_with_group.size(),
        dims_with_group.data(),
        strides.data()));
  }

  void set(const std::vector<int>& dims,
           const cudnnTensorFormat_t format,
           const cudnnDataType_t dtype) {
    std::vector<int> transformed_dims;
    if (format == CUDNN_TENSOR_NHWC) {
      transformed_dims = TransformDimOrder(dims);
    } else {
      transformed_dims = dims;
    }
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnSetTensorNdDescriptorEx(desc_.get(),
                                                   format,
                                                   dtype,
                                                   transformed_dims.size(),
                                                   transformed_dims.data()));
  }

  void set(const phi::DenseTensor& tensor, const cudnnTensorFormat_t format) {
    auto dims = phi::vectorize<int>(tensor.dims());
    auto dtype = ToCudnnDataType(tensor.dtype());
    set(dims, format, dtype);
  }

 private:
  std::unique_ptr<T, Deleter> desc_;
};

class FilterDescriptor {
 public:
  using T = cudnnFilterStruct;
  struct Deleter {
    void operator()(T* t) {
      if (t != nullptr) {
        PADDLE_ENFORCE_GPU_SUCCESS(
            phi::dynload::cudnnDestroyFilterDescriptor(t));
        t = nullptr;
      }
    }
  };
  FilterDescriptor() {
    T* raw_ptr;
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnCreateFilterDescriptor(&raw_ptr));
    desc_.reset(raw_ptr);
  }
  T* desc() { return desc_.get(); }
  T* desc() const { return desc_.get(); }

  void set(const std::vector<int>& dims,
           const cudnnTensorFormat_t format,
           const cudnnDataType_t dtype,
           const int groups = 1) {
    std::vector<int> transformed_dims;
    if (format == CUDNN_TENSOR_NHWC) {
      transformed_dims = TransformDimOrder(dims);
    } else {
      transformed_dims = dims;
    }
    if (groups > 1) {
      transformed_dims[1] = transformed_dims[1] / groups;
    }
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnSetFilterNdDescriptor(desc_.get(),
                                                 dtype,
                                                 format,
                                                 transformed_dims.size(),
                                                 transformed_dims.data()));
  }

  void set(const phi::DenseTensor& tensor,
           const cudnnTensorFormat_t format,
           const int groups = 1) {
    auto dims = phi::vectorize<int>(tensor.dims());
    auto dtype = ToCudnnDataType(tensor.dtype());
    set(dims, format, dtype, groups);
  }

 private:
  std::unique_ptr<T, Deleter> desc_;
};

class ConvolutionDescriptor {
 public:
  using T = cudnnConvolutionStruct;
  struct Deleter {
    void operator()(T* t) {
      if (t != nullptr) {
        PADDLE_ENFORCE_GPU_SUCCESS(
            phi::dynload::cudnnDestroyConvolutionDescriptor(t));
        t = nullptr;
      }
    }
  };
  ConvolutionDescriptor() {
    T* raw_ptr;
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnCreateConvolutionDescriptor(&raw_ptr));
    desc_.reset(raw_ptr);
  }
  T* desc() { return desc_.get(); }
  T* desc() const { return desc_.get(); }

  void set(cudnnDataType_t dtype,
           const std::vector<int>& pads,
           const std::vector<int>& strides,
           const std::vector<int>& dilations,
           bool allow_tf32,
           const int groups = 1) {
    allow_tf32_ = allow_tf32;
    cudnnDataType_t compute_type =
        (dtype == CUDNN_DATA_DOUBLE) ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT;
    T* desc = desc_.get();
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnSetConvolutionNdDescriptor(desc,
                                                      pads.size(),
                                                      pads.data(),
                                                      strides.data(),
                                                      dilations.data(),
                                                      CUDNN_CROSS_CORRELATION,
                                                      compute_type));
#if CUDNN_VERSION_MIN(7, 0, 1)
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnSetConvolutionGroupCount(desc, groups));
#if CUDA_VERSION >= 9000 && CUDNN_VERSION_MIN(7, 0, 1)
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnSetConvolutionMathType(desc, CUDNN_DEFAULT_MATH));
    if (dtype == CUDNN_DATA_HALF) {
      PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnSetConvolutionMathType(
          desc, CUDNN_TENSOR_OP_MATH));
#if CUDA_VERSION >= 11000
#if CUDNN_VERSION_MIN(8, 1, 0)
    } else if (dtype == CUDNN_DATA_BFLOAT16) {
      PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnSetConvolutionMathType(
          desc, CUDNN_TENSOR_OP_MATH));
#endif  // CUDNN_VERSION_MIN(8,1,0)
    } else if (dtype == CUDNN_DATA_FLOAT && !allow_tf32) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::cudnnSetConvolutionMathType(desc, CUDNN_FMA_MATH));
#endif  // CUDA_VERSION >= 11000
    }
#endif
#endif
  }

  bool allow_tf32_;

 private:
  std::unique_ptr<T, Deleter> desc_;
};

}  // namespace gpu
}  // namespace backends
}  // namespace phi
