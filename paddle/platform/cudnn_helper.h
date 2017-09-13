/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <vector>
#include "paddle/platform/dynload/cudnn.h"
#include "paddle/platform/enforce.h"
#include "paddle/platform/macros.h"

namespace paddle {
namespace platform {

enum class DataLayout {
  kNHWC,
  kNCHW,
  kNCHW_VECT_C,
};

enum class PoolingMode {
  kMaximum,
  kAverage,
};

template <typename T>
class CudnnDataType;

template <>
class CudnnDataType<float> {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_FLOAT;
};

template <>
class CudnnDataType<double> {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_DOUBLE;
};

inline cudnnTensorFormat_t GetCudnnTensorFormat(const DataLayout& order) {
  switch (order) {
    case DataLayout::kNHWC:
      return CUDNN_TENSOR_NHWC;
    case DataLayout::kNCHW:
      return CUDNN_TENSOR_NCHW;
    default:
      PADDLE_THROW("Unknown cudnn equivalent for order");
  }
  return CUDNN_TENSOR_NCHW;
}

class ScopedTensorDescriptor {
 public:
  ScopedTensorDescriptor() {
    PADDLE_ENFORCE(dynload::cudnnCreateTensorDescriptor(&desc_));
  }
  ~ScopedTensorDescriptor() {
    PADDLE_ENFORCE(dynload::cudnnDestroyTensorDescriptor(desc_));
  }

  inline cudnnTensorDescriptor_t descriptor(const cudnnTensorFormat_t format,
                                            const cudnnDataType_t type,
                                            const std::vector<int>& dims) {
    // the format is not used now, but it maybe useful feature
    std::vector<int> strides(dims.size());
    strides[dims.size() - 1] = 1;
    for (int i = dims.size() - 2; i >= 0; i--) {
      strides[i] = dims[i + 1] * strides[i + 1];
    }
    PADDLE_ENFORCE(dynload::cudnnSetTensorNdDescriptor(
        desc_, type, dims.size(), dims.data(), strides.data()));
    return desc_;
  }

  template <typename T>
  inline cudnnTensorDescriptor_t descriptor(const DataLayout& order,
                                            const std::vector<int>& dims) {
    return descriptor(GetCudnnTensorFormat(order), CudnnDataType<T>::type,
                      dims);
  }

 private:
  cudnnTensorDescriptor_t desc_;
  DISABLE_COPY_AND_ASSIGN(ScopedTensorDescriptor);
};

class ScopedFilterDescriptor {
 public:
  ScopedFilterDescriptor() {
    PADDLE_ENFORCE(dynload::cudnnCreateFilterDescriptor(&desc_));
  }
  ~ScopedFilterDescriptor() {
    PADDLE_ENFORCE(dynload::cudnnDestroyFilterDescriptor(desc_));
  }

  inline cudnnFilterDescriptor_t descriptor(const cudnnTensorFormat_t format,
                                            const cudnnDataType_t type,
                                            const std::vector<int>& kernel) {
    // filter layout: output input spatial_dim_y spatial_dim_x
    PADDLE_ENFORCE(dynload::cudnnSetFilterNdDescriptor(
        desc_, type, format, kernel.size(), kernel.data()));
    return desc_;
  }

  template <typename T>
  inline cudnnFilterDescriptor_t descriptor(const DataLayout& order,
                                            const std::vector<int>& kernel) {
    return descriptor(GetCudnnTensorFormat(order), CudnnDataType<T>::type,
                      kernel);
  }

 private:
  cudnnFilterDescriptor_t desc_;
  DISABLE_COPY_AND_ASSIGN(ScopedFilterDescriptor);
};

class ScopedConvolutionDescriptor {
 public:
  ScopedConvolutionDescriptor() {
    PADDLE_ENFORCE(dynload::cudnnCreateConvolutionDescriptor(&desc_));
  }
  ~ScopedConvolutionDescriptor() {
    PADDLE_ENFORCE(dynload::cudnnDestroyConvolutionDescriptor(desc_));
  }

  inline cudnnConvolutionDescriptor_t descriptor(
      cudnnDataType_t type, const std::vector<int>& pads,
      const std::vector<int>& strides, const std::vector<int>& dilations) {
    PADDLE_ENFORCE_EQ(pads.size(), strides.size());
    PADDLE_ENFORCE_EQ(pads.size(), dilations.size());

#if CUDNN_VERSION < 6000
    // cudnn v5 does not support dilation conv, the argument is called upscale
    // instead of dilations and it is must be one.
    for (size_t i = 0; i < dilations.size(); ++i) {
      PADDLE_ENFORCE_EQ(
          dilations[i], 1,
          "Dilations conv is not supported in this cuDNN version");
    }
#endif

    PADDLE_ENFORCE(dynload::cudnnSetConvolutionNdDescriptor(
        desc_, pads.size(), pads.data(), strides.data(), dilations.data(),
        CUDNN_CROSS_CORRELATION, type));
    return desc_;
  }

  template <typename T>
  inline cudnnConvolutionDescriptor_t descriptor(
      const std::vector<int>& pads, const std::vector<int>& strides,
      const std::vector<int>& dilations) {
    return descriptor(CudnnDataType<T>::type, pads, strides, dilations);
  }

 private:
  cudnnConvolutionDescriptor_t desc_;
  DISABLE_COPY_AND_ASSIGN(ScopedConvolutionDescriptor);
};

class ScopedPoolingDescriptor {
 public:
  ScopedPoolingDescriptor() {
    PADDLE_ENFORCE(dynload::cudnnCreatePoolingDescriptor(&desc_));
  }
  ~ScopedPoolingDescriptor() {
    PADDLE_ENFORCE(dynload::cudnnDestroyPoolingDescriptor(desc_));
  }

  inline cudnnPoolingDescriptor_t descriptor(const PoolingMode& mode,
                                             const std::vector<int>& kernel,
                                             const std::vector<int>& pads,
                                             const std::vector<int>& strides) {
    PADDLE_ENFORCE_EQ(kernel.size(), pads.size());
    PADDLE_ENFORCE_EQ(kernel.size(), strides.size());
    PADDLE_ENFORCE(dynload::cudnnSetPoolingNdDescriptor(
        desc_, (mode == PoolingMode::kMaximum
                    ? CUDNN_POOLING_MAX
                    : CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING),
        CUDNN_PROPAGATE_NAN,  // Always propagate nans.
        kernel.size(), kernel.data(), pads.data(), strides.data()));
    return desc_;
  }

 private:
  cudnnPoolingDescriptor_t desc_;
  DISABLE_COPY_AND_ASSIGN(ScopedPoolingDescriptor);
};

}  // namespace platform
}  // namespace paddle
