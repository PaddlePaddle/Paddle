/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#include <string>
#include <vector>

#include "gflags/gflags.h"
#include "paddle/phi/backends/dynload/mudnn.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/errors.h"
#include "paddle/phi/core/macros.h"

#define CUDNN_BN_MIN_EPSILON 1e-05

DECLARE_bool(cudnn_deterministic);

namespace phi {
namespace backends {
namespace gpu {

#define CUDNN_VERSION_MIN(major, minor, patch) \
  (CUDNN_VERSION >= ((major)*1000 + (minor)*100 + (patch)))

enum class DataLayout {  // Not use
  kNHWC,
  kNCHW,
  kNCDHW,
  kNDHWC,  // add, liyamei
  kNCHW_VECT_C,
};

enum class PoolingMode {
  kMaximum,
  kMaximumDeterministic,
  kAverageExclusive,
  kAverageInclusive,
};

// enum class ActivationMode {
//   kNone,  // activation identity
//   kSigmoid,
//   kRelu,
//   kRelu6,
//   kReluX,
//   kTanh,
//   kBandPass,
// };

inline dynload::Pooling::Mode GetPoolingMode(const PoolingMode& mode) {
  switch (mode) {
    // case PoolingMode::kMaximumDeterministic:
    //   return CUDNN_POOLING_MAX_DETERMINISTIC;
    case PoolingMode::kAverageExclusive:
      return dynload::Pooling::Mode::AVGPOOL_COUNT_WITHOUT_PAD;
    case PoolingMode::kAverageInclusive:
      return dynload::Pooling::Mode::AVGPOOL_COUNT_PAD;
    case PoolingMode::kMaximum:
      return dynload::Pooling::Mode::MAXPOOL;
    default:
      PADDLE_THROW(
          phi::errors::Unimplemented("Unexpected MUDNN pooling mode."));
  }
}

// inline ActivationMode StringToActivationMode(const std::string& str) {
//   if (str == "identity") {
//     return ActivationMode::kNone;
//   } else if (str == "sigmoid") {
//     return ActivationMode::kSigmoid;
//   } else if (str == "relu") {
//     return ActivationMode::kRelu;
//   } else if (str == "relu6") {
//     return ActivationMode::kRelu6;
//   } else if (str == "relux") {
//     return ActivationMode::kReluX;
//   } else if (str == "tanh") {
//     return ActivationMode::kTanh;
//   } else if (str == "bandpass") {
//     return ActivationMode::kBandPass;
//   } else {
//     PADDLE_THROW(phi::errors::Unimplemented(
//         "Unknown CUDNN activation string: %s.", str));
//   }
// }

template <typename T>
class CudnnDataType;

template <>
class CudnnDataType<phi::dtype::bfloat16> {
 public:
  static const dynload::Tensor::Type type = dynload::Tensor::Type::BFLOAT16;
  using ScalingParamType = const float;
  using BatchNormParamType = float;
  static ScalingParamType* kOne() {
    static ScalingParamType v = 1.0;
    return &v;
  }
  static ScalingParamType* kZero() {
    static ScalingParamType v = 0.0;
    return &v;
  }
};

template <>
class CudnnDataType<phi::dtype::float16> {
 public:
  static const dynload::Tensor::Type type = dynload::Tensor::Type::HALF;
  // The scaling param type is float for HALF and FLOAT tensors
  using ScalingParamType = const float;
  using BatchNormParamType = float;
  static ScalingParamType* kOne() {
    static ScalingParamType v = 1.0;
    return &v;
  }
  static ScalingParamType* kZero() {
    static ScalingParamType v = 0.0;
    return &v;
  }
};

template <>
class CudnnDataType<float> {
 public:
  static const dynload::Tensor::Type type = dynload::Tensor::Type::FLOAT;
  using ScalingParamType = const float;
  using BatchNormParamType = float;
  static ScalingParamType* kOne() {
    static ScalingParamType v = 1.0;
    return &v;
  }
  static ScalingParamType* kZero() {
    static ScalingParamType v = 0.0;
    return &v;
  }
};

template <>
class CudnnDataType<double> {
 public:
  static const dynload::Tensor::Type type = dynload::Tensor::Type::DOUBLE;
  using ScalingParamType = const double;
  using BatchNormParamType = double;
  static ScalingParamType* kOne() {
    static ScalingParamType v = 1.0;
    return &v;
  }
  static ScalingParamType* kZero() {
    static ScalingParamType v = 0.0;
    return &v;
  }
};

inline dynload::Tensor::Format GetCudnnTensorFormat(
    const DataLayout& order) {  // Not use
  switch (order) {
    case DataLayout::kNHWC:
      return dynload::Tensor::Format::NHWC;
    case DataLayout::kNCHW:
      return dynload::Tensor::Format::NCHW;
    case DataLayout::kNCDHW:
      return dynload::Tensor::Format::NCDHW;
    case DataLayout::kNDHWC:
      return dynload::Tensor::Format::NDHWC;
    default:
      PADDLE_THROW(phi::errors::Unimplemented(
          "MUDNN has no equivalent dataLayout for input order."));
  }
  return dynload::Tensor::Format::NCHW;
}

class ScopedTensorDescriptor {
 public:
  ScopedTensorDescriptor() {}
  ~ScopedTensorDescriptor() PADDLE_MAY_THROW {}

  inline dynload::Tensor descriptor(const dynload::Tensor::Format format,
                                    const dynload::Tensor::Type type,
                                    const std::vector<int>& dims,
                                    const int groups = 1) {
    // the format is not used now, will add later
    std::vector<int> strides(dims.size());
    strides[dims.size() - 1] = 1;
    for (int i = dims.size() - 2; i >= 0; i--) {
      strides[i] = dims[i + 1] * strides[i + 1];
    }
    // Update tensor descriptor dims setting if groups > 1
    // NOTE: Here, Assume using NCHW or NCDHW order
    std::vector<int> dims_with_group(dims.begin(), dims.end());
    if (groups > 1) {
      dims_with_group[1] = dims_with_group[1] / groups;
    }

    PADDLE_ENFORCE_EQ(
        format,
        dynload::Tensor::Format::NCHW,
        phi::errors::InvalidArgument("format should ONLY be NCHW in MUDNN."));

    desc_.SetNdInfo(
        dims_with_group.size(), dims_with_group.data(), strides.data());
    desc_.SetType(type);
    desc_.SetFormat(format);

    return desc_;
  }

  template <typename T>
  inline dynload::Tensor& descriptor(const DataLayout& order,
                                     const std::vector<int>& dims,
                                     const int groups = 1) {
    descriptor(
        GetCudnnTensorFormat(order), CudnnDataType<T>::type, dims, groups);
    return desc_;
  }

  template <typename T>
  inline dynload::Tensor& descriptor(const phi::DenseTensor& tensor,
                                     const DataLayout& order,
                                     const std::vector<int>& dims,
                                     const int groups = 1) {
    desc_.SetAddr(tensor.data());
    descriptor<T>(order, dims, groups);
    return desc_;
  }

  template <typename T>
  inline dynload::Tensor& descriptor(const T* data,
                                     const DataLayout& order,
                                     const std::vector<int>& dims,
                                     const int groups = 1) {
    desc_.SetAddr(data);
    descriptor<T>(order, dims, groups);
    return desc_;
  }

  inline dynload::Tensor& descriptor(const dynload::Tensor::Type mudnn_type,
                                     const std::vector<int>& dim,
                                     const std::vector<int>& stride) {
    desc_.SetType(mudnn_type);
    desc_.SetNdInfo(dim.size(), dim.data(), stride.data());
    return desc_;
  }

  template <typename T>
  inline dynload::Tensor& descriptor(const std::vector<int>& dim,
                                     const std::vector<int>& stride) {
    descriptor(CudnnDataType<T>::type, dim, stride);
    return desc_;
  }

  inline dynload::Tensor& desc() { return desc_; }

 private:
  dynload::Tensor desc_;
  DISABLE_COPY_AND_ASSIGN(ScopedTensorDescriptor);
};

class ScopedPoolingDescriptor {
 public:
  ScopedPoolingDescriptor() {}
  ~ScopedPoolingDescriptor() PADDLE_MAY_THROW {}

  inline dynload::Pooling& descriptor(const PoolingMode& mode,
                                      const std::vector<int>& kernel,
                                      const std::vector<int>& pads,
                                      const std::vector<int>& strides) {
    PADDLE_ENFORCE_EQ(kernel.size(),
                      pads.size(),
                      phi::errors::InvalidArgument(
                          "The size of kernel and pads should be equal. But "
                          "received size of kernel is %d, size of pads is %d.",
                          kernel.size(),
                          pads.size()));
    PADDLE_ENFORCE_EQ(
        kernel.size(),
        strides.size(),
        phi::errors::InvalidArgument(
            "The size of kernel and strides should be equal. But "
            "received size of kernel is %d, size of strides is %d.",
            kernel.size(),
            strides.size()));
    const std::vector<int> dilation(kernel.size(), 1);
    desc_.SetNdInfo(kernel.size(),
                    kernel.data(),
                    pads.data(),
                    strides.data(),
                    dilation.data());
    desc_.SetMode(GetPoolingMode(mode));
    return desc_;
  }

  dynload::Pooling& desc() { return desc_; }

 private:
  dynload::Pooling desc_;
  DISABLE_COPY_AND_ASSIGN(ScopedPoolingDescriptor);
};

class ScopedSoftmaxDescriptor {
 public:
  ScopedSoftmaxDescriptor() {}
  ~ScopedSoftmaxDescriptor() PADDLE_MAY_THROW {}

  inline dynload::Softmax& descriptor(const dynload::Softmax::Mode& mode,
                                      const dynload::Softmax::Algorithm& algo,
                                      const int& dim) {
    desc_.SetMode(mode);
    desc_.SetDim(dim);
    desc_.SetAlgorithm(algo);
    return desc_;
  }

  dynload::Softmax& desc() { return desc_; }

 private:
  dynload::Softmax desc_;
  DISABLE_COPY_AND_ASSIGN(ScopedSoftmaxDescriptor);
};

static void InternalMemFree(void* ptr) {
  if (!ptr) {
    return;
  }
  PADDLE_ENFORCE_GPU_SUCCESS(musaFree(ptr));
}

static dynload::MemoryHandler InternalMemAlloc(size_t s) {
  void* data = nullptr;
  if (s) {
    PADDLE_ENFORCE_GPU_SUCCESS(musaMalloc(&data, s));
  }
  return dynload::MemoryHandler(data, InternalMemFree);
}

#if 0
class ScopedBatchNormDescriptor {
 public:
  ScopedBatchNormDescriptor() {}
  ~ScopedBatchNormDescriptor() PADDLE_MAY_THROW {}

  inline dynload::BatchNorm& descriptor(const double& eps) {
    desc_.SetEpsilon(eps);
    return desc_;
  }

  dynload::BatchNorm& desc() { return desc_; }

 private:
  dynload::BatchNorm desc_;
  DISABLE_COPY_AND_ASSIGN(ScopedBatchNormDescriptor);
};

class ScopedActivationDescriptor {
 public:
  ScopedActivationDescriptor() {
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnCreateActivationDescriptor(&desc_));
  }
  ~ScopedActivationDescriptor() PADDLE_MAY_THROW {
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cudnnDestroyActivationDescriptor(desc_));
  }

  template <typename T>
  inline cudnnActivationDescriptor_t descriptor(
      const std::string& act, double value_max = static_cast<double>(0.)) {
    double relu_ceiling = 0.0;
    ActivationMode activation_mode = StringToActivationMode(act);
    cudnnActivationMode_t mode;
    switch (activation_mode) {
#if CUDNN_VERSION >= 7100
      case ActivationMode::kNone:
        mode = CUDNN_ACTIVATION_IDENTITY;
        break;
#endif
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
        PADDLE_THROW(phi::errors::Unimplemented(
            "Unrecognized CUDNN activation mode: %d.",
            static_cast<int>(activation_mode)));
    }
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cudnnSetActivationDescriptor(
        desc_, mode, CUDNN_NOT_PROPAGATE_NAN, relu_ceiling));
    return desc_;
  }

 private:
  cudnnActivationDescriptor_t desc_;
  DISABLE_COPY_AND_ASSIGN(ScopedActivationDescriptor);
};
#endif  // 0

}  // namespace gpu
}  // namespace backends
}  // namespace phi
