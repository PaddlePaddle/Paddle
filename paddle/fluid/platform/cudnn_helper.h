/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/dynload/cudnn.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/fluid/platform/macros.h"

DECLARE_bool(cudnn_deterministic);

namespace paddle {
namespace platform {

inline const char* cudnnGetErrorString(cudnnStatus_t status) {
  switch (status) {
    case CUDNN_STATUS_SUCCESS:
      return "CUDNN_STATUS_SUCCESS";
    case CUDNN_STATUS_NOT_INITIALIZED:
      return "CUDNN_STATUS_NOT_INITIALIZED";
    case CUDNN_STATUS_ALLOC_FAILED:
      return "CUDNN_STATUS_ALLOC_FAILED";
    case CUDNN_STATUS_BAD_PARAM:
      return "CUDNN_STATUS_BAD_PARAM";
    case CUDNN_STATUS_INTERNAL_ERROR:
      return "CUDNN_STATUS_INTERNAL_ERROR";
    case CUDNN_STATUS_INVALID_VALUE:
      return "CUDNN_STATUS_INVALID_VALUE";
    case CUDNN_STATUS_ARCH_MISMATCH:
      return "CUDNN_STATUS_ARCH_MISMATCH";
    case CUDNN_STATUS_MAPPING_ERROR:
      return "CUDNN_STATUS_MAPPING_ERROR";
    case CUDNN_STATUS_EXECUTION_FAILED:
      return "CUDNN_STATUS_EXECUTION_FAILED";
    case CUDNN_STATUS_NOT_SUPPORTED:
      return "CUDNN_STATUS_NOT_SUPPORTED";
    case CUDNN_STATUS_LICENSE_ERROR:
      return "CUDNN_STATUS_LICENSE_ERROR";
    default:
      return "Unknown cudnn error number";
  }
}

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

enum class ActivationMode {
  kNone,  // activation identity
  kSigmoid,
  kRelu,
  kRelu6,
  kReluX,
  kTanh,
  kBandPass,
};

#if CUDNN_VERSION < 6000
#pragma message "CUDNN version under 6.0 is supported at best effort."
#pragma message "We strongly encourage you to move to 6.0 and above."
#pragma message "This message is intended to annoy you enough to update."
#pragma message \
    "please see https://docs.nvidia.com/deeplearning/sdk/cudnn-release-notes/"

inline cudnnPoolingMode_t GetPoolingMode(const PoolingMode& mode) {
  switch (mode) {
    case PoolingMode::kMaximumDeterministic:
      return CUDNN_POOLING_MAX;
    case PoolingMode::kAverageExclusive:
      return CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
    case PoolingMode::kAverageInclusive:
      return CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
    case PoolingMode::kMaximum:
      return CUDNN_POOLING_MAX;
    default:
      PADDLE_THROW("Unexpected pooling mode.");
  }
}
#else

inline cudnnPoolingMode_t GetPoolingMode(const PoolingMode& mode) {
  switch (mode) {
    case PoolingMode::kMaximumDeterministic:
      return CUDNN_POOLING_MAX_DETERMINISTIC;
    case PoolingMode::kAverageExclusive:
      return CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
    case PoolingMode::kAverageInclusive:
      return CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
    case PoolingMode::kMaximum:
      return CUDNN_POOLING_MAX;
    default:
      PADDLE_THROW("Unexpected pooling mode.");
  }
}
#endif  // CUDNN_VERSION < 6000

inline ActivationMode StringToActivationMode(const std::string& str) {
  if (str == "identity") {
    return ActivationMode::kNone;
  } else if (str == "sigmoid") {
    return ActivationMode::kSigmoid;
  } else if (str == "relu") {
    return ActivationMode::kRelu;
  } else if (str == "relu6") {
    return ActivationMode::kRelu6;
  } else if (str == "relux") {
    return ActivationMode::kReluX;
  } else if (str == "tanh") {
    return ActivationMode::kTanh;
  } else if (str == "bandpass") {
    return ActivationMode::kBandPass;
  } else {
    PADDLE_THROW("Unknown activation string: %s", str);
  }
}

template <typename T>
class CudnnDataType;

template <>
class CudnnDataType<float16> {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_HALF;
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
  static const cudnnDataType_t type = CUDNN_DATA_FLOAT;
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
  static const cudnnDataType_t type = CUDNN_DATA_DOUBLE;
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

inline cudnnTensorFormat_t GetCudnnTensorFormat(
    const DataLayout& order) {  // Not use
  switch (order) {
    case DataLayout::kNHWC:
      return CUDNN_TENSOR_NHWC;
    case DataLayout::kNCHW:
      return CUDNN_TENSOR_NCHW;
    case DataLayout::kNCDHW:
      return CUDNN_TENSOR_NCHW;  // NOTE: cudnn treat NdTensor as the same
    case DataLayout::kNDHWC:
      return CUDNN_TENSOR_NHWC;  // add, liyamei
    default:
      PADDLE_THROW("Unknown cudnn equivalent for order");
  }
  return CUDNN_TENSOR_NCHW;
}

class ScopedTensorDescriptor {
 public:
  ScopedTensorDescriptor() {
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnCreateTensorDescriptor(&desc_));
  }
  ~ScopedTensorDescriptor() PADDLE_MAY_THROW {
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnDestroyTensorDescriptor(desc_));
  }

  inline cudnnTensorDescriptor_t descriptor(const cudnnTensorFormat_t format,
                                            const cudnnDataType_t type,
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

    if (dims.size() == 4) {
      if (format == CUDNN_TENSOR_NCHW) {
        PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnSetTensorNdDescriptor(
            desc_, type, dims_with_group.size(), dims_with_group.data(),
            strides.data()));
      } else {  // CUDNN_TENSOR_NHWC
        PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnSetTensor4dDescriptor(
            desc_, format, type, dims[0], dims[3], dims[1], dims[2]));
      }
    } else if (dims.size() == 5) {
      if (format == CUDNN_TENSOR_NCHW) {
        PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnSetTensorNdDescriptor(
            desc_, type, dims_with_group.size(), dims_with_group.data(),
            strides.data()));
      } else {  // CUDNN_TENSOR_NHWC
        PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnSetTensorNdDescriptorEx(
            desc_, format, type, dims.size(), dims.data()));
      }
    }
    return desc_;
  }

  template <typename T>
  inline cudnnTensorDescriptor_t descriptor(const DataLayout& order,
                                            const std::vector<int>& dims,
                                            const int groups = 1) {
    return descriptor(GetCudnnTensorFormat(order), CudnnDataType<T>::type, dims,
                      groups);
  }

 private:
  cudnnTensorDescriptor_t desc_;
  DISABLE_COPY_AND_ASSIGN(ScopedTensorDescriptor);
};

class ScopedFilterDescriptor {
 public:
  ScopedFilterDescriptor() {
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnCreateFilterDescriptor(&desc_));
  }
  ~ScopedFilterDescriptor() PADDLE_MAY_THROW {
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnDestroyFilterDescriptor(desc_));
  }

  inline cudnnFilterDescriptor_t descriptor(const cudnnTensorFormat_t format,
                                            const cudnnDataType_t type,
                                            const std::vector<int>& kernel,
                                            const int groups = 1) {
    // filter layout: MCHW(MCDHW), where M is the number of
    // output image channels, C is the number of input image channels,
    // D is the depth of the filter, H is the height of the filter, and W is the
    // width of the filter.
    std::vector<int> kernel_with_group(kernel.begin(), kernel.end());
    if (groups > 1) {
      kernel_with_group[0] /= groups;
      // NOTE: input filter(C) of the filter is already asserted to be C/groups.
    }
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnSetFilterNdDescriptor(
        desc_, type, format, kernel_with_group.size(),
        kernel_with_group.data()));
    return desc_;
  }

  template <typename T>
  inline cudnnFilterDescriptor_t descriptor(const DataLayout& order,
                                            const std::vector<int>& kernel,
                                            const int groups = 1) {
    return descriptor(GetCudnnTensorFormat(order), CudnnDataType<T>::type,
                      kernel, groups);
  }

 private:
  cudnnFilterDescriptor_t desc_;
  DISABLE_COPY_AND_ASSIGN(ScopedFilterDescriptor);
};

class ScopedConvolutionDescriptor {
 public:
  ScopedConvolutionDescriptor() {
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnCreateConvolutionDescriptor(&desc_));
  }
  ~ScopedConvolutionDescriptor() PADDLE_MAY_THROW {
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnDestroyConvolutionDescriptor(desc_));
  }

  inline cudnnConvolutionDescriptor_t descriptor(
      cudnnDataType_t type, const std::vector<int>& pads,
      const std::vector<int>& strides, const std::vector<int>& dilations) {
    PADDLE_ENFORCE_EQ(pads.size(), strides.size());
    PADDLE_ENFORCE_EQ(pads.size(), dilations.size());

#if !CUDNN_VERSION_MIN(6, 0, 0)
    // cudnn v5 does not support dilation conv, the argument is called upscale
    // instead of dilations and it is must be one.
    for (size_t i = 0; i < dilations.size(); ++i) {
      PADDLE_ENFORCE_EQ(
          dilations[i], 1,
          "Dilations conv is not supported in this cuDNN version(%d.%d.%d).",
          CUDNN_VERSION / 1000, CUDNN_VERSION % 1000 / 100,
          CUDNN_VERSION % 100);
    }
#endif

    cudnnDataType_t compute_type =
        (type == CUDNN_DATA_DOUBLE) ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT;
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnSetConvolutionNdDescriptor(
        desc_, pads.size(), pads.data(), strides.data(), dilations.data(),
        CUDNN_CROSS_CORRELATION, compute_type));
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
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnCreatePoolingDescriptor(&desc_));
  }
  ~ScopedPoolingDescriptor() PADDLE_MAY_THROW {
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnDestroyPoolingDescriptor(desc_));
  }

  inline cudnnPoolingDescriptor_t descriptor(const PoolingMode& mode,
                                             const std::vector<int>& kernel,
                                             const std::vector<int>& pads,
                                             const std::vector<int>& strides) {
    PADDLE_ENFORCE_EQ(kernel.size(), pads.size());
    PADDLE_ENFORCE_EQ(kernel.size(), strides.size());
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnSetPoolingNdDescriptor(
        desc_, (GetPoolingMode(mode)),
        CUDNN_PROPAGATE_NAN,  // Always propagate nans.
        kernel.size(), kernel.data(), pads.data(), strides.data()));
    return desc_;
  }

 private:
  cudnnPoolingDescriptor_t desc_;
  DISABLE_COPY_AND_ASSIGN(ScopedPoolingDescriptor);
};

class ScopedSpatialTransformerDescriptor {
 public:
  ScopedSpatialTransformerDescriptor() {
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnCreateSpatialTransformerDescriptor(&desc_));
  }
  ~ScopedSpatialTransformerDescriptor() PADDLE_MAY_THROW {
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnDestroySpatialTransformerDescriptor(desc_));
  }

  template <typename T>
  inline cudnnSpatialTransformerDescriptor_t descriptor(const int nbDims,
                                                        const int dimA[]) {
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnSetSpatialTransformerNdDescriptor(
        desc_, CUDNN_SAMPLER_BILINEAR, CudnnDataType<T>::type, nbDims, dimA));
    return desc_;
  }

 private:
  cudnnSpatialTransformerDescriptor_t desc_;
  DISABLE_COPY_AND_ASSIGN(ScopedSpatialTransformerDescriptor);
};

class ScopedActivationDescriptor {
 public:
  ScopedActivationDescriptor() {
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnCreateActivationDescriptor(&desc_));
  }
  ~ScopedActivationDescriptor() PADDLE_MAY_THROW {
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnDestroyActivationDescriptor(desc_));
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
        PADDLE_THROW("unrecognized activation mode: %d .",
                     static_cast<int>(activation_mode));
    }
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnSetActivationDescriptor(
        desc_, mode, CUDNN_NOT_PROPAGATE_NAN, relu_ceiling));
    return desc_;
  }

 private:
  cudnnActivationDescriptor_t desc_;
  DISABLE_COPY_AND_ASSIGN(ScopedActivationDescriptor);
};

inline bool CanCUDNNBeUsed(const framework::ExecutionContext& ctx) {
  bool use_cudnn = ctx.Attr<bool>("use_cudnn");
  use_cudnn &= paddle::platform::is_gpu_place(ctx.GetPlace());
#ifdef PADDLE_WITH_CUDA
  if (use_cudnn) {
    auto& dev_ctx = ctx.device_context<platform::CUDADeviceContext>();
    use_cudnn &= dev_ctx.cudnn_handle() != nullptr;
  }
#endif
  return use_cudnn;
}

#if CUDNN_VERSION >= 7001
class ScopedCTCLossDescriptor {
 public:
  ScopedCTCLossDescriptor() {
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnCreateCTCLossDescriptor(&desc_));
  }
  ~ScopedCTCLossDescriptor() PADDLE_MAY_THROW {
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::cudnnDestroyCTCLossDescriptor(desc_));
  }

  template <typename T>
  inline cudnnCTCLossDescriptor_t descriptor() {
    PADDLE_ENFORCE_CUDA_SUCCESS(
        dynload::cudnnSetCTCLossDescriptor(desc_, CudnnDataType<T>::type));
    return desc_;
  }

 private:
  cudnnCTCLossDescriptor_t desc_;
  DISABLE_COPY_AND_ASSIGN(ScopedCTCLossDescriptor);
};
#endif

}  // namespace platform
}  // namespace paddle
