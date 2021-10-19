// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/spectral_op.h"

#ifdef PADDLE_WITH_HIP
#include "paddle/fluid/platform/dynload/hipfft.h"
#endif

#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/dynload/cufft.h"
#endif

namespace paddle {
namespace operators {
using ScalarType = framework::proto::VarType::Type;
const int64_t kMaxCUFFTNdim = 3;
const int64_t kMaxDataNdim = kMaxCUFFTNdim + 1;
// This struct is used to easily compute hashes of the
// parameters. It will be the **key** to the plan cache.
struct PlanKey {
  // between 1 and kMaxCUFFTNdim, i.e., 1 <= signal_ndim <= 3
  int64_t signal_ndim_;
  // These include additional batch dimension as well.
  int64_t sizes_[kMaxDataNdim];
  int64_t input_shape_[kMaxDataNdim];
  int64_t output_shape_[kMaxDataNdim];
  FFTTransformType fft_type_;
  ScalarType value_type_;

  PlanKey() = default;

  PlanKey(const std::vector<int64_t>& in_shape,
          const std::vector<int64_t>& out_shape,
          const std::vector<int64_t>& signal_size, FFTTransformType fft_type,
          ScalarType value_type) {
    // Padding bits must be zeroed for hashing
    memset(this, 0, sizeof(*this));
    signal_ndim_ = signal_size.size() - 1;
    fft_type_ = fft_type;
    value_type_ = value_type;

    std::copy(signal_size.cbegin(), signal_size.cend(), sizes_);
    std::copy(in_shape.cbegin(), in_shape.cend(), input_shape_);
    std::copy(out_shape.cbegin(), out_shape.cend(), output_shape_);
  }
};

#if defined(PADDLE_WITH_CUDA)
// An RAII encapsulation of cuFFTHandle
class CuFFTHandle {
  ::cufftHandle handle_;

 public:
  CuFFTHandle() {
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cufftCreate(&handle_));
  }

  ::cufftHandle& get() { return handle_; }
  const ::cufftHandle& get() const { return handle_; }

  ~CuFFTHandle() {
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cufftDestroy(handle_));
  }
};

using plan_size_type = long long int;  // NOLINT
// This class contains all the information needed to execute a cuFFT plan:
//   1. the plan
//   2. the workspace size needed
class CuFFTConfig {
 public:
  // Only move semantics is enought for this class. Although we already use
  // unique_ptr for the plan, still remove copy constructor and assignment op so
  // we don't accidentally copy and take perf hit.
  explicit CuFFTConfig(const PlanKey& plan_key)
      : CuFFTConfig(
            std::vector<int64_t>(plan_key.sizes_,
                                 plan_key.sizes_ + plan_key.signal_ndim_ + 1),
            plan_key.signal_ndim_, plan_key.fft_type_, plan_key.value_type_) {}

  // sizes are full signal, including batch size and always two-sided
  CuFFTConfig(const std::vector<int64_t>& sizes, const int64_t signal_ndim,
              FFTTransformType fft_type, ScalarType dtype)
      : fft_type_(fft_type), value_type_(dtype) {
    // signal sizes (excluding batch dim)
    std::vector<plan_size_type> signal_sizes(sizes.begin() + 1, sizes.end());

    // input batch size
    const auto batch = static_cast<plan_size_type>(sizes[0]);
    // const int64_t signal_ndim = sizes.size() - 1;
    PADDLE_ENFORCE_EQ(signal_ndim, sizes.size() - 1,
                      platform::errors::InvalidArgument(
                          "The signal_ndim must be equal to sizes.size() - 1,"
                          "But signal_ndim is: [%d], sizes.size() - 1 is: [%d]",
                          signal_ndim, sizes.size() - 1));

    cudaDataType itype, otype, exec_type;
    const auto complex_input = has_complex_input(fft_type);
    const auto complex_output = has_complex_output(fft_type);
    if (dtype == framework::proto::VarType::FP32) {
      itype = complex_input ? CUDA_C_32F : CUDA_R_32F;
      otype = complex_output ? CUDA_C_32F : CUDA_R_32F;
      exec_type = CUDA_C_32F;
    } else if (dtype == framework::proto::VarType::FP64) {
      itype = complex_input ? CUDA_C_64F : CUDA_R_64F;
      otype = complex_output ? CUDA_C_64F : CUDA_R_64F;
      exec_type = CUDA_C_64F;
    } else if (dtype == framework::proto::VarType::FP16) {
      itype = complex_input ? CUDA_C_16F : CUDA_R_16F;
      otype = complex_output ? CUDA_C_16F : CUDA_R_16F;
      exec_type = CUDA_C_16F;
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "cuFFT only support transforms of type float16, float32 and "
          "float64"));
    }

    // disable auto allocation of workspace to use allocator from the framework
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cufftSetAutoAllocation(
        plan(), /* autoAllocate */ 0));

    size_t ws_size_t;

    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cufftXtMakePlanMany(
        plan(), signal_ndim, signal_sizes.data(),
        /* inembed */ nullptr, /* base_istride */ 1, /* idist */ 1, itype,
        /* onembed */ nullptr, /* base_ostride */ 1, /* odist */ 1, otype,
        batch, &ws_size_t, exec_type));

    ws_size = ws_size_t;
  }

  const cufftHandle& plan() const { return plan_ptr.get(); }

  FFTTransformType transform_type() const { return fft_type_; }
  ScalarType data_type() const { return value_type_; }
  size_t workspace_size() const { return ws_size; }

 private:
  CuFFTHandle plan_ptr;
  size_t ws_size;
  FFTTransformType fft_type_;
  ScalarType value_type_;
};

#elif defined(PADDLE_WITH_HIP)
// An RAII encapsulation of cuFFTHandle
class HIPFFTHandle {
  ::hipfftHandle handle_;

 public:
  HIPFFTHandle() {
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::hipfftCreate(&handle_));
  }

  ::hipfftHandle& get() { return handle_; }
  const ::hipfftHandle& get() const { return handle_; }

  ~HIPFFTHandle() {
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::hipfftDestroy(handle_));
  }
};
using plan_size_type = int;
// This class contains all the information needed to execute a cuFFT plan:
//   1. the plan
//   2. the workspace size needed
class HIPFFTConfig {
 public:
  // Only move semantics is enought for this class. Although we already use
  // unique_ptr for the plan, still remove copy constructor and assignment op so
  // we don't accidentally copy and take perf hit.
  explicit HIPFFTConfig(const PlanKey& plan_key)
      : HIPFFTConfig(
            std::vector<int64_t>(plan_key.sizes_,
                                 plan_key.sizes_ + plan_key.signal_ndim_ + 1),
            plan_key.signal_ndim_, plan_key.fft_type_, plan_key.value_type_) {}

  // sizes are full signal, including batch size and always two-sided
  HIPFFTConfig(const std::vector<int64_t>& sizes, const int64_t signal_ndim,
               FFTTransformType fft_type, ScalarType dtype)
      : fft_type_(fft_type), value_type_(dtype) {
    // signal sizes (excluding batch dim)
    std::vector<plan_size_type> signal_sizes(sizes.begin() + 1, sizes.end());

    // input batch size
    const auto batch = static_cast<plan_size_type>(sizes[0]);
    // const int64_t signal_ndim = sizes.size() - 1;
    PADDLE_ENFORCE_EQ(signal_ndim, sizes.size() - 1,
                      platform::errors::InvalidArgument(
                          "The signal_ndim must be equal to sizes.size() - 1,"
                          "But signal_ndim is: [%d], sizes.size() - 1 is: [%d]",
                          signal_ndim, sizes.size() - 1));

    hipfftType exec_type = [&] {
      if (dtype == framework::proto::VarType::FP32) {
        switch (fft_type) {
          case FFTTransformType::C2C:
            return HIPFFT_C2C;
          case FFTTransformType::R2C:
            return HIPFFT_R2C;
          case FFTTransformType::C2R:
            return HIPFFT_C2R;
        }
      } else if (dtype == framework::proto::VarType::FP64) {
        switch (fft_type) {
          case FFTTransformType::C2C:
            return HIPFFT_Z2Z;
          case FFTTransformType::R2C:
            return HIPFFT_D2Z;
          case FFTTransformType::C2R:
            return HIPFFT_Z2D;
        }
      }
      PADDLE_THROW(platform::errors::InvalidArgument(
          "hipFFT only support transforms of type float32 and float64"));
    }();

    // disable auto allocation of workspace to use allocator from the framework
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::hipfftSetAutoAllocation(
        plan(), /* autoAllocate */ 0));

    size_t ws_size_t;

    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::hipfftMakePlanMany(
        plan(), signal_ndim, signal_sizes.data(),
        /* inembed */ nullptr, /* base_istride */ 1, /* idist */ 1,
        /* onembed */ nullptr, /* base_ostride */ 1, /* odist */ 1, exec_type,
        batch, &ws_size_t));

    ws_size = ws_size_t;
  }

  const hipfftHandle& plan() const { return plan_ptr.get(); }

  FFTTransformType transform_type() const { return fft_type_; }
  ScalarType data_type() const { return value_type_; }
  size_t workspace_size() const { return ws_size; }

 private:
  HIPFFTHandle plan_ptr;
  size_t ws_size;
  FFTTransformType fft_type_;
  ScalarType value_type_;
};
#endif
}  // namespace operators
}  // namespace paddle
