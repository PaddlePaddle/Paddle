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
#include <vector>

#include "paddle/phi/backends/dynload/hipfft.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/kernels/funcs/fft.h"
#include "paddle/phi/kernels/funcs/fft_key.h"

namespace phi {
namespace funcs {
namespace detail {

// An RAII encapsulation of hipFFTHandle
class HIPFFTHandle {
 public:
  HIPFFTHandle() {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::hipfftCreate(&handle_));
  }

  HIPFFTHandle(const HIPFFTHandle& other) = delete;
  HIPFFTHandle& operator=(const HIPFFTHandle& other) = delete;

  HIPFFTHandle(HIPFFTHandle&& other) = delete;
  HIPFFTHandle& operator=(HIPFFTHandle&& other) = delete;

  ::hipfftHandle& get() { return handle_; }
  const ::hipfftHandle& get() const { return handle_; }

  ~HIPFFTHandle() { phi::dynload::hipfftDestroy(handle_); }

 private:
  ::hipfftHandle handle_;
};

class FFTConfig {
 public:
  using plan_size_type = int;
  explicit FFTConfig(const FFTConfigKey& key)
      : FFTConfig(
            std::vector<int64_t>(key.sizes_, key.sizes_ + key.signal_ndim_ + 1),
            key.fft_type_,
            key.value_type_) {}
  FFTConfig(const std::vector<int64_t>& sizes,
            FFTTransformType fft_type,
            DataType precision)
      : fft_type_(fft_type), precision_(precision) {
    std::vector<plan_size_type> signal_sizes(sizes.begin() + 1, sizes.end());
    const auto batch_size = static_cast<plan_size_type>(sizes[0]);
    const int signal_ndim = sizes.size() - 1;

    hipfftType exec_type = [&]() {
      if (precision == DataType::FLOAT32) {
        switch (fft_type) {
          case FFTTransformType::C2C:
            return HIPFFT_C2C;
          case FFTTransformType::R2C:
            return HIPFFT_R2C;
          case FFTTransformType::C2R:
            return HIPFFT_C2R;
        }
      } else if (precision == DataType::FLOAT64) {
        switch (fft_type) {
          case FFTTransformType::C2C:
            return HIPFFT_Z2Z;
          case FFTTransformType::R2C:
            return HIPFFT_D2Z;
          case FFTTransformType::C2R:
            return HIPFFT_Z2D;
        }
      }
      PADDLE_THROW(common::errors::InvalidArgument(
          "Only transforms of type float32 and float64 are supported."));
    }();

    // disable auto allocation of workspace to use allocator from the framework
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::hipfftSetAutoAllocation(plan(), /* autoAllocate */ 0));
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::hipfftMakePlanMany(plan(),
                                         signal_ndim,
                                         signal_sizes.data(),
                                         /* inembed */ nullptr,
                                         /* base_istride */ 1,
                                         /* idist */ 1,
                                         /* onembed */ nullptr,
                                         /* base_ostride */ 1,
                                         /* odist */ 1,
                                         exec_type,
                                         batch_size,
                                         &ws_size_));
  }

  const hipfftHandle& plan() const { return plan_.get(); }
  FFTTransformType transform_type() const { return fft_type_; }
  DataType data_type() const { return precision_; }
  size_t workspace_size() const { return ws_size_; }

 private:
  HIPFFTHandle plan_;
  size_t ws_size_;  // workspace size in bytes
  FFTTransformType fft_type_;
  DataType precision_;
};

// NOTE: R2C is forward-only, C2R is backward only
static void exec_plan(const FFTConfig& config,
                      void* in_data,
                      void* out_data,
                      bool forward) {
  const hipfftHandle& plan = config.plan();

  DataType value_type = config.data_type();
  if (value_type == DataType::FLOAT32) {
    switch (config.transform_type()) {
      case FFTTransformType::C2C: {
        PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::hipfftExecC2C(
            plan,
            static_cast<hipfftComplex*>(in_data),
            static_cast<hipfftComplex*>(out_data),
            forward ? HIPFFT_FORWARD : HIPFFT_BACKWARD));
        return;
      }
      case FFTTransformType::R2C: {
        PADDLE_ENFORCE_GPU_SUCCESS(
            phi::dynload::hipfftExecR2C(plan,
                                        static_cast<hipfftReal*>(in_data),
                                        static_cast<hipfftComplex*>(out_data)));
        return;
      }
      case FFTTransformType::C2R: {
        PADDLE_ENFORCE_GPU_SUCCESS(
            phi::dynload::hipfftExecC2R(plan,
                                        static_cast<hipfftComplex*>(in_data),
                                        static_cast<hipfftReal*>(out_data)));
        return;
      }
    }
  } else if (value_type == DataType::FLOAT64) {
    switch (config.transform_type()) {
      case FFTTransformType::C2C: {
        PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::hipfftExecZ2Z(
            plan,
            static_cast<hipfftDoubleComplex*>(in_data),
            static_cast<hipfftDoubleComplex*>(out_data),
            forward ? HIPFFT_FORWARD : HIPFFT_BACKWARD));
        return;
      }
      case FFTTransformType::R2C: {
        PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::hipfftExecD2Z(
            plan,
            static_cast<hipfftDoubleReal*>(in_data),
            static_cast<hipfftDoubleComplex*>(out_data)));
        return;
      }
      case FFTTransformType::C2R: {
        PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::hipfftExecZ2D(
            plan,
            static_cast<hipfftDoubleComplex*>(in_data),
            static_cast<hipfftDoubleReal*>(out_data)));
        return;
      }
    }
  }
  PADDLE_THROW(common::errors::InvalidArgument(
      "hipFFT only support transforms of type float32 and float64"));
}

}  // namespace detail
}  // namespace funcs
}  // namespace phi
