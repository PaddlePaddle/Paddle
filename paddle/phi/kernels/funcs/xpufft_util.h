// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
#ifdef PADDLE_WITH_XPU_FFT
#include <vector>

#include "paddle/common/ddim.h"
#include "paddle/phi/backends/dynload/xpufft.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/kernels/funcs/fft.h"
#include "paddle/phi/kernels/funcs/fft_key.h"

namespace phi {
namespace funcs {
namespace detail {

// An RAII encapsulation of cuFFTHandle
class CuFFTHandle {
 public:
  CuFFTHandle() {
    PADDLE_ENFORCE_FFT_SUCCESS(phi::dynload::cufftCreate(&handle_));
  }

  CuFFTHandle(const CuFFTHandle& other) = delete;
  CuFFTHandle& operator=(const CuFFTHandle& other) = delete;

  CuFFTHandle(CuFFTHandle&& other) = delete;
  CuFFTHandle& operator=(CuFFTHandle&& other) = delete;

  ::cufftHandle& get() { return handle_; }
  const ::cufftHandle& get() const { return handle_; }

  ~CuFFTHandle() { phi::dynload::cufftDestroy(handle_); }

 private:
  ::cufftHandle handle_;
};

inline cufftType type_input(FFTTransformType type) {
  switch (type) {
    case FFTTransformType::C2C:
      return CUFFT_C2C;
    case FFTTransformType::C2R:
      return CUFFT_C2R;
    case FFTTransformType::R2C:
      return CUFFT_R2C;
  }
  PADDLE_THROW(common::errors::InvalidArgument("Unknown FFTTransformType"));
}

class FFTConfig {
 public:
  using plan_size_type = int;  // NOLINT (be consistent with cufft)
  explicit FFTConfig(const FFTConfigKey& key)
      : FFTConfig(
            std::vector<int64_t>(key.sizes_, key.sizes_ + key.signal_ndim_ + 1),
            key.fft_type_,
            key.value_type_) {}
  // sizes are full signal, including batch size and always two-sided
  FFTConfig(const std::vector<int64_t>& sizes,
            FFTTransformType fft_type,
            DataType precision)
      : fft_type_(fft_type), precision_(precision) {
    const auto batch_size = static_cast<plan_size_type>(sizes[0]);
    std::vector<plan_size_type> signal_sizes(sizes.cbegin() + 1, sizes.cend());
    const int signal_ndim = sizes.size() - 1;

    // Check if the number of elements participating in FFT transformation is
    // greater than 8 (XPU hardware requirement)
    for (int i = 0; i < signal_ndim; ++i) {
      if (signal_sizes[i] <= 8) {
        PADDLE_THROW(phi::errors::InvalidArgument(
            "XPU FFT requires all axes to have greater than 8 elements, "
            "but axis %d has size %d.Set XFFT_DEBUG=1 environment variable "
            "to inspect dimensions.",
            i,
            signal_sizes[i]));
      }
    }

    cufftType exec_type;
    exec_type = type_input(fft_type);

    // disable auto allocation of workspace to use allocator from the framework
    PADDLE_ENFORCE_FFT_SUCCESS(
        phi::dynload::cufftSetAutoAllocation(plan(), /* autoAllocate */ 0));

    PADDLE_ENFORCE_FFT_SUCCESS(
        phi::dynload::cufftPlanMany(const_cast<cufftHandle*>(&plan()),
                                    signal_ndim,
                                    signal_sizes.data(),
                                    /* inembed */ nullptr,
                                    /* base_istride */ 1,
                                    /* idist */ 1,
                                    /* onembed */ nullptr,
                                    /* base_ostride */ 1,
                                    /* odist */ 1,
                                    exec_type,
                                    batch_size));

    PADDLE_ENFORCE_FFT_SUCCESS(
        phi::dynload::cufftGetSizeMany(plan(),
                                       signal_ndim,
                                       signal_sizes.data(),
                                       /* inembed */ nullptr,
                                       /* base_istride */ 1,
                                       /* idist */ 1,
                                       /* onembed */ nullptr,
                                       /* base_ostride */ 1,
                                       /* odist */ 1,
                                       exec_type,
                                       batch_size,
                                       &ws_size_));
  }

  FFTConfig(const FFTConfig& other) = delete;
  FFTConfig& operator=(const FFTConfig& other) = delete;

  FFTConfig(FFTConfig&& other) = delete;
  FFTConfig& operator=(FFTConfig&& other) = delete;

  const cufftHandle& plan() const { return plan_.get(); }
  FFTTransformType transform_type() const { return fft_type_; }
  DataType data_type() const { return precision_; }
  size_t workspace_size() const { return ws_size_; }

 private:
  CuFFTHandle plan_;
  size_t ws_size_;  // workspace size in bytes
  FFTTransformType fft_type_;
  DataType precision_;
};

// NOTE: R2C is forward-only, C2R is backward only
static void exec_plan(const FFTConfig& config,
                      void* in_data,
                      void* out_data,
                      bool forward) {
  auto& plan = config.plan();
  switch (config.transform_type()) {
    case FFTTransformType::C2C:
      PADDLE_ENFORCE_FFT_SUCCESS(phi::dynload::cufftExecC2C(
          plan,
          reinterpret_cast<cuFloatComplex*>(in_data),
          reinterpret_cast<cuFloatComplex*>(out_data),
          forward ? CUFFT_FORWARD : CUFFT_INVERSE));
      return;
    case FFTTransformType::C2R:
      PADDLE_ENFORCE_FFT_SUCCESS(
          phi::dynload::cufftExecC2R(plan,
                                     reinterpret_cast<cuFloatComplex*>(in_data),
                                     reinterpret_cast<float*>(out_data)));
      return;
    case FFTTransformType::R2C:
      PADDLE_ENFORCE_FFT_SUCCESS(phi::dynload::cufftExecR2C(
          plan,
          reinterpret_cast<float*>(in_data),
          reinterpret_cast<cuFloatComplex*>(out_data)));
      return;
  }
  PADDLE_THROW(common::errors::InvalidArgument("Unknown FFTTransformType"));
}

}  // namespace detail
}  // namespace funcs
}  // namespace phi
#endif
