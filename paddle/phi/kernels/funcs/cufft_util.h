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

#include "paddle/phi/backends/dynload/cufft.h"
#include "paddle/phi/core/ddim.h"
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
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cufftCreate(&handle_));
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

// Returns true if the transform type has complex input
inline bool has_complex_input(FFTTransformType type) {
  switch (type) {
    case FFTTransformType::C2C:
    case FFTTransformType::C2R:
      return true;

    case FFTTransformType::R2C:
      return false;
  }
  PADDLE_THROW(phi::errors::InvalidArgument("Unknown FFTTransformType"));
}

// Returns true if the transform type has complex output
inline bool has_complex_output(FFTTransformType type) {
  switch (type) {
    case FFTTransformType::C2C:
    case FFTTransformType::R2C:
      return true;

    case FFTTransformType::C2R:
      return false;
  }
  PADDLE_THROW(phi::errors::InvalidArgument("Unknown FFTTransformType"));
}

class FFTConfig {
 public:
  using plan_size_type = long long int;  // NOLINT (be consistent with cufft)
  explicit FFTConfig(const FFTConfigKey& key)
      : FFTConfig(
            std::vector<int64_t>(key.sizes_, key.sizes_ + key.signal_ndim_ + 1),
            key.fft_type_,
            key.value_type_) {}
  // sizes are full signal, including batch size and always two-sided
  FFTConfig(const std::vector<int64_t>& sizes,
            FFTTransformType fft_type,
            DataType precison)
      : fft_type_(fft_type), precision_(precison) {
    const auto batch_size = static_cast<plan_size_type>(sizes[0]);
    std::vector<plan_size_type> signal_sizes(sizes.cbegin() + 1, sizes.cend());
    const int signal_ndim = sizes.size() - 1;

    cudaDataType itype, otype, exec_type;
    const bool complex_input = has_complex_input(fft_type);
    const bool complex_output = has_complex_output(fft_type);
    if (precison == DataType::FLOAT32) {
      itype = complex_input ? CUDA_C_32F : CUDA_R_32F;
      otype = complex_output ? CUDA_C_32F : CUDA_R_32F;
      exec_type = CUDA_C_32F;
    } else if (precison == DataType::FLOAT64) {
      itype = complex_input ? CUDA_C_64F : CUDA_R_64F;
      otype = complex_output ? CUDA_C_64F : CUDA_R_64F;
      exec_type = CUDA_C_64F;
    } else {
      PADDLE_THROW(phi::errors::InvalidArgument(
          "Only transforms of type float32 and float64 are supported."));
    }

    // disable auto allocation of workspace to use allocator from the framework
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cufftSetAutoAllocation(plan(), /* autoAllocate */ 0));

    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::cufftXtMakePlanMany(plan(),
                                          signal_ndim,
                                          signal_sizes.data(),
                                          /* inembed */ nullptr,
                                          /* base_istride */ 1L,
                                          /* idist */ 1L,
                                          itype,
                                          /* onembed */ nullptr,
                                          /* base_ostride */ 1L,
                                          /* odist */ 1L,
                                          otype,
                                          batch_size,
                                          &ws_size_,
                                          exec_type));
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
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::cufftXtExec(
      plan, in_data, out_data, forward ? CUFFT_FORWARD : CUFFT_INVERSE));
}

}  // namespace detail
}  // namespace funcs
}  // namespace phi
