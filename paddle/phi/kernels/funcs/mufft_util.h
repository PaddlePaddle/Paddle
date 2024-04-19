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

#include "paddle/phi/backends/dynload/mufft.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/kernels/funcs/fft.h"
#include "paddle/phi/kernels/funcs/fft_key.h"

namespace phi {
namespace funcs {
namespace detail {

// An RAII encapsulation of muFFTHandle
class MUFFTHandle {
 public:
  MUFFTHandle() {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mufftCreate(&handle_));
  }

  MUFFTHandle(const MUFFTHandle& other) = delete;
  MUFFTHandle& operator=(const MUFFTHandle& other) = delete;

  MUFFTHandle(MUFFTHandle&& other) = delete;
  MUFFTHandle& operator=(MUFFTHandle&& other) = delete;

  ::mufftHandle& get() { return handle_; }
  const ::mufftHandle& get() const { return handle_; }

  ~MUFFTHandle() { phi::dynload::mufftDestroy(handle_); }

 private:
  ::mufftHandle handle_;
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

    mufftType exec_type = [&]() {
      if (precision == DataType::FLOAT32) {
        switch (fft_type) {
          case FFTTransformType::C2C:
            return MUFFT_C2C;
          case FFTTransformType::R2C:
            return MUFFT_R2C;
          case FFTTransformType::C2R:
            return MUFFT_C2R;
        }
      } else if (precision == DataType::FLOAT64) {
        switch (fft_type) {
          case FFTTransformType::C2C:
            return MUFFT_Z2Z;
          case FFTTransformType::R2C:
            return MUFFT_D2Z;
          case FFTTransformType::C2R:
            return MUFFT_Z2D;
        }
      }
      PADDLE_THROW(phi::errors::InvalidArgument(
          "Only transforms of type float32 and float64 are supported."));
    }();

    // disable auto allocation of workspace to use allocator from the framework
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::mufftSetAutoAllocation(plan(), /* autoAllocate */ 0));
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::mufftMakePlanMany(plan(),
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

  const mufftHandle& plan() const { return plan_.get(); }
  FFTTransformType transform_type() const { return fft_type_; }
  DataType data_type() const { return precision_; }
  size_t workspace_size() const { return ws_size_; }

 private:
  MUFFTHandle plan_;
  size_t ws_size_;  // workspace size in bytes
  FFTTransformType fft_type_;
  DataType precision_;
};

static void exec_plan(const FFTConfig& config,
                      void* in_data,
                      void* out_data,
                      bool forward) {
  auto& plan = config.plan();
  PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mufftXtExec(
      plan, in_data, out_data, forward ? MUFFT_FORWARD : MUFFT_INVERSE));
}

}  // namespace detail
}  // namespace funcs
}  // namespace phi
