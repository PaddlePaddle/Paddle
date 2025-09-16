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

// #The file has been adapted from pytorch project
// #Licensed under   BSD-style license -
// https://github.com/pytorch/pytorch/blob/main/LICENSE

#pragma once

#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Float8_e4m3fn.h>
// #include <c10/util/Float8_e4m3fnuz.h>
#include <c10/util/Float8_e5m2.h>
// #include <c10/util/Float8_e5m2fnuz.h>
#include <c10/util/Half.h>

#if defined(__CUDACC__)
#include <cuda.h>
#include <cuda_fp16.h>
#elif defined(__HIPCC__)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

namespace at {

template <typename T, c10::DeviceType D>
struct AccumulateTypeDevice {};

template <typename T, bool>
struct AccumulateType {};

template <typename T>
struct AccumulateType<T, false> {
  using type = typename AccumulateTypeDevice<T, c10::DeviceType::CPU>::type;
};

template <typename T>
struct AccumulateType<T, true> {
  using type = typename AccumulateTypeDevice<T, c10::DeviceType::CUDA>::type;
};

template <typename T, c10::DeviceType device>
using acc_type_device = typename AccumulateTypeDevice<T, device>::type;

template <typename T, bool is_cuda>
using acc_type = typename AccumulateType<T, is_cuda>::type;

#define ACC_TYPE(t, acc_t, device_type)         \
  template <>                                   \
  struct AccumulateTypeDevice<t, device_type> { \
    using type = acc_t;                         \
  };

#define CUDA_ACC_TYPE(t, acc_t) ACC_TYPE(t, acc_t, c10::DeviceType::CUDA)
#define CPU_ACC_TYPE(t, acc_t) ACC_TYPE(t, acc_t, c10::DeviceType::CPU)

#if defined(__CUDACC__) || defined(__HIPCC__)
CUDA_ACC_TYPE(half, float)
#endif
CUDA_ACC_TYPE(BFloat16, float)
CUDA_ACC_TYPE(Half, float)
CUDA_ACC_TYPE(Float8_e5m2, float)
CUDA_ACC_TYPE(Float8_e4m3fn, float)
// CUDA_ACC_TYPE(Float8_e5m2fnuz, float)
// CUDA_ACC_TYPE(Float8_e4m3fnuz, float)
CUDA_ACC_TYPE(float, float)
CUDA_ACC_TYPE(double, double)
CUDA_ACC_TYPE(int8_t, int64_t)
CUDA_ACC_TYPE(uint8_t, int64_t)
CUDA_ACC_TYPE(char, int64_t)
CUDA_ACC_TYPE(int16_t, int64_t)
CUDA_ACC_TYPE(int32_t, int64_t)
CUDA_ACC_TYPE(int64_t, int64_t)
CUDA_ACC_TYPE(bool, bool)
CUDA_ACC_TYPE(c10::complex<Half>, c10::complex<float>)
CUDA_ACC_TYPE(c10::complex<float>, c10::complex<float>)
CUDA_ACC_TYPE(c10::complex<double>, c10::complex<double>)

CPU_ACC_TYPE(BFloat16, float)
CPU_ACC_TYPE(Half, float)
CPU_ACC_TYPE(Float8_e5m2, float)
CPU_ACC_TYPE(Float8_e4m3fn, float)
// CPU_ACC_TYPE(Float8_e5m2fnuz, float)
// CPU_ACC_TYPE(Float8_e4m3fnuz, float)
CPU_ACC_TYPE(float, double)
CPU_ACC_TYPE(double, double)
CPU_ACC_TYPE(int8_t, int64_t)
CPU_ACC_TYPE(uint8_t, int64_t)
CPU_ACC_TYPE(char, int64_t)
CPU_ACC_TYPE(int16_t, int64_t)
CPU_ACC_TYPE(int32_t, int64_t)
CPU_ACC_TYPE(int64_t, int64_t)
CPU_ACC_TYPE(bool, bool)
CPU_ACC_TYPE(c10::complex<Half>, c10::complex<float>)
CPU_ACC_TYPE(c10::complex<float>, c10::complex<double>)
CPU_ACC_TYPE(c10::complex<double>, c10::complex<double>)

c10::ScalarType toAccumulateType(c10::ScalarType type, c10::DeviceType device);
c10::ScalarType toAccumulateType(c10::ScalarType type, bool is_cuda);

}  // namespace at
