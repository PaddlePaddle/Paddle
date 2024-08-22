// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_device_function.h"
#include "paddle/phi/backends/gpu/gpu_dnn.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/load_store_util.h"
#include "paddle/phi/kernels/gpu/gelu_funcs.h"
// for windows build
#define M_SQRT1_2 0.70710678118654752440

namespace phi {
namespace fusion {

template <typename T>
struct FastGeluFunctor {
  inline __device__ T operator()(const T x) const {
    return phi::GeluFwd<T, true>(x);
  }
};

template <typename T>
struct GeluComputeType;

template <>
struct GeluComputeType<phi::dtype::bfloat16> {
  using Type = float;
};

template <>
struct GeluComputeType<phi::dtype::float16> {
  using Type = float;
};

template <>
struct GeluComputeType<float> {
  using Type = float;
};

template <typename T>
using GeluType = typename GeluComputeType<T>::Type;

using phi::funcs::DequantLoad;
using phi::funcs::Load;
using phi::funcs::QuantStore;
using phi::funcs::Store;

template <typename T>
struct BaseActivationFunctor {
  using ELEMENT_TYPE = T;

  using AttrPair = std::vector<std::pair<const char *, float *>>;

  AttrPair GetAttrs() { return AttrPair(); }
};

// For windows build
template <typename T>
struct CudaSwishFunctor : public BaseActivationFunctor<T> {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;
  MPType one = static_cast<MPType>(1.0f);
  float beta = 1.0;

  typename BaseActivationFunctor<T>::AttrPair GetAttrs() {
    return {{"beta", &beta}};
  }

  // swish(x) = x / (1 + exp(-beta * x))
  __device__ __forceinline__ T operator()(const T arg_x) const {
    MPType x = static_cast<MPType>(arg_x);
    MPType b = static_cast<MPType>(beta);
    return static_cast<T>(x / (one + exp(-b * x)));
  }
};

// TODO(lzc): transfer to phi::funcs
template <typename T>
struct GeluFunctor {
  inline __host__ __device__ T operator()(const T x) const {
    using U = GeluType<T>;
    const U casted_x = static_cast<U>(x);
    const U temp = erf(casted_x * static_cast<U>(M_SQRT1_2));
    const U out = (casted_x * static_cast<U>(0.5) * (static_cast<U>(1) + temp));
    return static_cast<T>(out);
  }
};

template <typename T>
struct ReluFunctor {
  inline __host__ __device__ T operator()(const T x) const {
    T zero = static_cast<T>(0.0);
    return x > zero ? x : zero;
  }
};

inline gpuError_t GetNumBlocks(int64_t n, int *num_blocks) {
  constexpr int kBlockSize = 128;
  constexpr int kNumWaves = 16;

  const int device_id = phi::backends::gpu::GetCurrentDeviceId();
  const int sm_count = phi::backends::gpu::GetGPUMultiProcessors(device_id);
  const int max_thread_per_multiprocessor =
      phi::backends::gpu::GetGPUMaxThreadsPerMultiProcessor(device_id);

  *num_blocks =
      std::max<int>(1,
                    std::min<int64_t>((n + kBlockSize - 1) / kBlockSize,
                                      sm_count * max_thread_per_multiprocessor /
                                          kBlockSize * kNumWaves));
  return gpuSuccess;
}

}  // namespace fusion
}  // namespace phi
