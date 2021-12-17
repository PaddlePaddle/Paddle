/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include <curand_kernel.h>
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/for_range.h"
#include "paddle/fluid/platform/hostdevice.h"

namespace paddle {
namespace distribution {

using Tensor = framework::Tensor;

template <typename T>
struct exponential_transform {
  explicit exponential_transform(T lambda) : lambda_(lambda) {}

  HOSTDEVICE inline T operator()(T val) const {
#if (defined(__NVCC__) || defined(__HIPCC__))
    if (UNLIKELY(val >= static_cast<T>(1.) -
                            (std::numeric_limits<T>::epsilon() / 2))) {
      return static_cast<T>(-std::numeric_limits<T>::epsilon() / 2);
    }
    if (std::is_same<T, double>::value) {
      return static_cast<T>(-1.0) / lambda_ * log(val);
    } else {
      return static_cast<T>(-1.0) / lambda_ * __logf(val);
    }
#else
    return static_cast<T>(-1.0) / lambda_ * std::log(static_cast<T>(1.0) - val);
#endif
  }

 private:
  T lambda_;
};

#if defined(__NVCC__) || defined(__HIPCC__)

template <typename T>
struct uniform_distribution;

template <>
struct uniform_distribution<float> {
  __device__ inline float4 operator()(curandStatePhilox4_32_10_t *state) const {
    return curand_uniform4(state);
  }
};

template <>
struct uniform_distribution<double> {
  __device__ inline double2 operator()(
      curandStatePhilox4_32_10_t *state) const {
    return curand_uniform2_double(state);
  }
};

template <typename T>
struct normal_distribution;

template <>
struct normal_distribution<float> {
  __device__ inline float4 operator()(curandStatePhilox4_32_10_t *state) const {
    return curand_normal4(state);
  }
};

template <>
struct normal_distribution<double> {
  __device__ inline double2 operator()(
      curandStatePhilox4_32_10_t *state) const {
    return curand_normal2_double(state);
  }
};

template <typename T, typename DistOp, typename TransformOp>
void distribution_and_transform(const platform::CUDADeviceContext &dev_ctx,
                                Tensor *out, DistOp dist, TransformOp trans) {
  T *out_data = out->mutable_data<T>(dev_ctx.GetPlace());
  auto size = out->numel();

  int64_t device_id =
      BOOST_GET_CONST(platform::CUDAPlace, dev_ctx.GetPlace()).GetDeviceId();
  auto gen_cuda = framework::GetDefaultCUDAGenerator(device_id);
  auto seed_offset = gen_cuda->IncrementOffset(4);
  uint64_t seed = seed_offset.first;
  uint64_t offset = seed_offset.second;

  platform::ForRange<platform::CUDADeviceContext> for_range(dev_ctx, size);
  for_range([=] __device__(size_t idx) {
    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, offset, &state);

    auto random_tuple = dist(&state);
    auto random = static_cast<T>((&random_tuple.x)[0]);
    out_data[idx] = trans(random);
  });
}

#endif

}  // namespace distribution
}  // namespace paddle
