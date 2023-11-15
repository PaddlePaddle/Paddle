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

#include "paddle/phi/kernels/impl/dirichlet_kernel_impl.h"

#ifdef PADDLE_WITH_CUDA
#include <curand_kernel.h>
#endif
#ifdef PADDLE_WITH_HIP
#include <hiprand_kernel.h>
#endif

#if defined(PADDLE_WITH_CUDA)
using COMPAT_RANDSTATEPHILOX4_32_10_T = curandStatePhilox4_32_10_t;
#define COMPAT_RAND_INIT curand_init
#define COMPAT_RAND_UNIFORM curand_uniform
#define COMPAT_RAND_NORMAL curand_normal
#elif defined(PADDLE_WITH_HIP)
using COMPAT_RANDSTATEPHILOX4_32_10_T = hiprandStatePhilox4_32_10_t;
#define COMPAT_RAND_INIT hiprand_init
#define COMPAT_RAND_UNIFORM hiprand_uniform
#define COMPAT_RAND_NORMAL hiprand_normal
#endif

namespace phi {
template <typename T>
struct GammaCUDAFunctor {
  GammaCUDAFunctor(const T* alpha, T* gamma, uint64_t seed, uint64_t offset)
      : alpha_(alpha), gamma_(gamma), seed_(seed), offset_(offset) {}

  DEVICE void operator()(int64_t index) {
    // curand initialization
    COMPAT_RANDSTATEPHILOX4_32_10_T state;
    COMPAT_RAND_INIT(
        /*seed=*/seed_, /*subsequence=*/index, /*offset=*/offset_, &state);

    // sample
    auto uniform_lambda = [&state]() { return COMPAT_RAND_UNIFORM(&state); };
    BaseSampler<T, decltype(uniform_lambda)> standard_uniform(uniform_lambda);
    auto normal_lambda = [&state]() { return COMPAT_RAND_NORMAL(&state); };
    BaseSampler<T, decltype(normal_lambda)> standard_normal(normal_lambda);

    auto sample =
        sample_gamma<T, T, decltype(uniform_lambda), decltype(normal_lambda)>(
            alpha_[index], standard_uniform, standard_normal);
    gamma_[index] = std::max(std::numeric_limits<T>::min(), sample);
  }

  const T* alpha_;
  T* gamma_;
  const uint64_t seed_;
  const uint64_t offset_;
};

}  // namespace phi
