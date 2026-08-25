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

#include "paddle/phi/kernels/truncated_gaussian_random_kernel.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/transform.h>

#include <limits>

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/distribution_helper.h"

namespace phi {

template <typename T, typename MT>
struct GPUTruncatedNormal {
  MT mean, std, a, b;
  MT a_normal_cdf;
  MT b_normal_cdf;
  unsigned int seed;
  MT numeric_min;

  __host__ __device__
  GPUTruncatedNormal(MT mean, MT std, MT numeric_min, int seed, MT a, MT b)
      : mean(mean), std(std), seed(seed), numeric_min(numeric_min), a(a), b(b) {
    a_normal_cdf = (1.0 + erff((a - mean) / std / sqrtf(2.0))) / 2.0;
    b_normal_cdf = (1.0 + erff((b - mean) / std / sqrtf(2.0))) / 2.0;
  }

  __host__ __device__ T operator()(const unsigned int n) const {
    thrust::minstd_rand rng;
    rng.seed(seed);
    thrust::uniform_real_distribution<MT> dist(numeric_min, 1);
    rng.discard(n);
    MT value = dist(rng);
    auto p = a_normal_cdf + (b_normal_cdf - a_normal_cdf) * value;
    MT ret = std::sqrt(2.0) * erfinvf(2 * p - 1) * std + mean;
    return static_cast<T>(std::clamp(ret, a, b));
  }
};

template <typename T, typename MT>
struct TruncatedNormalTransform {
  MT mean, std, a, b;
  MT a_normal_cdf;
  MT b_normal_cdf;

  HOSTDEVICE TruncatedNormalTransform(MT mean, MT std, MT a, MT b)
      : mean(mean), std(std), a(a), b(b) {
    a_normal_cdf = (1.0 + erff((a - mean) / std / sqrtf(2.0))) / 2.0;
    b_normal_cdf = (1.0 + erff((b - mean) / std / sqrtf(2.0))) / 2.0;
  }

  // Inverts the CDF of the truncated distribution, mapping a uniform sample in
  // (0, 1] onto [a, b].
  HOSTDEVICE inline T operator()(MT value) const {
    auto p = a_normal_cdf + (b_normal_cdf - a_normal_cdf) * value;
    MT ret = std::sqrt(2.0) * erfinvf(2 * p - 1) * std + mean;
    return static_cast<T>(std::clamp(ret, a, b));
  }
};

template <typename T, typename Context>
void TruncatedGaussianRandomKernel(const Context& dev_ctx,
                                   const std::vector<int>& shape,
                                   float mean,
                                   float std,
                                   int seed,
                                   float a,
                                   float b,
                                   DataType dtype,
                                   DenseTensor* out) {
  T* data = dev_ctx.template Alloc<T>(out);

  using MT = typename MPTypeTrait<T>::Type;

  if (seed == 0) {
    funcs::uniform_distribution<MT> dist;
    TruncatedNormalTransform<T, MT> trans(static_cast<MT>(mean),
                                          static_cast<MT>(std),
                                          static_cast<MT>(a),
                                          static_cast<MT>(b));
    funcs::distribution_and_transform<T>(dev_ctx, out, dist, trans);
  } else {
    // use OP seed
    thrust::counting_iterator<int64_t> index_sequence_begin(0);
    int64_t size = out->numel();
    thrust::transform(
        index_sequence_begin,
        index_sequence_begin + size,
        thrust::device_ptr<T>(data),
        GPUTruncatedNormal<T, MT>(
            mean, std, std::numeric_limits<MT>::min(), seed, a, b));
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(truncated_gaussian_random,
                   GPU,
                   ALL_LAYOUT,
                   phi::TruncatedGaussianRandomKernel,
                   float,
                   double,
                   phi::bfloat16) {}
