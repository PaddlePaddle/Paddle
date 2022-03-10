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
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"

#include "paddle/fluid/framework/generator.h"

namespace phi {

template <typename T>
struct GPUTruncatedNormal {
  T mean, std;
  T a_normal_cdf;
  T b_normal_cdf;
  unsigned int seed;
  T numeric_min;

  __host__ __device__ GPUTruncatedNormal(T mean, T std, T numeric_min, int seed)
      : mean(mean), std(std), seed(seed), numeric_min(numeric_min) {
    a_normal_cdf = (1.0 + erff(-2.0 / sqrtf(2.0))) / 2.0;
    b_normal_cdf = (1.0 + erff(2.0 / sqrtf(2.0))) / 2.0;
  }

  __host__ __device__ T operator()(const unsigned int n) const {
    thrust::minstd_rand rng;
    rng.seed(seed);
    thrust::uniform_real_distribution<T> dist(numeric_min, 1);
    rng.discard(n);
    T value = dist(rng);
    auto p = a_normal_cdf + (b_normal_cdf - a_normal_cdf) * value;
    return std::sqrt(2.0) * erfinvf(2 * p - 1) * std + mean;
  }
};

template <typename T>
struct TruncatedNormalOffset {
  T mean, std;
  T a_normal_cdf;
  T b_normal_cdf;
  unsigned int seed;
  T numeric_min;
  int offset_;

  __host__ __device__
  TruncatedNormalOffset(T mean, T std, T numeric_min, int seed, int offset)
      : mean(mean),
        std(std),
        seed(seed),
        numeric_min(numeric_min),
        offset_(offset) {
    a_normal_cdf = (1.0 + erff(-2.0 / sqrtf(2.0))) / 2.0;
    b_normal_cdf = (1.0 + erff(2.0 / sqrtf(2.0))) / 2.0;
  }

  __host__ __device__ T operator()(const unsigned int n) const {
    thrust::minstd_rand rng;
    rng.seed(seed);
    thrust::uniform_real_distribution<T> dist(numeric_min, 1);
    rng.discard(n + offset_);
    T value = dist(rng);
    auto p = a_normal_cdf + (b_normal_cdf - a_normal_cdf) * value;
    return std::sqrt(2.0) * erfinvf(2 * p - 1) * std + mean;
  }
};

template <typename T, typename Context>
void TruncatedGaussianRandomKernel(const Context& dev_ctx,
                                   const std::vector<int>& shape,
                                   float mean,
                                   float std,
                                   int seed,
                                   DataType dtype,
                                   DenseTensor* out) {
  auto tensor = out;

  T* data = dev_ctx.template Alloc<T>(tensor);

  bool seed_flag = false;
  if (seed == 0) {
    std::random_device rd;
    seed = rd();
    seed_flag = true;
  }

  thrust::counting_iterator<int64_t> index_sequence_begin(0);
  int64_t size = tensor->numel();

  int device_id = dev_ctx.GetPlace().GetDeviceId();
  auto gen_cuda = paddle::framework::GetDefaultCUDAGenerator(device_id);

  if (gen_cuda->GetIsInitPy() && seed_flag) {
    auto seed_offset = gen_cuda->IncrementOffset(1);
    int64_t gen_offset = size * seed_offset.second;
    thrust::transform(index_sequence_begin,
                      index_sequence_begin + size,
                      thrust::device_ptr<T>(data),
                      TruncatedNormalOffset<T>(mean,
                                               std,
                                               std::numeric_limits<T>::min(),
                                               seed_offset.first,
                                               gen_offset));
  } else {
    thrust::transform(
        index_sequence_begin,
        index_sequence_begin + size,
        thrust::device_ptr<T>(data),
        GPUTruncatedNormal<T>(mean, std, std::numeric_limits<T>::min(), seed));
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(truncated_gaussian_random,
                   GPU,
                   ALL_LAYOUT,
                   phi::TruncatedGaussianRandomKernel,
                   float) {}
