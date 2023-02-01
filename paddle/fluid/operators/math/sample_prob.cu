/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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
#include <thrust/random.h>
#include <thrust/sort.h>

#include <iostream>
#include <vector>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/math/sample_prob.h"
#include "paddle/fluid/operators/math/sampler.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {
namespace math {

template <typename T>
__device__ T gpu_adjust_prob(const T prob,
                             const int num_samples,
                             const int num_tries) {
  if (num_samples == num_tries) {
    return prob * num_samples;
  } else {
    return -expm1(num_tries * log1p(-prob));
  }
}

class GPULogUniformSampler {
 public:
  __device__ int64_t Sample(float random,
                            const int range,
                            const float log_range) const;
  __device__ float Probability(int64_t value, const float log_range) const;
};

__device__ int64_t GPULogUniformSampler::Sample(float random,
                                                const int range,
                                                const float log_range) const {
  // Got Log Uniform distribution from uniform distribution by
  // inverse_transform_sampling method
  const int64_t value = static_cast<int64_t>(exp(random * log_range)) - 1;
  // Mathematically, value should be <= range_, but might not be due to some
  // floating point roundoff, so we mod by range_.
  return value % range;
}

__device__ float GPULogUniformSampler::Probability(
    int64_t value, const float log_range) const {
  // Given f(x) = 1/[(x+1) * log_range_]
  // The value's  probability  is integral of f(x) from value to (value + 1)
  return (log((value + 2.0) / (value + 1.0))) / log_range;
}

template <typename T>
__global__ void SamplingCondidate(const size_t n,
                                  const int num_tries,
                                  const int range,
                                  const float log_range,
                                  const int num_true,
                                  const std::size_t num_samples,
                                  const int64_t* label_data,
                                  int64_t* samples_data,
                                  T* probabilities_data) {
  const int num_sampled_classes = num_true + num_samples;

  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int step_size = 0;
  GPULogUniformSampler sampler;

  for (; idx < n; idx += blockDim.x * gridDim.x) {
    int col_idx = idx % num_sampled_classes;
    int row_idx = idx / num_sampled_classes;
    if (col_idx < num_true) {
      samples_data[idx] = label_data[row_idx * num_true + col_idx];
    } else {
      samples_data[idx] = samples_data[col_idx];
    }
    probabilities_data[idx] = sampler.Probability(samples_data[idx], log_range);
    probabilities_data[idx] =
        gpu_adjust_prob(probabilities_data[idx], num_samples, num_tries);
  }
}

template <typename T>
int UniqSampler(const Sampler& sampler,
                const std::size_t num_samples,
                int64_t* samples_data) {
  // sample num_samles unique samples for an example, note that they are not
  // all negative samples
  std::unordered_set<int64_t> tmp_samples;
  tmp_samples.clear();
  int num_tries = 0;
  int j = 0;
  while (j < num_samples) {
    ++num_tries;
    auto v = sampler.Sample();
    auto insert_ok = tmp_samples.insert(v).second;
    if (!insert_ok) {
      continue;
    }
    samples_data[j] = v;
    ++j;
  }
  return num_tries;
}

template <typename T>
void GPUSampleWithProb<T>::operator()(const phi::GPUContext& context,
                                      const int seed,
                                      const int dict_size,
                                      const bool uniq,
                                      const std::size_t num_samples,
                                      const phi::DenseTensor* L,
                                      phi::DenseTensor* S,
                                      phi::DenseTensor* P) {
  // UNDERSTAND: dimension issues
  const auto lbl_dim = L->dims();
  const int batch_size = lbl_dim[0];
  const int num_true = lbl_dim[1];
  const int num_sampled_classes = num_true + num_samples;
  framework::DDim ret_dim{batch_size, num_sampled_classes};

  // UNDERSTAND: raw data view
  const int64_t* label_data = L->data<int64_t>();
  int64_t* samples_data = S->data<int64_t>();
  T* probabilities_data = P->data<T>();

  int s_size = num_samples;
  framework::DDim s_dim{s_size};
  phi::DenseTensor s;
  int64_t* s_data = s.mutable_data<int64_t>(s_dim, platform::CPUPlace());

  math::LogUniformSampler sampler(dict_size, seed);

  int range = dict_size;
  float log_range = log(range + 1);

  int num_tries = UniqSampler<T>(sampler, num_samples, s_data);
  VLOG(1) << "num_tries: " << num_tries;

#ifdef PADDLE_WITH_HIP
  PADDLE_ENFORCE_GPU_SUCCESS(hipMemcpy(samples_data + num_true,
                                       s_data,
                                       sizeof(int64_t) * num_samples,
                                       hipMemcpyHostToDevice));
#else
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpy(samples_data + num_true,
                                        s_data,
                                        sizeof(int64_t) * num_samples,
                                        cudaMemcpyHostToDevice));
#endif

  int threads = 512;
  const size_t size = batch_size * num_sampled_classes;
  int grid = (batch_size * num_sampled_classes + threads - 1) / threads;
#ifdef PADDLE_WITH_HIP
  hipLaunchKernelGGL(HIP_KERNEL_NAME(SamplingCondidate<T>),
                     dim3(grid),
                     dim3(threads),
                     0,
                     context.stream(),
                     size,
                     num_tries,
                     range,
                     log_range,
                     num_true,
                     num_samples,
                     label_data,
                     samples_data,
                     probabilities_data);
#else
  SamplingCondidate<T>
      <<<grid, threads, 0, context.stream()>>>(size,
                                               num_tries,
                                               range,
                                               log_range,
                                               num_true,
                                               num_samples,
                                               label_data,
                                               samples_data,
                                               probabilities_data);
#endif
}

template class GPUSampleWithProb<float>;
template class GPUSampleWithProb<double>;
}  // namespace math
}  // namespace operators
}  // namespace paddle
