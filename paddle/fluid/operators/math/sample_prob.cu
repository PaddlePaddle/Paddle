/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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
#include <iostream>
#include <unordered_set>
#include <vector>
#include <thrust/random.h>

#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/math/sampler.h"
#include "paddle/fluid/operators/math/sample_prob.h"
namespace paddle {
namespace operators {
namespace math {

using Tensor = framework::Tensor;

template <typename T>
__device__ T gpu_adjust_prob(const T prob, const int num_samples, const int num_tries) {
  if (num_samples == num_tries) {
    return prob * num_samples;
  } else {
    return -expm1(num_tries * log1p(-prob));
  }
}

class GPULogUniformSampler{
 public:
  __device__ int64_t Sample(float random, const int range, const float log_range) const;
  __device__ float Probability(int64_t value, const float log_range) const;
};

__device__ int64_t GPULogUniformSampler::Sample(float random, const int range, const float log_range) const {
  // Got Log Uniform distribution from uniform distribution by
  // inverse_transform_sampling method
  const int64_t value =
      static_cast<int64_t>(exp(random * log_range)) - 1;
  // Mathematically, value should be <= range_, but might not be due to some
  // floating point roundoff, so we mod by range_.
  return value % range;
}

__device__ float GPULogUniformSampler::Probability(int64_t value, const float log_range) const {
  // Given f(x) = 1/[(x+1) * log_range_]
  // The value's  probability  is integral of f(x) from value to (value + 1)
  return (log((value + 2.0) / (value + 1.0))) / log_range;
}


template<typename T>
__global__ void SamplingCondidate(const size_t n, const int seed, const int range, 
    const float log_range, const int num_true, const std::size_t num_samples, 
    const int64_t* label_data, int64_t* samples_data, T* probabilities_data) {
  thrust::minstd_rand rng;
  rng.seed(seed);
  thrust::uniform_real_distribution<float> dist(0, 1);
  const int num_sampled_classes = num_true + num_samples;

  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int step_size = 0;

  for (; idx < n; idx += blockDim.x * gridDim.x) {
    if (step_size == 0) {
      rng.discard(idx);
      step_size = blockDim.x * gridDim.x;
    } else {
      rng.discard(step_size);
    }
    int col_cnt = idx % num_sampled_classes;
    GPULogUniformSampler sampler;
    if (col_cnt < num_true) {
      samples_data[idx] = label_data[idx];
    } else {
      samples_data[idx] = sampler.Sample(dist(rng), range, log_range);
    }
    probabilities_data[idx] = sampler.Probability(samples_data[idx], range);
    probabilities_data[idx] = gpu_adjust_prob(
        probabilities_data[idx], num_samples, num_sampled_classes);
  } 
}

template <typename T>
void GPUSampleWithProb<T>::operator()(const platform::CUDADeviceContext& context, const int seed, const int dict_size,
                  const std::size_t num_samples, const Tensor* L, Tensor* S,
                  Tensor* P) {
    // UNDERSTAND: dimension issues
    const auto lbl_dim = L->dims();
    const int batch_size = lbl_dim[0];
    const int num_true = lbl_dim[1];
    const int num_sampled_classes = num_true + num_samples;
    framework::DDim ret_dim{batch_size, num_sampled_classes};

    // UNDERSTAND: raw data view
    const int64_t* label_data = L->data<int64_t>();
    int64_t* samples_data =
        S->mutable_data<int64_t>(ret_dim, context.GetPlace());
    T* probabilities_data = P->mutable_data<T>(ret_dim, context.GetPlace());
    int threads = 512;
    const size_t size = batch_size * num_sampled_classes; 
    int grid = (batch_size * num_sampled_classes + threads - 1) / threads;
    SamplingCondidate<T><<<grid, threads, 0, context.stream()>>>(size, seed, dict_size, log(dict_size), num_true, num_samples, label_data, samples_data, probabilities_data); 
};

template class GPUSampleWithProb<float>;
template class GPUSampleWithProb<double>;
}  // namespace math
}  // namespace operators
}  // namespace paddle
