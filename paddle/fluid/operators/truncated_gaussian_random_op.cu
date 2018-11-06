/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <thrust/random.h>
#include <thrust/transform.h>
#include <limits>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace operators {

template <typename T>
struct TruncatedNormal {
  T mean, std;
  T a_normal_cdf;
  T b_normal_cdf;
  unsigned int seed;
  T numeric_min;

  __host__ __device__ TruncatedNormal(T mean, T std, T numeric_min, int seed)
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
class GPUTruncatedGaussianRandomKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* tensor = context.Output<framework::Tensor>("Out");
    T* data = tensor->mutable_data<T>(context.GetPlace());

    unsigned int seed = static_cast<unsigned int>(context.Attr<int>("seed"));
    if (seed == 0) {
      std::random_device rd;
      seed = rd();
    }
    T mean = static_cast<T>(context.Attr<float>("mean"));
    T std = static_cast<T>(context.Attr<float>("std"));
    thrust::counting_iterator<unsigned int> index_sequence_begin(0);
    int64_t size = tensor->numel();
    thrust::transform(
        index_sequence_begin, index_sequence_begin + size,
        thrust::device_ptr<T>(data),
        TruncatedNormal<T>(mean, std, std::numeric_limits<T>::min(), seed));
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(
    truncated_gaussian_random,
    paddle::operators::GPUTruncatedGaussianRandomKernel<float>);
