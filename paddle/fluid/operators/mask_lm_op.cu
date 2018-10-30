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

#define EIGEN_USE_GPU
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include "paddle/fluid/operators/mask_lm_op.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

template <typename T>
__global__ void RandomGenerator(const size_t n, const int seed, const int voc_size, 
                                const float masked_prob, const T mask_id,
                                const T* src, 
                                T* mask_data, T* dst) {
  thrust::minstd_rand rng;
  rng.seed(seed);
  thrust::uniform_real_distribution<float> dist(0, 1);

  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int step_size = 0;
  
  float fake_masked_prob = masked_prob * 0.1;
  float rand_masked_prob = masked_prob * 0.2;
  float rand_scale = rand_masked_prob - fake_masked_prob;

  T dest;
  for (; idx < n; idx += blockDim.x * gridDim.x) {
    T s = src[idx];
    if (step_size == 0) {
      rng.discard(idx);
      step_size = blockDim.x * gridDim.x;
    } else {
      rng.discard(step_size);
    }
    
    if (dist(rng) < fake_masked_prob) {
        mask_data[idx] = s;
        dst[idx] = s;
        continue;
    }
    if (dist(rng) < rand_masked_prob) {
        mask_data[idx] = s;
        dest = static_cast<T>(floor(
                (dist(rng) - fake_masked_prob) / rand_scale * voc_size));
        dst[idx] = dest;
        continue;
    }
    if (dist(rng) < masked_prob) {
        mask_data[idx] = s;
        dst[idx] = mask_id;
        continue;
    }
    //else
    mask_data[idx] = static_cast<T>(-1);
    dst[idx] = s;
  }
}

// It seems that Eigen::Tensor::setRandom in GPU will SEGFAULT.
// Use std::random and thrust::random(thrust is a std library in CUDA) to
// implement uniform random.
template <typename Place, typename T>
class GPUMaskLMKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<Tensor>("X");
    auto* x_data = x->data<T>();
    auto* y = context.Output<Tensor>("Out");
    auto* y_data = y->mutable_data<T>(context.GetPlace());
    auto* mask = context.Output<Tensor>("Mask");
    auto* mask_data = mask->mutable_data<T>(context.GetPlace());
    
    T mask_id = static_cast<T>(context.Attr<int>("mask_id"));
    float masked_prob = context.Attr<float>("masked_prob");
    int voc_size = context.Attr<int>("voc_size");

    std::random_device rnd;
    int seed =
        context.Attr<bool>("fix_seed") ? context.Attr<int>("seed") : rnd();

    size_t size = framework::product(mask->dims());
    int threads = 512;
    int grid = (x->numel() + threads - 1) / threads;
    RandomGenerator<
        T><<<grid, threads, 0, context.cuda_device_context().stream()>>>(
        size, seed, voc_size, masked_prob, mask_id, x_data, mask_data, y_data);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(
    mask_lm, ops::GPUMaskLMKernel<plat::CUDADeviceContext, float>,
    ops::GPUMaskLMKernel<plat::CUDADeviceContext, plat::float16>,
    ops::GPUMaskLMKernel<plat::CUDADeviceContext, double>,
    ops::GPUMaskLMKernel<plat::CUDADeviceContext, uint8_t>,
    ops::GPUMaskLMKernel<plat::CUDADeviceContext, int>,
    ops::GPUMaskLMKernel<plat::CUDADeviceContext, int64_t>);
