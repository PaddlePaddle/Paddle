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
#include "paddle/fluid/operators/dropout_op.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

inline __host__ __device__ size_t HashCombine(size_t seed, size_t idx) {
  // use boost::hash_combine to make seed more random
  seed ^= idx + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}

template <typename T, typename MaskType>
__global__ void RandomGenerator(const size_t n, const size_t seed,
                                const uint32_t int_dropout_prob, const T* src,
                                MaskType* mask_data, T* dst) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  thrust::minstd_rand rng;
  rng.seed(HashCombine(seed, idx));
  constexpr uint32_t kUInt32Max = static_cast<uint32_t>(-1);
  thrust::uniform_int_distribution<uint32_t> dist(0, kUInt32Max - 1);

  if (idx < n) {
    if (dist(rng) < int_dropout_prob) {
      mask_data[idx] = static_cast<MaskType>(0);
      dst[idx] = static_cast<T>(0);
    } else {
      mask_data[idx] = static_cast<MaskType>(1);
      dst[idx] = src[idx];
    }
  }
}

// It seems that Eigen::Tensor::setRandom in GPU will SEGFAULT.
// Use std::random and thrust::random(thrust is a std library in CUDA) to
// implement uniform random.
template <typename Place, typename T>
class GPUDropoutKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<Tensor>("X");
    auto* y = context.Output<Tensor>("Out");
    y->mutable_data<T>(context.GetPlace());
    float dropout_prob = context.Attr<float>("dropout_prob");

    auto& place = *context.template device_context<Place>().eigen_device();
    if (!context.Attr<bool>("is_test")) {
      auto* mask = context.Output<Tensor>("Mask");
      auto* mask_data = mask->mutable_data<uint8_t>(context.GetPlace());
      size_t size = framework::product(mask->dims());
      auto* x_data = x->data<T>();
      auto* y_data = y->mutable_data<T>(context.GetPlace());

      std::random_device rnd;
      int seed =
          context.Attr<bool>("fix_seed") ? context.Attr<int>("seed") : rnd();

      size_t threads = 512;
      size_t grid = (size + threads - 1) / threads;

      // cast dropout_prob to double type to prevent overflow when dropout_prob
      // is near 1
      uint32_t int_dropout_prob = static_cast<uint32_t>(
          static_cast<double>(dropout_prob) * static_cast<uint32_t>(-1));
      RandomGenerator<<<grid, threads, 0,
                        context.cuda_device_context().stream()>>>(
          size, seed, int_dropout_prob, x_data, mask_data, y_data);
    } else {
      auto X = EigenMatrix<T>::Reshape(*x, 1);
      auto Y = EigenMatrix<T>::Reshape(*y, 1);
      Y.device(place) = X * static_cast<T>(1.0f - dropout_prob);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(
    dropout, ops::GPUDropoutKernel<plat::CUDADeviceContext, float>,
    ops::GPUDropoutKernel<plat::CUDADeviceContext, plat::float16>);
REGISTER_OP_CUDA_KERNEL(dropout_grad,
                        ops::DropoutGradKernel<plat::CUDADeviceContext, float>);
