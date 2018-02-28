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

namespace paddle {
namespace operators {

template <typename T, typename AttrType>
__global__ void RandomGenerator(const size_t n, const int seed,
                                const AttrType dropout_prob, const T* src,
                                T* mask_data, T* dst) {
  thrust::minstd_rand rng;
  rng.seed(seed);
  thrust::uniform_real_distribution<AttrType> dist(0, 1);

  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (; idx < n; idx += blockDim.x * gridDim.x) {
    if (dist(rng) < dropout_prob) {
      mask_data[idx] = static_cast<T>(0);
    } else {
      mask_data[idx] = static_cast<T>(1);
    }
    dst[idx] = mask_data[idx] * src[idx];
  }
}

// It seems that Eigen::Tensor::setRandom in GPU will SEGFAULT.
// Use std::random and thrust::random(thrust is a std library in CUDA) to
// implement uniform random.
template <typename Place, typename T, typename AttrType>
class GPUDropoutKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<Tensor>("X");
    auto* y = context.Output<Tensor>("Out");
    y->mutable_data<T>(context.GetPlace());
    AttrType dropout_prob = context.Attr<AttrType>("dropout_prob");

    auto X = EigenMatrix<T>::Reshape(*x, 1);
    auto Y = EigenMatrix<T>::Reshape(*y, 1);

    auto& place = *context.template device_context<Place>().eigen_device();
    if (!context.Attr<bool>("is_test")) {
      auto* mask = context.Output<Tensor>("Mask");
      auto* mask_data = mask->mutable_data<T>(context.GetPlace());
      size_t size = framework::product(mask->dims());
      auto* x_data = x->data<T>();
      auto* y_data = y->mutable_data<T>(context.GetPlace());

      std::random_device rnd;
      int seed =
          context.Attr<bool>("fix_seed") ? context.Attr<int>("seed") : rnd();

      int threads = 512;
      int grid = (x->numel() + threads - 1) / threads;
      RandomGenerator<T, AttrType><<<grid, threads, 0,
                                     context.cuda_device_context().stream()>>>(
          size, seed, dropout_prob, x_data, mask_data, y_data);
    } else {
      Y.device(place) = X * (1.0f - dropout_prob);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    dropout,
    ops::GPUDropoutKernel<paddle::platform::CUDADeviceContext, float, float>);
REGISTER_OP_CUDA_KERNEL(
    dropout_grad,
    ops::DropoutGradKernel<paddle::platform::CUDADeviceContext, float>);
