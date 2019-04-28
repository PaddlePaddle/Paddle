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
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <string>
#include "paddle/fluid/operators/dropout_op.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

template <typename T, typename MaskType>
__global__ void RandomGenerator(const size_t n, const int seed,
                                const float dropout_prob, const T* src,
                                MaskType* mask_data, T* dst,
                                bool is_upscale_in_train) {
  thrust::minstd_rand rng;
  rng.seed(seed);
  thrust::uniform_real_distribution<float> dist(0, 1);

  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int step_size = 0;

  MaskType mask;
  T dest;
  for (; idx < n; idx += blockDim.x * gridDim.x) {
    T s = src[idx];
    if (step_size == 0) {
      rng.discard(idx);
      step_size = blockDim.x * gridDim.x;
    } else {
      rng.discard(step_size);
    }
    if (dist(rng) < dropout_prob) {
      mask = 0;
      dest = 0;
    } else {
      mask = 1;
      if (is_upscale_in_train) {
        dest = s / static_cast<T>(1.0f - dropout_prob);
      } else {
        dest = s;
      }
    }
    mask_data[idx] = mask;
    dst[idx] = dest;
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

    auto& dropout_implementation =
        context.Attr<std::string>("dropout_implementation");
    bool upscale_in_train = (dropout_implementation == "upscale_in_train");

    auto& place = *context.template device_context<Place>().eigen_device();
    if (!context.Attr<bool>("is_test")) {
      int64_t x_numel = x->numel();
      auto stream = context.cuda_device_context().stream();

      auto* mask = context.Output<Tensor>("Mask");
      auto* mask_data = mask->mutable_data<uint8_t>(context.GetPlace());
      size_t size = framework::product(mask->dims());
      auto* x_data = x->data<T>();
      auto* y_data = y->mutable_data<T>(context.GetPlace());
      if (dropout_prob == 1.0f) {
        PADDLE_ENFORCE(cudaMemsetAsync(y_data, 0, x_numel * sizeof(T), stream));
        PADDLE_ENFORCE(cudaMemsetAsync(mask_data, 0,
                                       x_numel * sizeof(*mask_data), stream));
        return;
      }

      std::random_device rnd;
      int seed =
          context.Attr<bool>("fix_seed") ? context.Attr<int>("seed") : rnd();

      int threads = 512;
      int grid = (x_numel + threads - 1) / threads;
      RandomGenerator<T, uint8_t><<<grid, threads, 0, stream>>>(
          size, seed, dropout_prob, x_data, mask_data, y_data,
          upscale_in_train);
    } else {
      auto X = EigenMatrix<T>::Reshape(*x, 1);
      auto Y = EigenMatrix<T>::Reshape(*y, 1);
      if (upscale_in_train) {
        Y.device(place) = X;
      } else {
        Y.device(place) = X * static_cast<T>(1.0f - dropout_prob);
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(
    dropout, ops::GPUDropoutKernel<plat::CUDADeviceContext, float>,
    ops::GPUDropoutKernel<plat::CUDADeviceContext, plat::float16>,
    ops::GPUDropoutKernel<plat::CUDADeviceContext, double>);
REGISTER_OP_CUDA_KERNEL(
    dropout_grad, ops::DropoutGradKernel<plat::CUDADeviceContext, float>,
    ops::DropoutGradKernel<plat::CUDADeviceContext, plat::float16>,
    ops::DropoutGradKernel<plat::CUDADeviceContext, double>);
