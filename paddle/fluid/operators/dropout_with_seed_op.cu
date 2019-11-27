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

#include <glog/logging.h>

#include <cuda.h>
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <string>
#include "paddle/fluid/operators/dropout_with_seed_op.h"
#include "paddle/fluid/platform/dynload/curand.h"
#include "paddle/fluid/platform/float16.h"
namespace paddle {
namespace operators {

template <typename T, typename MaskType>
__global__ void RandomGenerator(const size_t n, const int* seed_ptr,
                                const float dropout_prob, const T* src,
                                MaskType* mask_data, T* dst,
                                bool is_upscale_in_train) {
  // VLOG(3) << "random generator";
  curandStatePhilox4_32_10_t state;
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int step_size = 0;

  int seed = *seed_ptr;
  MaskType mask;
  T dest;
  for (; idx < n; idx += blockDim.x * gridDim.x) {
    T s = src[idx];
    if (step_size == 0) {
      curand_init(seed, idx, idx, &state);
      step_size = blockDim.x * gridDim.x;
    } else {
      curand_init(seed, idx, step_size, &state);
    }
    if (curand_uniform(&state) < dropout_prob) {
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
  // VLOG(3) << "random generator finished";
}

// It seems that Eigen::Tensor::setRandom in GPU will SEGFAULT.
// Use std::random and thrust::random(thrust is a std library in CUDA) to
// implement uniform random.
template <typename Place, typename T>
class GPUDropoutWithSeedKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    VLOG(3) << "Compute!!!";
    VLOG(3) << "try get x";
    auto* x = context.Input<Tensor>("X");
    VLOG(3) << "try get seed";
    auto* seed = context.Input<Tensor>("Seed");
    VLOG(3) << "try get out";
    auto* y = context.Output<Tensor>("Out");
    VLOG(3) << "Get X, seed out";
    y->mutable_data<T>(context.GetPlace());
    float dropout_prob = context.Attr<float>("dropout_prob");

    auto& dropout_implementation =
        context.Attr<std::string>("dropout_implementation");
    bool upscale_in_train = (dropout_implementation == "upscale_in_train");
    VLOG(3) << "Get attr";
    auto& place = *context.template device_context<Place>().eigen_device();
    VLOG(3) << "Get place";
    if (!context.Attr<bool>("is_test")) {
      VLOG(3) << "is not test";
      int64_t x_numel = x->numel();
      auto stream = context.cuda_device_context().stream();

      auto* mask = context.Output<Tensor>("Mask");
      auto* mask_data = mask->mutable_data<uint8_t>(context.GetPlace());
      VLOG(3) << "mask data";
      size_t size = framework::product(mask->dims());
      auto* x_data = x->data<T>();
      VLOG(3) << "x data";
      VLOG(3) << "try to print x";
      VLOG(3) << "X data: " << *x;
      VLOG(3) << "Seed data";
      const auto* seed_data = seed->data<int>();
      auto* y_data = y->mutable_data<T>(context.GetPlace());
      VLOG(3) << "try to print seed";
      VLOG(3) << "Seed data: " << *seed;
      VLOG(3) << "y data";
      if (dropout_prob == 1.0f) {
        PADDLE_ENFORCE_CUDA_SUCCESS(
            cudaMemsetAsync(y_data, 0, x_numel * sizeof(T), stream));
        PADDLE_ENFORCE_CUDA_SUCCESS(cudaMemsetAsync(
            mask_data, 0, x_numel * sizeof(*mask_data), stream));
        return;
      }

      // std::random_device rnd;
      // int seed =
      //    context.Attr<bool>("fix_seed") ? context.Attr<int>("seed") : rnd();

      int threads = 512;
      int grid = (x_numel + threads - 1) / threads;
      VLOG(3) << "start random genrator";
      RandomGenerator<T, uint8_t><<<grid, threads, 0, stream>>>(
          size, seed_data, dropout_prob, x_data, mask_data, y_data,
          upscale_in_train);
      VLOG(3) << "End random genrator";
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
    dropout_with_seed,
    ops::GPUDropoutWithSeedKernel<plat::CUDADeviceContext, float>,
    ops::GPUDropoutWithSeedKernel<plat::CUDADeviceContext, plat::float16>,
    ops::GPUDropoutWithSeedKernel<plat::CUDADeviceContext, double>);
REGISTER_OP_CUDA_KERNEL(
    dropout_with_seed_grad,
    ops::DropoutWithSeedGradKernel<plat::CUDADeviceContext, float>,
    ops::DropoutWithSeedGradKernel<plat::CUDADeviceContext, plat::float16>,
    ops::DropoutWithSeedGradKernel<plat::CUDADeviceContext, double>);
