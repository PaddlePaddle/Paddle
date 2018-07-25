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

__device__ inline int hash_combine(int seed, int idx) {
  // boost::hash_combine() to make seed more random
  seed ^= idx + 0x9e3779b9 + ((seed) << 6) + ((seed) >> 2);
  return seed;
}

template <typename T, typename MaskType>
__global__ void DropoutForward(size_t n, int seed, uint32_t threshold,
                               const T* src, MaskType* mask_data, T* dst) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= n) return;

  thrust::minstd_rand rng;
  seed = hash_combine(seed, idx);
  rng.seed(seed);
  thrust::uniform_int_distribution<uint32_t> dist(
      0, static_cast<uint32_t>(-1) - 1);

  if (dist(rng) < threshold) {
    mask_data[idx] = static_cast<MaskType>(0);
    dst[idx] = static_cast<T>(0);
  } else {
    mask_data[idx] = static_cast<MaskType>(1);
    dst[idx] = src[idx];
  }
}

template <typename T, typename MaskType>
__global__ void DropoutBackward(size_t n, const T* dy,
                                const MaskType* mask_data, T* dx) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= n) return;
  dx[idx] =
      mask_data[idx] > static_cast<MaskType>(0) ? dy[idx] : static_cast<T>(0);
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
    float dropout_prob = context.Attr<float>("dropout_prob");

    auto& place = *context.template device_context<Place>().eigen_device();
    if (!context.Attr<bool>("is_test")) {
      auto* mask = context.Output<Tensor>("Mask");
      auto* mask_data = mask->mutable_data<T>(context.GetPlace());
      size_t size = x->numel();
      auto* x_data = x->data<T>();
      auto* y_data = y->mutable_data<T>(context.GetPlace());

      std::random_device rnd;
      int seed =
          context.Attr<bool>("fix_seed") ? context.Attr<int>("seed") : rnd();

      uint32_t threshold = static_cast<uint32_t>(
          static_cast<uint32_t>(-1) * static_cast<double>(dropout_prob));

      size_t threads = 512;
      size_t grids = (size + threads - 1) / threads;
      DropoutForward<
          T><<<grids, threads, 0, context.cuda_device_context().stream()>>>(
          size, seed, threshold, x_data, mask_data, y_data);
    } else {
      y->mutable_data<T>(context.GetPlace());
      auto X = EigenMatrix<T>::Reshape(*x, 1);
      auto Y = EigenMatrix<T>::Reshape(*y, 1);
      Y.device(place) = X * static_cast<T>(1.0f - dropout_prob);
    }
  }
};

template <typename T>
class GPUDropoutGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE(!context.Attr<bool>("is_test"),
                   "GradOp is only callable when is_test is false");

    auto* grad_x = context.Output<Tensor>(framework::GradVarName("X"));
    auto* grad_y = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* mask = context.Input<Tensor>("Mask");

    auto* grad_x_data = grad_x->mutable_data<T>(context.GetPlace());
    auto* grad_y_data = grad_y->data<T>();
    auto* mask_data = mask->data<T>();

    size_t size = grad_y->numel();
    size_t threads = 512;
    size_t grids = (size + threads - 1) / threads;
    DropoutBackward<
        T><<<grids, threads, 0, context.cuda_device_context().stream()>>>(
        size, grad_y_data, mask_data, grad_x_data);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(
    dropout, ops::GPUDropoutKernel<plat::CUDADeviceContext, float>,
    ops::GPUDropoutKernel<plat::CUDADeviceContext, plat::float16>);
REGISTER_OP_CUDA_KERNEL(
    dropout_grad, ops::GPUDropoutGradKernel<float>
    /*ops::DropoutGradKernel<plat::CUDADeviceContext, float>*/);
