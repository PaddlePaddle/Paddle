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

#ifdef PADDLE_WITH_CUDA
#include <cuda.h>
#include <curand_kernel.h>
#include "paddle/fluid/platform/dynload/curand.h"
#endif
#ifdef PADDLE_WITH_HIP
#include <hip/hip_runtime.h>
#include <hiprand_kernel.h>
#include "paddle/fluid/platform/dynload/hiprand.h"
#endif
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <algorithm>
#include <string>
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/operators/dropout_op.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

template <typename T, typename MaskType>
__global__ void RandomGenerator(const size_t n, uint64_t seed,
                                const float dropout_prob, const T* src,
                                MaskType* mask, T* dst,
                                bool is_upscale_in_train, uint64_t increment) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
#ifdef PADDLE_WITH_HIP
  hiprandStatePhilox4_32_10_t state;
  hiprand_init(seed, idx, increment, &state);
#else
  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx, increment, &state);
#endif

  MaskType mask_val;
  T dst_val;
  T factor = static_cast<T>(1.0f / (1.0f - dropout_prob));
  for (; idx < n; idx += blockDim.x * gridDim.x) {
    T src_val = src[idx];
#ifdef PADDLE_WITH_HIP
    if (hiprand_uniform(&state) < dropout_prob) {
#else
    if (curand_uniform(&state) < dropout_prob) {
#endif
      mask_val = 0;
      dst_val = 0;
    } else {
      mask_val = 1;
      dst_val = is_upscale_in_train ? src_val * factor : src_val;
    }
    mask[idx] = mask_val;
    dst[idx] = dst_val;
  }
}

template <typename T, typename MaskType, int VecSize>
__global__ void VectorizedRandomGenerator(const size_t n, uint64_t seed,
                                          const float dropout_prob,
                                          const T* src, MaskType* mask, T* dst,
                                          bool is_upscale_in_train,
                                          uint64_t increment) {
  using LoadT = platform::AlignedVector<T, VecSize>;
  using MaskLoadT = platform::AlignedVector<MaskType, VecSize>;

#ifdef PADDLE_WITH_HIP
  int64_t idx = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  hiprandStatePhilox4_32_10_t state;
  hiprand_init(seed, idx, increment, &state);
#else
  int64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx, increment, &state);
#endif

  T factor = static_cast<T>(1.0f / (1.0f - dropout_prob));
  for (int i = idx * VecSize; i < n; i += blockDim.x * gridDim.x * VecSize) {
    LoadT src_val;
    platform::Load<T, VecSize>(&src[i], &src_val);

#ifdef PADDLE_WITH_HIP
    float4 rand = hiprand_uniform4(&state);
#else
    float4 rand = curand_uniform4(&state);
#endif

    LoadT dst_val;
    MaskLoadT mask_val;

#pragma unroll
    for (int j = 0; j < VecSize; j++) {
      if ((&rand.x)[j] < dropout_prob) {
        dst_val[j] = 0;
        mask_val[j] = 0;
      } else {
        dst_val[j] = is_upscale_in_train ? src_val[j] * factor : src_val[j];
        mask_val[j] = 1;
      }
    }

    platform::Store<T, VecSize>(dst_val, &dst[i]);
    platform::Store<MaskType, VecSize>(mask_val, &mask[i]);
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
    auto* seed =
        context.HasInput("Seed") ? context.Input<Tensor>("Seed") : nullptr;
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
#ifdef PADDLE_WITH_HIP
        PADDLE_ENFORCE_CUDA_SUCCESS(
            hipMemsetAsync(y_data, 0, x_numel * sizeof(T), stream));
        PADDLE_ENFORCE_CUDA_SUCCESS(
            hipMemsetAsync(mask_data, 0, x_numel * sizeof(*mask_data), stream));
#else
        PADDLE_ENFORCE_CUDA_SUCCESS(
            cudaMemsetAsync(y_data, 0, x_numel * sizeof(T), stream));
        PADDLE_ENFORCE_CUDA_SUCCESS(cudaMemsetAsync(
            mask_data, 0, x_numel * sizeof(*mask_data), stream));
#endif
        return;
      }

      const auto& dev_ctx = context.cuda_device_context();
      platform::GpuLaunchConfig config =
          platform::GetGpuLaunchConfig1D(dev_ctx, size);

      // increment is used to set the args(offset) of curand_init, which defines
      // offset in subsequence.
      // The detail:
      // https://docs.nvidia.com/cuda/curand/device-api-overview.html
      // Increment should be at least the number of curand() random numbers used
      // in each thread to avoid the random number generated this time being the
      // same as the previous calls.
      uint64_t seed_data;
      uint64_t increment;
      int vec_size = platform::GetVectorizedSize<T>(x_data);
      auto offset = ((x_numel - 1) / (config.block_per_grid.x *
                                      config.thread_per_block.x * vec_size) +
                     1) *
                    vec_size;
      int device_id = BOOST_GET_CONST(platform::CUDAPlace, context.GetPlace())
                          .GetDeviceId();
      auto gen_cuda = framework::GetDefaultCUDAGenerator(device_id);

      if (seed && platform::is_gpu_place(seed->place())) {
        framework::Tensor seed_cpu_tensor;
        TensorCopySync(*seed, platform::CPUPlace(), &seed_cpu_tensor);
        seed_data = static_cast<uint64_t>(seed_cpu_tensor.data<int>()[0]);
        increment = offset;
      } else if (gen_cuda->GetIsInitPy() && (!context.Attr<bool>("fix_seed"))) {
        auto seed_offset = gen_cuda->IncrementOffset(offset);
        seed_data = seed_offset.first;
        increment = seed_offset.second;
      } else {
        if (seed) {
          seed_data = *(seed->data<int>());
        } else {
          std::random_device rnd;
          seed_data = context.Attr<bool>("fix_seed") ? context.Attr<int>("seed")
                                                     : rnd();
        }
        increment = offset;
      }

#ifdef __HIPCC__
      if (vec_size == 4 && size % 4 == 0) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(VectorizedRandomGenerator<T, uint8_t, 4>),
            config.block_per_grid, config.thread_per_block, 0, stream, size,
            seed_data, dropout_prob, x_data, mask_data, y_data,
            upscale_in_train, increment);
      } else {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(RandomGenerator<T, uint8_t>),
                           config.block_per_grid, config.thread_per_block, 0,
                           stream, size, seed_data, dropout_prob, x_data,
                           mask_data, y_data, upscale_in_train, increment);
      }
#else
      if (vec_size == 4 && size % 4 == 0) {
        VectorizedRandomGenerator<
            T, uint8_t,
            4><<<config.block_per_grid, config.thread_per_block, 0, stream>>>(
            size, seed_data, dropout_prob, x_data, mask_data, y_data,
            upscale_in_train, increment);
      } else {
        RandomGenerator<T, uint8_t><<<config.block_per_grid,
                                      config.thread_per_block, 0, stream>>>(
            size, seed_data, dropout_prob, x_data, mask_data, y_data,
            upscale_in_train, increment);
      }
#endif
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
