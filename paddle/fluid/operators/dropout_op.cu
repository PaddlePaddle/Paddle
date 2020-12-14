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
#include <cuda.h>
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <algorithm>
#include <string>
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/operators/dropout_op.h"
#include "paddle/fluid/platform/dynload/curand.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

// aligned vector generates vectorized load/store on CUDA
template <typename T, int Size>
struct alignas(sizeof(T) * Size) AlignedVector {
  T val[Size];
};

template <typename T>
inline int VectorizedSize(const T* pointer) {
  uint64_t address = reinterpret_cast<uint64_t>(pointer);
  constexpr int vec4 = std::alignment_of<AlignedVector<T, 4>>::value;  // NOLINT
  if (address % vec4 == 0) {
    return 4;
  }
  return 1;
}

template <typename T, typename MaskType>
__global__ void RandomGenerator(const size_t n, uint64_t seed,
                                const float dropout_prob, const T* src,
                                MaskType* mask_data, T* dst,
                                bool is_upscale_in_train, uint64_t increment) {
  curandStatePhilox4_32_10_t state;
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  curand_init(seed, idx, increment, &state);

  MaskType mask;
  T dest;
  for (; idx < n; idx += blockDim.x * gridDim.x) {
    T s = src[idx];
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
}

template <typename T, typename MaskType, int VecSize>
__global__ void VectorizedRandomGenerator(const size_t n, uint64_t seed,
                                          const float dropout_prob,
                                          const T* src, MaskType* mask_data,
                                          T* dst, bool is_upscale_in_train,
                                          uint64_t increment) {
  int64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx, increment, &state);

  MaskType mask;
  T dest;
  using LoadT = AlignedVector<T, VecSize>;
  using MaskLoadT = AlignedVector<MaskType, VecSize>;
  T factor = static_cast<T>(1.0f / (1.0f - dropout_prob));
  for (int i = idx * VecSize; i < n; i += blockDim.x * gridDim.x * VecSize) {
    T src_vec[VecSize];
    LoadT* value = reinterpret_cast<LoadT*>(&src_vec);
    *value = *reinterpret_cast<const LoadT*>(&src[i]);
    float4 rand = curand_uniform4(&state);

    T dest_vec[VecSize];
    MaskType mask_vec[VecSize];

#pragma unroll
    for (int ii = 0; ii < VecSize; ii++) {
      if ((&rand.x)[ii] < dropout_prob) {
        dest_vec[ii] = 0;
        mask_vec[ii] = 0;
      } else {
        if (is_upscale_in_train) {
          dest_vec[ii] = src_vec[ii] * factor;
        } else {
          dest_vec[ii] = src_vec[ii];
        }
        mask_vec[ii] = 1;
      }
    }

    *(reinterpret_cast<LoadT*>(&dst[i])) =
        *reinterpret_cast<LoadT*>(&dest_vec[0]);
    *(reinterpret_cast<MaskLoadT*>(&mask_data[i])) =
        *reinterpret_cast<MaskLoadT*>(&mask_vec[0]);
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
        PADDLE_ENFORCE_CUDA_SUCCESS(
            cudaMemsetAsync(y_data, 0, x_numel * sizeof(T), stream));
        PADDLE_ENFORCE_CUDA_SUCCESS(cudaMemsetAsync(
            mask_data, 0, x_numel * sizeof(*mask_data), stream));
        return;
      }

      int threads = 512;
      int grid = (x_numel + threads - 1) / threads;
      const auto& dev_ctx = context.cuda_device_context();
      int blocks_per_sm =
          dev_ctx.GetMaxPhysicalThreadCount() / dev_ctx.GetSMCount() / threads;
      grid = std::min(dev_ctx.GetSMCount() * blocks_per_sm, grid);

      // increment is used to set the args(offset) of curand_init, which defines
      // offset in subsequence.
      // The detail:
      // https://docs.nvidia.com/cuda/curand/device-api-overview.html
      // Increment should be at least the number of curand() random numbers used
      // in each thread to avoid the random number generated this time being the
      // same as the previous calls.
      uint64_t seed_data;
      uint64_t increment;
      int vec_size = VectorizedSize<T>(x_data);
      auto offset =
          ((x_numel - 1) / (threads * grid * vec_size) + 1) * vec_size;
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

      if (vec_size == 4) {
        VectorizedRandomGenerator<T, uint8_t, 4><<<grid, threads, 0, stream>>>(
            size, seed_data, dropout_prob, x_data, mask_data, y_data,
            upscale_in_train, increment);
      } else {
        RandomGenerator<T, uint8_t><<<grid, threads, 0, stream>>>(
            size, seed_data, dropout_prob, x_data, mask_data, y_data,
            upscale_in_train, increment);
      }

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
