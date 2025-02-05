// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/fusion/gpu/fused_dropout_add_utils.h"

#include "paddle/phi/kernels/funcs/distribution_helper.h"
#include "paddle/phi/kernels/funcs/dropout_impl_util.h"
#include "paddle/phi/kernels/funcs/functors.h"
#include "paddle/phi/kernels/primitive/compute_primitives.h"

#include "paddle/phi/kernels/funcs/dropout_impl.cu.h"

static constexpr int kNumCUDAThreads = 512;
static constexpr int kNumMaximumNumBlocks = 4096;

static inline int NumBlocks(const int N) {
  return std::min((N + kNumCUDAThreads - 1) / kNumCUDAThreads,
                  kNumMaximumNumBlocks);
}

namespace phi {
namespace fusion {

template <typename T, typename MT>
__global__ void FuseScaleAddGrad(const T* grad,
                                 T* x,
                                 T* y,
                                 const MT factor,
                                 const int64_t limit,
                                 bool upscale_in_train) {
  CUDA_KERNEL_LOOP(i, limit) {
    y[i] = grad[i];
    x[i] = upscale_in_train ? grad[i]
                            : static_cast<T>(static_cast<MT>(grad[i]) * factor);
  }
}

template <typename T>
__global__ void FuseScaleAddGradRateZero(const T* grad,
                                         T* src,
                                         T* res,
                                         const int64_t limit) {
  CUDA_KERNEL_LOOP(i, limit) {
    res[i] = grad[i];
    src[i] = 0;
  }
}

template <typename T1, typename T2 = T1, typename OutT = T1>
struct NoMaskBwFunctor {
  const float retain_prob_;
  using MT = typename phi::dtype::MPTypeTrait<T1>::Type;
  MT factor_;
  HOSTDEVICE inline NoMaskBwFunctor(const float retain_prob)
      : retain_prob_(retain_prob) {
    factor_ = static_cast<MT>(1.0f / retain_prob_);
  }

  HOSTDEVICE inline NoMaskBwFunctor(const float retain_prob, const MT factor)
      : retain_prob_(retain_prob), factor_(factor) {}

  HOSTDEVICE inline void operator()(OutT* dst,
                                    const T1* src_val,
                                    const T2* rand,
                                    int num) const {
    static constexpr int kCount =
        phi::funcs::uniform_distribution<T2>::kReturnsCount;
#pragma unroll
    for (int i = 0; i < kCount; i++) {
      dst[i + kCount] = src_val[i];
      dst[i] = rand[i] < retain_prob_
                   ? static_cast<T1>(static_cast<MT>(src_val[i]) * factor_)
                   : static_cast<T1>(0);
    }
  }
};

template <typename T, typename Functor>
__global__ void VectorizedDropoutBackward(
    /* This is used to relate kernel to cudaGraph nodes*/
    unsigned int identifier,
    const size_t n,
    uint64_t seed,
    T* x,
    T* y,
    const T* out_grad,
    uint64_t increment,
    size_t main_offset,
    Functor functor) {
  size_t idx = static_cast<size_t>(BLOCK_ID_X * BLOCK_NUM_X);
  static constexpr int kCount =
      phi::funcs::uniform_distribution<float>::kReturnsCount;
  size_t stride = BLOCK_NUM_X * GRID_NUM_X * kCount;
#ifdef PADDLE_WITH_HIP
  hiprandStatePhilox4_32_10_t state;
  hiprand_init(seed, idx + THREAD_ID_X, increment, &state);
  using SType = hiprandStatePhilox4_32_10_t;
#else
  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx + THREAD_ID_X, increment, &state);
  using SType = curandStatePhilox4_32_10_t;
#endif

  float rands[kCount];
  T x_y[kCount * 2];

  using Rand = phi::funcs::uniform_distribution<float>;
  using Cast = kps::IdentityFunctor<T>;

  int deal_size = BLOCK_NUM_X * kCount;
  size_t fix = idx * kCount;

  for (; fix < main_offset; fix += stride) {
    kps::ReadData<T, kCount, 1, false>(&x_y[0], out_grad + fix, deal_size);
    kps::ElementwiseRandom<SType, float, kCount, Rand>(
        &rands[0], Rand(), &state);
    kps::OperatorTernary<T, float, T, Functor>(
        &x_y[0], &x_y[0], &rands[0], functor, kCount);

    kps::WriteData<T, kCount, 1, false>(x + fix, &x_y[0], deal_size);
    kps::WriteData<T, kCount, 1, false>(y + fix, &x_y[kCount], deal_size);
    if (fix > idx * kCount + 1) {
      __syncthreads();
    }
  }

  int remainder = n - fix;
  if (remainder > 0) {
    kps::ReadData<T, kCount, 1, true>(&x_y[0], out_grad + fix, remainder);
    kps::ElementwiseRandom<SType, float, kCount, Rand>(
        &rands[0], Rand(), &state);
    kps::OperatorTernary<T, float, T, Functor>(
        &x_y[0], &x_y[0], &rands[0], functor, kCount);

    kps::WriteData<T, kCount, 1, true>(x + fix, &x_y[0], remainder);
    kps::WriteData<T, kCount, 1, true>(y + fix, &x_y[kCount], remainder);
    __syncthreads();
  }
}

template <typename T, typename Context>
void FusedDropoutAddGradKernel(const Context& dev_ctx,
                               const DenseTensor& seed_offset,
                               const DenseTensor& out_grad,
                               const Scalar& p,
                               bool is_test,
                               const std::string& mode,
                               bool fix_seed,
                               DenseTensor* x_grad,
                               DenseTensor* y_grad) {
  int64_t numel = out_grad.numel();
  auto stream = dev_ctx.stream();
  float dropout_rate = p.to<float>();
  bool upscale_in_train = (mode == "upscale_in_train");

  const auto* seed_offset_data = seed_offset.data<int64_t>();
  const uint64_t seed_data = static_cast<uint64_t>(seed_offset_data[0]);
  const uint64_t increment = static_cast<uint64_t>(seed_offset_data[1]);

  auto* x_grad_data = dev_ctx.template Alloc<T>(x_grad);
  auto* y_grad_data = dev_ctx.template Alloc<T>(y_grad);

  const auto* out_grad_data = out_grad.data<T>();
  using MT = typename phi::dtype::MPTypeTrait<T>::Type;
  int blocks = NumBlocks(numel);
  int threads = kNumCUDAThreads;

  if (is_test) {
    MT factor = static_cast<MT>(1.0f - dropout_rate);
    FuseScaleAddGrad<T, MT><<<blocks, threads, 0, stream>>>(out_grad_data,
                                                            x_grad_data,
                                                            y_grad_data,
                                                            factor,
                                                            numel,
                                                            upscale_in_train);

  } else {
    if (upscale_in_train && dropout_rate == 1.0f) {
      FuseScaleAddGradRateZero<T><<<blocks, threads, 0, stream>>>(
          out_grad_data, x_grad_data, y_grad_data, numel);
      return;
    }
    auto random_prop = GetRandomCudaProp(numel, dev_ctx);
    size_t grid_size = random_prop[0];
    size_t block_size = random_prop[1];
    size_t offset = random_prop[2];
    size_t main_offset = random_prop[3];
    auto functor = upscale_in_train
                       ? NoMaskBwFunctor<T, float>(1.0f - dropout_rate)
                       : NoMaskBwFunctor<T, float>(1.0f - dropout_rate, 1.0f);

    // we assume seed/offset is same across iterations
    // seed_offset_data should preserved by cudaGraph pool
    const phi::GPUContext* dev_ctx_p = &dev_ctx;
    auto parameterSetter = [offset, dev_ctx_p, seed_offset](
                               phi::backends::gpu::gpuKernelParams& params) {
      const auto* seed_offset_data = seed_offset.data<int64_t>();
      const uint64_t seed_data = static_cast<uint64_t>(seed_offset_data[0]);
      const uint64_t increment = static_cast<uint64_t>(seed_offset_data[1]);

      params.As<uint64_t>(2) = seed_data;
      params.As<uint64_t>(6) = increment;
      VLOG(10) << "CUDA_GRAPH seed = " << seed_data
               << ", increment = " << increment;
    };

    phi::backends::gpu::CUDAGraphNodeLauncher::gpuKernelCallback_t
        cudaKernelCallback = [=](unsigned int id) {
          void* functionPtr = reinterpret_cast<void*>(
              &(VectorizedDropoutBackward<T, NoMaskBwFunctor<T, float>>));
#ifdef PADDLE_WITH_HIP
          hipFunction_t cudaFunc = reinterpret_cast<hipFunction_t>(functionPtr);
#else
          cudaFunction_t cudaFunc;
          PADDLE_ENFORCE_GPU_SUCCESS(
              cudaGetFuncBySymbol(&cudaFunc, functionPtr));
#endif
          VLOG(10) << "[cudaKernelCallback] cudaFunc = " << cudaFunc
                   << " functionPtr = " << functionPtr;

          VectorizedDropoutBackward<T, NoMaskBwFunctor<T, float>>
              <<<grid_size, block_size, 0, stream>>>(
                  id,
                  numel,
                  seed_data,  //  idx: 2 need save
                  x_grad_data,
                  y_grad_data,
                  out_grad_data,
                  increment,  //  idx: 6 need save
                  main_offset,
                  functor);
          return cudaFunc;
        };
    phi::backends::gpu::CUDAGraphNodeLauncher::Instance().KernelNodeLaunch(
        parameterSetter, cudaKernelCallback);

    VLOG(10) << "NON_CUDA_GRAPH seed = " << seed_data
             << ", increment = " << increment;
  }
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_dropout_add_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedDropoutAddGradKernel,
                   float,
                   double,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {
  kernel->InputAt(0).SetBackend(phi::Backend::CPU);  // seed_offset
}
