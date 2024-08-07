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

#include "paddle/phi/kernels/funcs/dropout_impl_util.h"
#include "paddle/phi/kernels/funcs/functors.h"
#include "paddle/phi/kernels/primitive/compute_primitives.h"

#include "paddle/phi/kernels/funcs/dropout_impl.cu.h"

namespace phi {
namespace fusion {

template <typename T1, typename T2 = T1, typename OutT = T1>
struct NoMaskFwFunctor {
  const float retain_prob_;
  const bool is_upscale_in_train_;
  using MT = typename phi::dtype::MPTypeTrait<T1>::Type;
  MT factor;
  HOSTDEVICE inline NoMaskFwFunctor(const float retain_prob,
                                    const bool is_upscale_in_train)
      : retain_prob_(retain_prob), is_upscale_in_train_(is_upscale_in_train) {
    factor = static_cast<MT>(1.0f / retain_prob_);
  }

  HOSTDEVICE inline void operator()(OutT* dst,
                                    const T1* src_val,
                                    const T2* rand,
                                    int num) const {
    static constexpr int kCount =
        phi::funcs::uniform_distribution<T2>::kReturnsCount;
#pragma unroll
    for (int i = 0; i < kCount; i++) {
      if (rand[i] < retain_prob_) {
        dst[i] = is_upscale_in_train_
                     ? static_cast<T1>(static_cast<MT>(src_val[i]) * factor)
                     : static_cast<T1>(src_val[i]);
        dst[i] += src_val[i + kCount];
      } else {
        dst[i] = src_val[i + kCount];
      }
    }
  }
};

template <typename T>
struct ScaleAddFuctor {
  using MT = typename phi::dtype::MPTypeTrait<T>::Type;
  explicit ScaleAddFuctor(const MT factor, bool upscale_in_train)
      : factor_(factor), upscale_in_train_(upscale_in_train) {}

  __device__ __forceinline__ T operator()(const T src, const T res) const {
    return upscale_in_train_
               ? src + res
               : static_cast<T>(static_cast<MT>(src) * factor_) + res;
  }

 private:
  MT factor_;
  bool upscale_in_train_;
};

template <typename T, typename Functor>
__global__ void VectorizedDropoutForward(
    /* This is used to relate kernel to cudaGraph nodes*/
    unsigned int identifier,
    const size_t n,
    uint64_t seed,
    const T* src,
    const T* res,
    T* dst,
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
  T dst_res[kCount * 2];
  float rands[kCount];

  using Rand = phi::funcs::uniform_distribution<float>;
  int deal_size = BLOCK_NUM_X * kCount;
  size_t fix = idx * kCount;

  for (; fix < main_offset; fix += stride) {
    kps::ReadData<T, kCount, 1, false>(&dst_res[0], src + fix, deal_size);
    kps::ReadData<T, kCount, 1, false>(&dst_res[kCount], res + fix, deal_size);

    kps::ElementwiseRandom<SType, float, kCount, Rand>(
        &rands[0], Rand(), &state);

    // dst
    kps::OperatorTernary<T, float, T, Functor>(
        &dst_res[0], &dst_res[0], &rands[0], functor, kCount);

    kps::WriteData<T, kCount, 1, false>(dst + fix, &dst_res[0], deal_size);

    if (fix > idx * kCount + 1) {
      __syncthreads();
    }
  }

  int remainder = n - fix;
  if (remainder > 0) {
    kps::ReadData<T, kCount, 1, true>(&dst_res[0], src + fix, remainder);
    kps::ReadData<T, kCount, 1, true>(&dst_res[kCount], res + fix, remainder);
    kps::ElementwiseRandom<SType, float, kCount, Rand>(
        &rands[0], Rand(), &state);
    // dst
    kps::OperatorTernary<T, float, T, Functor>(
        &dst_res[0], &dst_res[0], &rands[0], functor, kCount);
    kps::WriteData<T, kCount, 1, true>(dst + fix, &dst_res[0], remainder);
    __syncthreads();
  }
}

template <typename T, typename Context>
void FusedDropoutAddKernel(const Context& dev_ctx,
                           const DenseTensor& x,
                           const DenseTensor& y,
                           const paddle::optional<DenseTensor>& seed_tensor,
                           const Scalar& p,
                           bool is_test,
                           const std::string& mode,
                           int seed,
                           bool fix_seed,
                           DenseTensor* out,
                           DenseTensor* seed_offset) {
  auto* out_data = dev_ctx.template Alloc<T>(out);
  auto* seed_offset_data = dev_ctx.template HostAlloc<int64_t>(seed_offset);
  int64_t numel = x.numel();
  auto stream = dev_ctx.stream();
  bool upscale_in_train = (mode == "upscale_in_train");

  const auto* x_data = x.data<T>();
  const auto* y_data = y.data<T>();
  float dropout_rate = p.to<float>();

  if (!is_test) {
    if (dropout_rate == 1.0f) {
      phi::Copy(dev_ctx, y, dev_ctx.GetPlace(), false, out);
      return;
    }
    uint64_t seed_data;
    uint64_t increment;
    auto random_prop = GetRandomCudaProp(numel, dev_ctx);
    size_t grid_size = random_prop[0];
    size_t block_size = random_prop[1];
    size_t offset = random_prop[2];
    size_t main_offset = random_prop[3];
    auto seed_tensor_ptr = seed_tensor.get_ptr();
    funcs::GetSeedDataAndIncrement(dev_ctx,
                                   seed_tensor_ptr,
                                   fix_seed,
                                   seed,
                                   offset,
                                   &seed_data,
                                   &increment);
    seed_offset_data[0] = static_cast<int64_t>(seed_data);
    seed_offset_data[1] = static_cast<int64_t>(increment);

    auto dst_functor =
        NoMaskFwFunctor<T, float>(1.0f - dropout_rate, upscale_in_train);

    // we assume seed/offset is same across iterations
    // seed_offset_data should preserved by cudaGraph pool
    const phi::GPUContext* dev_ctx_p = &dev_ctx;

    // seed_offset_data should preserved by cudaGraph pool
    auto gen_cuda = dev_ctx.GetGenerator();
    auto state_index = gen_cuda->GetStateIndex();
    auto parameterSetter =
        [dev_ctx_p,
         offset,
         seed_offset_data,
         state_index,
         seed_tensor_ptr,
         fix_seed](phi::backends::gpu::gpuKernelParams& params) {
          if (!fix_seed) {
            auto gen_cuda = dev_ctx_p->GetGenerator();
            // ensure the generator use correct state index
            gen_cuda->SetStateIndex(state_index);

            // we assume seed is null pointer
            // seed copy to cpu is meaningless here
            assert(seed_tensor_ptr == nullptr);

            uint64_t seed, increment;
            std::tie(seed, increment) = gen_cuda->IncrementOffset(offset);
            VLOG(10) << "CUDA_GRAPH seed = " << seed
                     << ", increment = " << increment;

            params.As<uint64_t>(2) = seed;
            params.As<uint64_t>(6) = increment;

            seed_offset_data[0] = static_cast<int64_t>(seed);
            seed_offset_data[1] = static_cast<int64_t>(increment);
          }
        };
    phi::backends::gpu::CUDAGraphNodeLauncher::gpuKernelCallback_t
        cudaKernelCallback = [=](unsigned int id) {
          void* functionPtr = reinterpret_cast<void*>(
              &(VectorizedDropoutForward<T, NoMaskFwFunctor<T, float>>));
#ifdef PADDLE_WITH_HIP
          hipFunction_t cudaFunc = reinterpret_cast<hipFunction_t>(functionPtr);
#else
          cudaFunction_t cudaFunc;
          PADDLE_ENFORCE_GPU_SUCCESS(
              cudaGetFuncBySymbol(&cudaFunc, functionPtr));
#endif
          VLOG(10) << "[cudaKernelCallback] cudaFunc = " << cudaFunc
                   << " functionPtr = " << functionPtr;

          VectorizedDropoutForward<T, NoMaskFwFunctor<T, float>>
              <<<grid_size, block_size, 0, stream>>>(id,
                                                     numel,
                                                     seed_data,  // need save
                                                     x_data,
                                                     y_data,
                                                     out_data,
                                                     increment,  // need save
                                                     main_offset,
                                                     dst_functor);
          return cudaFunc;
        };
    phi::backends::gpu::CUDAGraphNodeLauncher::Instance().KernelNodeLaunch(
        parameterSetter, cudaKernelCallback);

    VLOG(10) << "NON_CUDA_GRAPH seed = " << seed_data
             << ", increment = " << increment;
  } else {
    using MT = typename phi::dtype::MPTypeTrait<T>::Type;
    MT factor = static_cast<MT>(1.0f - dropout_rate);
    std::vector<phi::DenseTensor*> outs = {out};
    std::vector<const phi::DenseTensor*> ins = {&x, &y};

    phi::funcs::ElementwiseKernel<T>(
        dev_ctx, ins, &outs, ScaleAddFuctor<T>(factor, upscale_in_train));
  }
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_dropout_add,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedDropoutAddKernel,
                   float,
                   double,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {
  kernel->OutputAt(1).SetDataType(phi::DataType::INT64);
}
