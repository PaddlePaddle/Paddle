// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#ifdef __NVCC__
#include <curand_kernel.h>
#endif
#ifdef __HIPCC__
#include <hiprand_kernel.h>
#endif

#include "paddle/phi/kernels/rrelu_kernel.h"
// #include "paddle/phi/kernels/gpu/rrelu_impl.cu.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/for_range.h"


namespace phi {

// template <typename T, typename Context>
// void RReluKernel(const Context& dev_ctx,
//                  const DenseTensor& x,
//                  const float lower,
//                  const float upper,
//                  bool is_test,
//                //   bool fix_seed,
//                //   int seed,
//                  DenseTensor* out,
//                  DenseTensor* mask) {
//   out->mutable_data<T>(dev_ctx.GetPlace());
//   mask->mutable_data<T>(dev_ctx.GetPlace());

//   paddle::operators::RReluFwGPUKernelDriver<T>(dev_ctx,
//                                                is_test,
//                                                lower,
//                                                upper, 
//                                                fix_seed,
//                                                seed,
//                                                x,
//                                             //    seed_tensor.get_ptr(),
//                                                nullptr,
//                                                mask,
//                                                out);

template <typename T>
struct RReluCudaFunctor {
 public:
  RReluCudaFunctor(const T* in,
                   T* out,
                   T* mask,
                   const float lower,
                   const float upper,
                   unsigned int seed,
                   unsigned int offset)
      : in_(in), out_(out), mask_(mask), lower_(lower), upper_(upper), seed_(seed), offset_(offset) {}

  using MT = typename kps::details::MPTypeTrait<T>::Type;

   auto zero = static_cast<T>(0);
   auto one = static_cast<T>(1);

  __device__ void operator()(int64_t idx) {
   if (in_[idx] < zero) {
#ifdef __NVCC__
      curandStatePhilox4_32_10_t state;
      curand_init(seed_, idx, offset_, &state);
      // random_sampled_value should be in [0, 1]
      float random_sampled_value = static_cast<float>(curand_uniform(&state));
#elif __HIPCC__
      hiprandStatePhilox4_32_10_t state;
      hiprand_init(seed_, idx, offset_, &state);
      // random_sampled_value should be in [0, 1]
      float random_sampled_value = static_cast<float>(hiprand_uniform(&state));
#endif
      random_sampled_value = random_sampled_value * (upper_ - lower_) + lower_;
      mask_[idx] = static_cast<T>(random_sampled_value);
      out_[idx] = static_cast<T>(static_cast<MT>(in_[idx]) * static_cast<MT>(random_sampled_value));
   } else {
      mask_[idx] = one;
      out_[idx] = in_[idx];
   }
  }

 private:
  const T* in_;
  T* out_;
  T* mask_;
  const float lower_;
  const float upper_;
  const unsigned int seed_;
  const unsigned int offset_;
};

template <typename T, typename Context>
void RReluKernel(const Context& ctx, 
                 const DenseTensor& x, 
                 const float lower,
                 const float upper,
                 bool is_test,
                 DenseTensor* out,
                 DenseTensor* mask) {
  const T* x_data = x.data<T>();
  T* out_data = ctx.template Alloc<T>(out);
  T* mask_data = ctx.template Alloc<T>(mask);
  auto size = x.numel();

  auto gen_cuda = ctx.GetGenerator();
  auto seed_offset = gen_cuda->IncrementOffset(20);
  uint64_t seed = seed_offset.first;
  uint64_t offset = seed_offset.second;

  phi::funcs::ForRange<Context> for_range(ctx, size);

  RReluCudaFunctor<T> functor(x_data, out_data, mask_data, lower, upper, seed, offset);
  for_range(functor);
}


}  // namespace phi

PD_REGISTER_KERNEL(rrelu,
                   GPU,
                   ALL_LAYOUT,
                   phi::RReluKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
