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

// #include "paddle/phi/kernels/gpu/rrelu_impl.cu.h"
#include "paddle/phi/kernels/rrelu_grad_kernel.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/for_range.h"

namespace phi {

// template <typename T, typename Context>
// void RReluGradKernel(const Context& dev_ctx,
//                      const DenseTensor& mask,
//                      const DenseTensor& out_grad,
//                      DenseTensor* x_grad) {
//   x_grad->mutable_data<T>(dev_ctx.GetPlace());
//   auto size = mask.numel();
//   paddle::operators::RReluGradGPUKernelDriver<T>(
//       dev_ctx, out_grad, mask, x_grad);
// }


template <typename T>
struct RReluGradCudaFunctor {
 public:
  RReluGradCudaFunctor(const T* mask,
                       const T* out_grad,
                       T* x_grad)
      : mask_(mask), out_grad_(out_grad), x_grad_(x_grad) {}

  __device__ void operator()(int64_t idx) {
      x_grad_[idx] = mask_[idx] * out_grad_[idx];
  }

 private:
  const T* mask_;
  const T* out_grad_;
  T* x_grad_;
};

template <typename T, typename Context>
void RReluGradKernel(const Context& ctx, 
                     const DenseTensor& mask,
                     const DenseTensor& out_grad,
                     DenseTensor* x_grad) {
  const T* mask_data = mask.data<T>();
  const T* out_grad_data = out_grad.data<T>();
  T* x_grad_data = ctx.template Alloc<T>(x_grad);
  auto size = mask.numel();

  phi::funcs::ForRange<Context> for_range(ctx, size);

  RReluGradCudaFunctor<T> functor(mask_data, out_grad_data, x_grad_data);
  for_range(functor);
}


}  // namespace phi

PD_REGISTER_KERNEL(rrelu_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::RReluGradKernel,
                   float,
                   double,
                   phi::dtype::float16) {} 
                  //  phi::dtype::bfloat16) {}                  
