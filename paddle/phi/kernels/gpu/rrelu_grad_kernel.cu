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

#include "paddle/phi/kernels/rrelu_grad_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_meta.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/reduce_function.h"
#include "paddle/phi/kernels/gpu/prelu_funcs.h"
#include "paddle/phi/kernels/primitive/functor_primitives.h"

namespace phi {

template <typename T>
__global__ void RReluOpGradKernel(const T* x_ptr,
                                  const T* noise_ptr,
                                  const T* out_grad_ptr,
                                  T* x_grad_ptr,
                                  int numel) {
  CUDA_KERNEL_LOOP(index, numel) {
    T scale = noise_ptr[index];
    T x = x_ptr[index];
    T out_grad = out_grad_ptr[index];
    T zero = static_cast<T>(0);
    x_grad_ptr[index] = (x < zero) ? scale * out_grad : out_grad;
  }
}

template <typename T>
class RReluOpGradFunctor {
 public:
  void operator()(gpuStream_t stream,
                  const T* x,
                  const T* noise,
                  const T* out_grad,
                  T* x_grad,
                  int numel) {
    RReluOpGradKernel<T>
        <<<PADDLE_GET_BLOCKS(numel), CUDA_NUM_THREADS, 0, stream>>>(
            x, noise, out_grad, x_grad, numel);
  }
};

template <typename T, typename Context>
void RReluGradKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& noise,
                     const DenseTensor& out_grad,
                     DenseTensor* x_grad) {
  if (!x_grad) return;
  dev_ctx.template Alloc<T>(x_grad);

  const T* x_ptr = x.data<T>();
  const T* n_ptr = noise.data<T>();
  const T* out_grad_ptr = out_grad.data<T>();
  T* x_grad_ptr = dev_ctx.template Alloc<T>(x_grad);

  int numel = x.numel();
  auto stream = dev_ctx.stream();

  RReluOpGradFunctor<T> rrelu_grad;
  rrelu_grad(stream, x_ptr, n_ptr, out_grad_ptr, x_grad_ptr, numel);
}

}  // namespace phi

PD_REGISTER_KERNEL(rrelu_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::RReluGradKernel,
                   float,
                   phi::dtype::float16,
                   double) {}
