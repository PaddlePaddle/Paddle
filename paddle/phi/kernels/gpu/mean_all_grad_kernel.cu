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

#include "paddle/phi/kernels/mean_all_kernel.h"

#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T>
__global__ void MeanRunKernel(const T* in_data, T* out_data, int N) {
  using MT = typename dtype::MPTypeTrait<T>::Type;
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  auto data = static_cast<MT>(in_data[0]);
  for (; idx < N; idx += blockDim.x * gridDim.x) {
    out_data[idx] = static_cast<T>(data / (static_cast<MT>(N)));
  }
}

template <typename T, typename Context>
void MeanAllGradKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const DenseTensor& out_grad,
                       DenseTensor* x_grad) {
  PADDLE_ENFORCE_EQ(out_grad.numel(),
                    1,
                    phi::errors::InvalidArgument(
                        "Mean Gradient Input Tensor len should be 1. But "
                        "received Out@Grad's elements num is %d.",
                        out_grad.numel()));
  dev_ctx.template Alloc<T>(x_grad);

  auto in_data = out_grad.data<T>();
  auto size_prob = x_grad->numel();
  auto out_data = x_grad->data<T>();
  int threads = 512;
  int grid = (size_prob + threads - 1) / threads;
  auto stream = dev_ctx.stream();
  MeanRunKernel<T><<<grid, threads, 0, stream>>>(in_data, out_data, size_prob);
}

}  // namespace phi

PD_REGISTER_KERNEL(mean_all_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::MeanAllGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
