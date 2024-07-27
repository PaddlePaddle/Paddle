/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/stack_grad_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/stack_functor.h"

namespace phi {

template <typename T, typename Context>
void StackGradKernel(const Context& dev_ctx,
                     const DenseTensor& out,
                     int axis,
                     std::vector<DenseTensor*> x_grad) {
  if (axis < 0) axis += out.dims().size();
  int n = static_cast<int>(out.dims()[axis]);
  std::vector<T*> dx_datas(n);  // NOLINT

  for (int i = 0; i < n; i++) {
    if (x_grad[i] == nullptr) {
      dx_datas[i] = nullptr;
    } else {
      dx_datas[i] = dev_ctx.template Alloc<T>(x_grad[i]);
    }
  }
  auto dy_data = out.data<T>();

  // zero sized tensor case
  if (out.numel() == 0) {
    for (int i = 0; i < n; i++) {
      auto x_grad_dim = x_grad[i]->dims();
      x_grad[i]->Resize(x_grad_dim);
    }
    return;
  }

  int pre = 1;
  for (int i = 0; i < axis; ++i) pre *= static_cast<int>(out.dims()[i]);
  int total_num = static_cast<int>(out.numel());
  int post = total_num / (n * pre);
  auto dx_data_arr = dx_datas.data();
  phi::funcs::StackGradFunctorForRange(
      dev_ctx, dx_data_arr, dy_data, total_num, n, post);
}

}  // namespace phi

PD_REGISTER_KERNEL(stack_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::StackGradKernel,
                   bool,
                   float,
                   double,
                   int,
                   int8_t,
                   int16_t,
                   int64_t,
                   uint8_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
