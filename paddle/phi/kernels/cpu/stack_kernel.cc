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

#include "paddle/phi/kernels/stack_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void StackKernel(const Context& dev_ctx,
                 const std::vector<const DenseTensor*>& x,
                 int axis,
                 DenseTensor* out) {
  if (axis < 0) axis += (x[0]->dims().size() + 1);

  auto x_dims = x[0]->dims();
  for (int i = 0; i < x_dims.size(); i++) {
    PADDLE_ENFORCE_GE(
        x_dims[i],
        0,
        common::errors::InvalidArgument(
            "The dims of Input(X) should be greater than or equal to 0"));
  }
  // zero sized tensor case
  if (x[0]->numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    auto out_dims = out->dims();
    out->Resize(out_dims);
    return;
  }

  int n = static_cast<int>(x.size());
  T* y_data = dev_ctx.template Alloc<T>(out);
  std::vector<const T*> x_datas(n);
  for (int i = 0; i < n; i++) x_datas[i] = x[i]->data<T>();

  int pre = 1, post = 1;
  auto& dim = x[0]->dims();
  for (auto i = 0; i < axis; ++i) pre *= static_cast<int>(dim[i]);
  for (auto i = axis; i < dim.size(); ++i) post *= static_cast<int>(dim[i]);

  auto x_data_arr = x_datas.data();

  size_t x_offset = 0;
  size_t y_offset = 0;
  for (int i = 0; i < pre; i++) {
    for (int j = 0; j < n; j++) {
      std::memcpy(
          y_data + y_offset, x_data_arr[j] + x_offset, post * sizeof(T));
      y_offset += post;
    }
    x_offset += post;
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(stack,
                   CPU,
                   ALL_LAYOUT,
                   phi::StackKernel,
                   bool,
                   float,
                   double,
                   int,
                   int8_t,
                   int64_t,
                   int16_t,
                   uint8_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
