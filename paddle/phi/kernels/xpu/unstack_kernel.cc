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

#include "paddle/phi/kernels/unstack_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void UnStackKernel(const Context &dev_ctx,
                   const DenseTensor &x,
                   int axis,
                   int num,
                   std::vector<DenseTensor *> outs) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  auto x_dims = x.dims();

  if (axis < 0) axis += x_dims.size();
  auto x_shape = common::vectorize<int>(x_dims);

  std::vector<int> dx_dims_list(outs.size(), 1);
  std::vector<XPUType *> dx_lists;
  for (size_t j = 0; j < outs.size(); ++j) {
    dev_ctx.template Alloc<T>(outs[j]);
    dx_lists.push_back(reinterpret_cast<XPUType *>(outs[j]->data<T>()));
  }

  int r = xpu::split<XPUType>(dev_ctx.x_context(),
                              reinterpret_cast<const XPUType *>(x.data<T>()),
                              dx_lists,
                              x_shape,
                              dx_dims_list,
                              axis);

  PADDLE_ENFORCE_XDNN_SUCCESS(r, "split in unstack op");
}

}  // namespace phi

PD_REGISTER_KERNEL(unstack,
                   XPU,
                   ALL_LAYOUT,
                   phi::UnStackKernel,
                   phi::dtype::float16,
                   float,
                   int,
                   int64_t) {}
