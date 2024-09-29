//   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/unbind_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void UnbindKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  int axis,
                  std::vector<DenseTensor*> outs) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  auto x_dims = x.dims();
  axis = axis < 0 ? x_dims.size() + axis : axis;

  std::vector<XPUType*> y_ptrs;
  for (size_t j = 0; j < outs.size(); ++j) {
    dev_ctx.template Alloc<T>(outs[j]);
    y_ptrs.emplace_back(reinterpret_cast<XPUType*>(outs[j]->data<T>()));
  }
  auto x_shape = common::vectorize<int>(x.dims());
  int r = xpu::unbind(dev_ctx.x_context(),
                      reinterpret_cast<const XPUType*>(x.data<T>()),
                      y_ptrs,
                      x_shape,
                      axis);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "unbind");
}

}  // namespace phi

PD_REGISTER_KERNEL(
    unbind, XPU, ALL_LAYOUT, phi::UnbindKernel, float, phi::dtype::bfloat16) {}
