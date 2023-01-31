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

#include "paddle/phi/kernels/stack_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void StackKernel(const Context& dev_ctx,
                 const std::vector<const DenseTensor*>& x,
                 int axis,
                 DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  if (axis < 0) {
    axis += x[0]->dims().size() + 1;
  }
  dev_ctx.template Alloc<T>(out);
  auto& dim = x[0]->dims();
  std::vector<int> xdims;
  for (auto i = 0; i < dim.size(); ++i) {
    xdims.push_back(dim[i]);
  }
  xdims.push_back(1);
  std::vector<std::vector<int>> xdims_list;
  int n = static_cast<int>(x.size());
  for (int i = 0; i < n; i++) {
    xdims_list.push_back(xdims);
  }

  std::vector<const XPUType*> x_list;
  for (int i = 0; i < n; i++) {
    x_list.push_back(reinterpret_cast<const XPUType*>(x[i]->data<T>()));
  }

  int r = xpu::concat<XPUType>(dev_ctx.x_context(),
                               x_list,
                               reinterpret_cast<XPUType*>(out->data<T>()),
                               xdims_list,
                               axis);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "concat in stack op");
}
}  // namespace phi

PD_REGISTER_KERNEL(
    stack, XPU, ALL_LAYOUT, phi::StackKernel, float, int, int64_t) {}
