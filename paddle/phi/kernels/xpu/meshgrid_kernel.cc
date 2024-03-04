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

#include "paddle/phi/kernels/meshgrid_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void MeshgridKernel(const Context& ctx,
                    const std::vector<const DenseTensor*>& inputs,
                    std::vector<DenseTensor*> outputs) {
  using XPUType = typename XPUTypeTrait<T>::Type;

  std::vector<const XPUType*> x_list;
  std::vector<XPUType*> y_list;
  std::vector<std::vector<int64_t>> xshape_list;

  for (const auto& x : inputs) {
    x_list.push_back(reinterpret_cast<const XPUType*>(x->data<T>()));
    xshape_list.emplace_back(common::vectorize<int64_t>(x->dims()));
  }
  for (auto& x : outputs) {
    ctx.template Alloc<T>(x);
    y_list.push_back(reinterpret_cast<XPUType*>(x->data<T>()));
  }
  int r = xpu::meshgrid<XPUType>(ctx.x_context(), x_list, y_list, xshape_list);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "meshgrid");
}

}  // namespace phi

PD_REGISTER_KERNEL(
    meshgrid, XPU, ALL_LAYOUT, phi::MeshgridKernel, float, int, int64_t) {}
