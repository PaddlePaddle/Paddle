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

#include "paddle/phi/kernels/diagonal_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void DiagonalKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    int offset,
                    int axis1,
                    int axis2,
                    DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  T* out_data = dev_ctx.template Alloc<T>(out);
  std::vector<int64_t> xshape = common::vectorize<int64_t>(x.dims());
  std::vector<int64_t> yshape = common::vectorize<int64_t>(out->dims());

  int r = xpu::diagonal(dev_ctx.x_context(),
                        reinterpret_cast<const XPUType*>(x.data<T>()),
                        reinterpret_cast<XPUType*>(out_data),
                        xshape,
                        yshape,
                        axis1,
                        axis2,
                        offset);

  PADDLE_ENFORCE_XDNN_SUCCESS(r, "diagonal");
}
}  // namespace phi
PD_REGISTER_KERNEL(diagonal,
                   XPU,
                   ALL_LAYOUT,
                   phi::DiagonalKernel,
                   float,
                   phi::dtype::float16,
                   int,
                   int64_t,
                   bool) {}
