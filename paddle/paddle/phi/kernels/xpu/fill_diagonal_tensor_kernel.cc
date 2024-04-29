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

#include "paddle/phi/kernels/fill_diagonal_tensor_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void FillDiagonalTensorKernel(const Context &ctx,
                              const DenseTensor &x,
                              const DenseTensor &y,
                              int64_t offset,
                              int dim1,
                              int dim2,
                              DenseTensor *out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  T *out_data = ctx.template Alloc<T>(out);
  int r = xpu::copy(ctx.x_context(),
                    reinterpret_cast<const XPUType *>(x.data<T>()),
                    reinterpret_cast<XPUType *>(out_data),
                    x.numel());
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy");

  std::vector<int64_t> xshape = common::vectorize<int64_t>(x.dims());
  std::vector<int64_t> yshape = common::vectorize<int64_t>(y.dims());

  r = xpu::fill_diagonal_tensor(ctx.x_context(),
                                reinterpret_cast<const XPUType *>(x.data<T>()),
                                reinterpret_cast<const XPUType *>(y.data<T>()),
                                reinterpret_cast<XPUType *>(out_data),
                                xshape,
                                yshape,
                                dim1,
                                dim2,
                                offset);

  PADDLE_ENFORCE_XDNN_SUCCESS(r, "fill_diagonal_tensor");
}
}  // namespace phi

PD_REGISTER_KERNEL(fill_diagonal_tensor,
                   XPU,
                   ALL_LAYOUT,
                   phi::FillDiagonalTensorKernel,
                   float,
                   int64_t,
                   int,
                   phi::dtype::float16,
                   bool) {}
