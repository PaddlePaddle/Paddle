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

#include "paddle/phi/kernels/norm_kernel.h"

#include <vector>

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void NormKernel(const Context& ctx,
                const DenseTensor& x,
                int axis,
                float epsilon,
                bool is_test,
                DenseTensor* out,
                DenseTensor* norm) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  ctx.template Alloc<T>(out);
  ctx.template Alloc<T>(norm);

  std::vector<int> xshape;
  auto x_dims = x.dims();
  auto x_dims_size = x_dims.size();
  xshape.resize(x_dims_size);

  if (axis < 0) {
    axis += x_dims_size;
  }

  PADDLE_ENFORCE_GE(
      axis,
      0,
      common::errors::InvalidArgument("axis must be greater than or equal to 0."
                                      "But received axis: %d.",
                                      axis));
  PADDLE_ENFORCE_LT(axis,
                    x_dims_size,
                    common::errors::InvalidArgument(
                        "Attr(axis) value must be less than rank of Input(X)"
                        "But received axis: %d, rank: %d.",
                        axis,
                        x_dims_size));

  for (int i = 0; i < x_dims_size; i++) {
    xshape[i] = static_cast<int>(x_dims[i]);
  }

  int r = xpu::l2_norm(ctx.x_context(),
                       reinterpret_cast<const XPUType*>(x.data<T>()),
                       reinterpret_cast<XPUType*>(out->data<T>()),
                       reinterpret_cast<XPUType*>(norm->data<T>()),
                       xshape,
                       axis,
                       epsilon);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "l2_norm");
}

}  // namespace phi

PD_REGISTER_KERNEL(
    norm, XPU, ALL_LAYOUT, phi::NormKernel, float, phi::dtype::float16) {}
// TODO(zhangyikun02): add bfloat16 when xpu support it
