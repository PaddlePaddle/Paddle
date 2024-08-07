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

#include "paddle/phi/kernels/flip_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void FlipKernel(const Context& dev_ctx,
                const DenseTensor& x,
                const std::vector<int>& axis,
                DenseTensor* out) {
  using XPUInTDType = typename XPUTypeTrait<T>::Type;
  int x_rank = x.dims().size();
  std::vector<int64_t> formatted_axis(std::begin(axis), std::end(axis));
  for (size_t i = 0; i < axis.size(); i++) {
    if (axis[i] < 0) {
      formatted_axis[i] = static_cast<int64_t>(axis[i] + x_rank);
    }
  }
  dev_ctx.template Alloc<T>(out);
  if (out->numel() == 0) {
    return;
  }
  if (formatted_axis.size() == 0) {
    phi::Copy<Context>(dev_ctx, x, dev_ctx.GetPlace(), false, out);
    return;
  }
  std::vector<int64_t> x_shape = common::vectorize(x.dims());
  auto x_data = reinterpret_cast<const XPUInTDType*>(x.data<T>());
  auto out_data = reinterpret_cast<XPUInTDType*>(out->data<T>());
  auto numel = x.numel();
  if (numel <= 0) {
    return;
  }
  int r = xpu::flip<XPUInTDType>(
      /* Context* ctx */ dev_ctx.x_context(),
      /* const T* x */ x_data,
      /* T* y */ out_data,
      /* const std::vector<int64_t>& xshape */ x_shape,
      /* const std::vector<int64_t>& axis */ formatted_axis);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "flip");
}

}  // namespace phi

PD_REGISTER_KERNEL(flip, XPU, ALL_LAYOUT, phi::FlipKernel, float) {}
