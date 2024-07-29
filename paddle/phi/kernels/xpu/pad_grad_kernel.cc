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

#include "paddle/phi/kernels/pad_grad_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
template <typename T, typename Context>
void PadGradKernel(const Context& dev_ctx,
                   const DenseTensor& d_out,
                   const std::vector<int>& paddings,
                   const Scalar& pad_value,
                   bool pad_from_first_axis,
                   DenseTensor* d_x) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  std::vector<int> pad_left, pad_right;
  std::vector<int> out_shape = common::vectorize<int>(d_out.dims());
  dev_ctx.template Alloc<T>(d_x);

  // pad the length of paddings to 2*x.ndim
  auto x_dim = d_out.dims();
  std::vector<int> pad(2 * x_dim.size());
  int paddings_len = paddings.size();
  for (size_t i = 0; i < pad.size(); ++i) {
    int pad_i = static_cast<int>(i) < paddings_len ? paddings[i] : 0;
    pad[i] = pad_i;
  }

  if ((static_cast<int>(paddings_len) == x_dim.size() * 2) &&
      pad_from_first_axis) {
    for (size_t i = 0; i < pad.size() / 2; ++i) {
      pad_left.push_back(-pad[i * 2]);
      pad_right.push_back(-pad[i * 2 + 1]);
    }
  } else {
    std::vector<int> pad_reversed(2 * x_dim.size());
    for (int i = 2 * x_dim.size() - 1; i >= 0; --i) {
      int index = 2 * x_dim.size() - 1 - i;
      pad_reversed[i] = (index % 2 == 1) ? pad[index - 1] : pad[index + 1];
    }
    for (size_t i = 0; i < pad_reversed.size() / 2; ++i) {
      pad_left.push_back(-pad_reversed[i * 2]);
      pad_right.push_back(-pad_reversed[i * 2 + 1]);
    }
  }

  XPUType value = static_cast<XPUType>(pad_value.to<T>());
  int r = xpu::pad<XPUType>(dev_ctx.x_context(),
                            reinterpret_cast<const XPUType*>(d_out.data<T>()),
                            reinterpret_cast<XPUType*>(d_x->data<T>()),
                            out_shape,
                            pad_left,
                            pad_right,
                            value);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "pad");
}
}  // namespace phi

PD_REGISTER_KERNEL(pad_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::PadGradKernel,
                   float,
                   int,
                   int16_t,
                   phi::dtype::float16) {}
