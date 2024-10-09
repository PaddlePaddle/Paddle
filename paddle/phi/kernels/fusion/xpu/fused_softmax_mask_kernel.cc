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

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
namespace fusion {

template <typename T, typename Context>
void FusedSoftmaxMaskKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            const DenseTensor& mask,
                            DenseTensor* out) {
  auto* x_data = x.data<T>();
  auto* mask_data = mask.data<T>();
  auto* y_data = dev_ctx.template Alloc<T>(out);

  auto x_dim = x.dims();
  auto mask_dim = mask.dims();

  PADDLE_ENFORCE_EQ(mask_dim[1],
                    1,
                    common::errors::InvalidArgument(
                        "Input mask's second dim must be 1 "
                        "received the second dimension of mask is %d",
                        mask_dim[1]));

  // dim of x and mask must be equal
  for (size_t idx = 0; idx < 4; ++idx) {
    if (idx == 1) continue;
    PADDLE_ENFORCE_EQ(
        x_dim[idx],
        mask_dim[idx],
        common::errors::InvalidArgument(
            "Input x's %dth dim should be equal with input mask's %dth dim "
            "but "
            "received the %dth dimension of x and mask are not equal "
            "the %dth dim of x is %d, while the %dth dim of mask is %d.",
            idx,
            idx,
            idx,
            idx,
            x_dim[idx],
            idx,
            mask_dim[idx]));
  }
  std::vector<int64_t> x_shape = common::vectorize<int64_t>(x.dims());
  std::vector<int64_t> mask_shape = common::vectorize<int64_t>(mask.dims());

  // int softmax_with_mask(Context* ctx, const T* x, const T* mask, T* y, const
  // std::vector<int64_t>& x_shape, const std::vector<int64_t>& mask_shape);
  int r = xpu::softmax_with_mask(
      dev_ctx.x_context(), x_data, mask_data, y_data, x_shape, mask_shape);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "softmax_with_mask");
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_softmax_mask,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::FusedSoftmaxMaskKernel,
                   float) {}
