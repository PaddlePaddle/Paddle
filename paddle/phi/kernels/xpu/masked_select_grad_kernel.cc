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

#include "paddle/phi/kernels/masked_select_grad_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/expand_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/common_shape.h"

namespace phi {

template <typename T, typename Context>
void MaskedSelectGradKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            const DenseTensor& mask,
                            const DenseTensor& out_grad,
                            DenseTensor* x_grad) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  dev_ctx.template Alloc<T>(x_grad);
  if (x_grad && x_grad->numel() == 0) {
    return;
  }
  // If out_grad is empty (e.g. mask all false), x_grad should be all zeros.
  if (out_grad.numel() == 0 && x_grad) {
    Full<T, Context>(dev_ctx, x_grad->dims(), 0, x_grad);
    return;
  }

  auto expanded_size = funcs::MatrixGetBroadcastBatchPortion(
      vectorize(x_grad->dims()), vectorize(mask.dims()));
  auto expanded_dims = make_ddim(expanded_size);

  DenseTensor mask_expand;
  // XDNN masked_select_grad requires x and mask to have matching shapes.
  if (mask.dims() != expanded_dims) {
    ExpandKernel<bool, Context>(
        dev_ctx, mask, IntArray(expanded_size), &mask_expand);
  } else {
    mask_expand = mask;
  }

  auto* mask_data = mask_expand.data<bool>();
  auto* input_data = reinterpret_cast<const XPUType*>(out_grad.data<T>());
  auto* out_data = reinterpret_cast<XPUType*>(x_grad->data<T>());

  auto mask_shape = vectorize<int64_t>(mask_expand.dims());
  auto xshape = vectorize<int64_t>(x_grad->dims());
  if (mask_expand.dims().size() == 0) {
    mask_shape = std::vector<int64_t>({1});
  }
  if (x_grad->dims().size() == 0) {
    xshape = std::vector<int64_t>({1});
  }

  const int64_t out_size = out_grad.numel();
  int r = xpu::masked_select_grad(dev_ctx.x_context(),
                                  input_data,
                                  mask_data,
                                  out_data,
                                  xshape,
                                  mask_shape,
                                  out_size);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "masked_select_grad");
}

}  // namespace phi

PD_REGISTER_KERNEL(masked_select_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::MaskedSelectGradKernel,
                   float,
                   phi::float16,
                   phi::bfloat16,
                   int,
                   bool,
                   int64_t) {}
