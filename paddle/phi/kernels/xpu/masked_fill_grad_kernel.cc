// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/masked_fill_grad_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/expand_grad_kernel.h"
#include "paddle/phi/kernels/expand_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/common_infer_shape_functions.h"

namespace phi {

template <typename T, typename Context>
void MaskedFillGradKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const DenseTensor& mask,
                          const DenseTensor& value,
                          const DenseTensor& out_grad,
                          DenseTensor* x_grad,
                          DenseTensor* v_grad) {
  using XPUType = typename XPUTypeTrait<T>::Type;

  if (out_grad.numel() == 0 || mask.numel() == 0) {
    if (x_grad) {
      Full<T, Context>(dev_ctx, x_grad->dims(), 0, x_grad);
    }
    if (v_grad) {
      Full<T, Context>(dev_ctx, v_grad->dims(), 0, v_grad);
    }
    return;
  }

  auto x_dims = x.dims();
  auto mask_dims = mask.dims();

  auto expanded_size =
      vectorize(funcs::BroadcastTwoDims(x_dims, mask_dims, -1));
  auto expanded_dims = make_ddim(expanded_size);

  DenseTensor mask_expand;
  DenseTensor x_grad_expand;
  DenseTensor value_grad_expand;

  bool expand_x = false;
  bool expand_value = false;

  if (mask.dims() != expanded_dims) {
    ExpandKernel<bool, Context>(
        dev_ctx, mask, IntArray(expanded_size), &mask_expand);
  } else {
    mask_expand = mask;
  }

  DenseTensor* x_grad_tmp = nullptr;
  if (x_grad) {
    if (x_grad->dims() != expanded_dims) {
      x_grad_expand = Empty<T, Context>(dev_ctx, IntArray(expanded_size));
      x_grad_tmp = &x_grad_expand;
      expand_x = true;
    } else {
      x_grad_tmp = x_grad;
    }
  }

  DenseTensor* value_grad_tmp = nullptr;
  if (v_grad) {
    if (v_grad->dims() != expanded_dims) {
      value_grad_expand = Empty<T, Context>(dev_ctx, IntArray(expanded_size));
      value_grad_tmp = &value_grad_expand;
      expand_value = true;
    } else {
      value_grad_tmp = v_grad;
    }
  }

  auto* cond_data = mask_expand.data<bool>();
  auto* dout_data = out_grad.data<T>();
  const int64_t len = mask_expand.numel();
  if (len <= 0) {
    return;
  }

  if (x_grad_tmp) {
    dev_ctx.template Alloc<T>(x_grad_tmp);
  }
  if (value_grad_tmp) {
    dev_ctx.template Alloc<T>(value_grad_tmp);
  }

  DenseTensor dx_dummy;
  DenseTensor dy_dummy;

  T* dx_ptr = nullptr;
  T* dy_ptr = nullptr;

  if (x_grad_tmp) {
    dx_ptr = x_grad_tmp->data<T>();
  } else {
    dx_dummy = Empty<T, Context>(dev_ctx, IntArray(expanded_size));
    dx_ptr = dx_dummy.data<T>();
  }

  if (value_grad_tmp) {
    dy_ptr = value_grad_tmp->data<T>();
  } else {
    dy_dummy = Empty<T, Context>(dev_ctx, IntArray(expanded_size));
    dy_ptr = dy_dummy.data<T>();
  }

  int r = xpu::masked_fill_grad<XPUType>(
      dev_ctx.x_context(),
      cond_data,
      reinterpret_cast<const XPUType*>(dout_data),
      reinterpret_cast<XPUType*>(dx_ptr),
      reinterpret_cast<XPUType*>(dy_ptr),
      len);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "masked_fill_grad");

  if (x_grad && expand_x) {
    ExpandGradKernel<T, Context>(
        dev_ctx, x, x_grad_expand, IntArray(expanded_size), x_grad);
  }

  if (v_grad) {
    if (expand_value) {
      ExpandGradKernel<T, Context>(
          dev_ctx, value, value_grad_expand, IntArray(expanded_size), v_grad);
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(masked_fill_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::MaskedFillGradKernel,
                   float,
                   int64_t,
                   phi::float16,
                   phi::bfloat16) {
  kernel->InputAt(1).SetDataType(phi::DataType::BOOL);
}
