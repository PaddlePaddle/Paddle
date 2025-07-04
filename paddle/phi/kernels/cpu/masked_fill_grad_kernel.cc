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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/expand_grad_kernel.h"
#include "paddle/phi/kernels/expand_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/common_infer_shape_functions.h"
#include "paddle/phi/kernels/funcs/common_shape.h"
namespace phi {

template <typename T, typename Context>
void MaskedFillGradKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const DenseTensor& mask,
                          const DenseTensor& value UNUSED,
                          const DenseTensor& out_grad,
                          DenseTensor* x_grad,
                          DenseTensor* v_grad) {
  if (out_grad.numel() == 0 || mask.numel() == 0) {
    // x shape [2, 1, 3], mask shape [2, 0, 3], x_grad shape [2, 1, 3]
    if (x_grad) {
      phi::Full<T, Context>(
          dev_ctx, phi::IntArray(common::vectorize(x_grad->dims())), 0, x_grad);
    }
    if (v_grad) {
      phi::Full<T, Context>(
          dev_ctx, phi::IntArray(common::vectorize(v_grad->dims())), 0, v_grad);
    }
    return;
  }

  auto x_grad_dims = x_grad->dims();
  auto mask_dims = mask.dims();
  bool expand_x = false;
  bool expand_value = false;
  auto expanded_size =
      common::vectorize(funcs::BroadcastTwoDims(x_grad_dims, mask_dims, -1));

  DenseTensor mask_expand;
  DenseTensor x_grad_expand;
  DenseTensor value_grad_expand;

  auto expanded_dims = common::make_ddim(expanded_size);

  if (mask.dims() != expanded_dims) {
    ExpandKernel<bool, Context>(
        dev_ctx, mask, IntArray(expanded_size), &mask_expand);
  } else {
    mask_expand = mask;
  }

  if (x_grad->dims() != expanded_dims) {
    x_grad_expand = Empty<T, Context>(dev_ctx, IntArray(expanded_size));
    expand_x = true;
  } else {
    x_grad_expand = *x_grad;
  }

  if (v_grad) {
    if (v_grad->dims() != expanded_dims && v_grad->numel() != 1) {
      value_grad_expand = Empty<T, Context>(dev_ctx, IntArray(expanded_size));
      expand_value = true;
    } else {
      value_grad_expand = *x_grad;
    }
  }

  auto* mask_data = mask_expand.data<bool>();
  auto* dout = out_grad.data<T>();
  auto numel = mask_expand.numel();

  if (numel <= 0) return;
  if (x_grad) {
    dev_ctx.template Alloc<T>(x_grad);

    DenseTensor* x_grad_tmp = x_grad;
    if (expand_x) {
      x_grad_tmp = &x_grad_expand;
    }
    auto* dx = x_grad_tmp->data<T>();
    for (int i = 0; i < numel; i++) {
      dx[i] = mask_data[i] ? T{} : dout[i];
    }

    if (expand_x) {
      ExpandGradKernel<T, Context>(
          dev_ctx, x, x_grad_expand, IntArray(expanded_size), x_grad);
    }
  }

  if (v_grad) {
    dev_ctx.template Alloc<T>(v_grad);

    DenseTensor* value_grad_tmp = v_grad;
    if (expand_value) {
      value_grad_tmp = &value_grad_expand;
    }

    auto* dv = value_grad_tmp->data<T>();
    if (v_grad->numel() == 1) {
      dv[0] = 0;
      for (int i = 0; i < numel; i++) {
        if (mask_data[i]) {
          dv[0] += dout[i];
        }
      }
    } else {
      for (int i = 0; i < numel; i++) {
        if (mask_data[i]) {
          dv[i] = dout[i];
        }
      }

      if (expand_value) {
        ExpandGradKernel<T, Context>(
            dev_ctx, x, value_grad_expand, IntArray(expanded_size), v_grad);
      }
    }
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(masked_fill_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::MaskedFillGradKernel,
                   bool,
                   float,
                   double,
                   int,
                   int8_t,
                   int64_t,
                   int16_t,
                   uint8_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->InputAt(1).SetDataType(phi::DataType::BOOL);
}
