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

#include "paddle/phi/kernels/masked_scatter_grad_kernel.h"

#include <cstring>

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/expand_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/common_infer_shape_functions.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"

namespace phi {

template <typename T, typename Context>
void MaskedScatterGradKernel(const Context& dev_ctx,
                             const DenseTensor& x,
                             const DenseTensor& mask,
                             const DenseTensor& value,
                             const DenseTensor& out_grad,
                             DenseTensor* x_grad,
                             DenseTensor* value_grad) {
  if (out_grad.numel() == 0 || mask.numel() == 0) {
    if (x_grad) {
      phi::Full<T, Context>(dev_ctx, x_grad->dims(), static_cast<T>(0), x_grad);
    }
    if (value_grad) {
      phi::Full<T, Context>(
          dev_ctx, value_grad->dims(), static_cast<T>(0), value_grad);
    }
    return;
  }

  auto out_grad_dims = out_grad.dims();
  auto mask_dims = mask.dims();
  auto expanded_size =
      vectorize(funcs::BroadcastTwoDims(out_grad_dims, mask_dims, -1));
  DDim expanded_dims = make_ddim(expanded_size);

  DenseTensor mask_expand;
  if (mask_dims != expanded_dims) {
    ExpandKernel<bool, Context>(
        dev_ctx, mask, IntArray(expanded_size), &mask_expand);
  } else {
    mask_expand = mask;
  }

  auto* mask_data = mask_expand.data<bool>();
  auto* out_grad_data = out_grad.data<T>();
  int64_t total = out_grad.numel();

  if (x_grad) {
    auto x_grad_dims = x_grad->dims();
    if (x_grad_dims == out_grad_dims) {
      // No broadcast happened, compute directly into x_grad.
      dev_ctx.template Alloc<T>(x_grad);
      auto* x_grad_data = x_grad->data<T>();
      for (int64_t i = 0; i < total; i++) {
        x_grad_data[i] = mask_data[i] ? static_cast<T>(0) : out_grad_data[i];
      }
    } else {
      // Broadcast happened: compute at broadcast shape, then reduce-sum.
      DenseTensor x_grad_broadcast;
      x_grad_broadcast.Resize(expanded_dims);
      dev_ctx.template Alloc<T>(&x_grad_broadcast);
      auto* x_grad_broadcast_data = x_grad_broadcast.data<T>();
      for (int64_t i = 0; i < total; i++) {
        x_grad_broadcast_data[i] =
            mask_data[i] ? static_cast<T>(0) : out_grad_data[i];
      }
      std::vector<int> reduce_dims =
          funcs::GetReduceDim(x_grad_dims, expanded_dims, -1);
      phi::SumKernel<T, Context>(dev_ctx,
                                 x_grad_broadcast,
                                 reduce_dims,
                                 x_grad_broadcast.dtype(),
                                 false,
                                 x_grad);
    }
  }

  if (value_grad) {
    dev_ctx.template Alloc<T>(value_grad);
    auto* value_grad_data = value_grad->data<T>();
    int64_t value_numel = value_grad->numel();
    std::memset(value_grad_data, 0, value_numel * sizeof(T));

    int64_t count = 0;
    for (int64_t i = 0; i < total; i++) {
      if (mask_data[i]) {
        if (count < value_numel) {
          value_grad_data[count] = out_grad_data[i];
        }
        count++;
      }
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(masked_scatter_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::MaskedScatterGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   int16_t,
                   int8_t,
                   uint8_t,
                   phi::float16,
                   phi::bfloat16) {
  kernel->InputAt(1).SetDataType(phi::DataType::BOOL);
}
