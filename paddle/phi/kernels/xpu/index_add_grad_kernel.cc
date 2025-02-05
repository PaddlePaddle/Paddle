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

#include "paddle/phi/kernels/index_add_grad_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/phi/kernels/index_select_kernel.h"

namespace phi {

template <typename T, typename Context>
void IndexAddGradKernel(const Context& ctx,
                        const DenseTensor& index,
                        const DenseTensor& add_value,
                        const DenseTensor& out_grad,
                        int dim,
                        DenseTensor* x_grad,
                        DenseTensor* add_value_grad) {
  if (dim < 0) {
    dim += out_grad.dims().size();
  }

  if (x_grad) {
    phi::Copy(ctx, out_grad, ctx.GetPlace(), false, x_grad);
  }
  if (add_value_grad) {
    phi::IndexSelectKernel<T, Context>(
        ctx, out_grad, index, dim, add_value_grad);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(index_add_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::IndexAddGradKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   int,
                   int64_t) {}
