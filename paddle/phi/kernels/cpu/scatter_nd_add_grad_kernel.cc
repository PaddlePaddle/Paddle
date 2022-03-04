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

#include "paddle/phi/kernels/scatter_nd_add_grad_kernel.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/copy_kernel.h"
#include "paddle/phi/kernels/funcs/gather.h"

namespace phi {

template <typename T, typename Context>
void ScatterNdAddGradKernel(const Context &ctx,
                            const DenseTensor &index,
                            const DenseTensor &updates,
                            const DenseTensor &out_grad,
                            DenseTensor *x_grad,
                            DenseTensor *updates_grad) {
  if (x_grad) {
    Copy(ctx, out_grad, ctx.GetPlace(), false, x_grad);
  }
  if (updates_grad) {
    ctx.template Alloc<T>(updates_grad);
    // Gradient by Gather: dUpdates = dO[Ids]
    const auto &index_type = index.dtype();
    if (index_type == phi::DataType::INT32) {
      phi::funcs::CPUGatherNd<T, int32_t>(ctx, out_grad, index, updates_grad);
    } else {
      phi::funcs::CPUGatherNd<T, int64_t>(ctx, out_grad, index, updates_grad);
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(scatter_nd_add_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::ScatterNdAddGradKernel,
                   float,
                   double,
                   int64_t,
                   int,
                   uint8_t) {}
