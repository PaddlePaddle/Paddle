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

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/copy_kernel.h"
#include "paddle/phi/kernels/index_add_grad_kernel.h"

DECLARE_bool(cudnn_deterministic);

namespace phi {

template <typename T, typename Context>
void IndexAddGradKernel(const Context& dev_ctx,
                        const DenseTensor& out_grad,
                        int axis,
                        float added_value,
                        DenseTensor* x_grad) {
  phi::Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), false, x_grad);
}

}  // namespace phi

PD_REGISTER_KERNEL(index_add_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::IndexAddGradKernel,
                   float,
                   phi::dtype::float16,
                   double,
                   int,
                   int64_t) {}
