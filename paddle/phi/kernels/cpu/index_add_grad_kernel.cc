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
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/phi/kernels/copy_kernel.h"
#include "paddle/phi/kernels/cpu/index_add_impl.h"

namespace phi {

template <typename T, typename Context>
void IndexAddGradKernel(const Context& dev_ctx,
                         const DenseTensor& out_grad,
                         const IntArray& index_arr,
                         const Scalar& axis_scalar,
                         float add_value,
                         DenseTensor* x_grad) {
  // float add_val = 0.0;
  // IndexAddBaseKernel<T, Context>(
  //     dev_ctx, out_grad, index_arr, axis_scalar, add_val, x_grad, nullptr);

  phi::Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), false, x_grad);
}

}  // namespace phi

PD_REGISTER_KERNEL(index_add_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::IndexAddGradKernel,
                   bool,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   double,
                   int,
                   int64_t) {}