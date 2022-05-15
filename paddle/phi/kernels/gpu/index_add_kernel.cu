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
#include "paddle/phi/kernels/gpu/index_add_funcs.h"
#include "paddle/phi/kernels/index_add_kernel.h"

namespace phi {

template <typename T, typename Context>
void IndexAddKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const IntArray& index_arr,
                     const Scalar& axis_scalar,
                     float add_value,
                     DenseTensor* output) {
  IndexAddBaseKernel<T, Context>(
      dev_ctx, x, index_arr, axis_scalar, add_value, output, nullptr);
}

}  // namespace phi

PD_REGISTER_KERNEL(index_add,
                   GPU,
                   ALL_LAYOUT,
                   phi::IndexAddKernel,
                   bool,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   double,
                   int,
                   int64_t) {}