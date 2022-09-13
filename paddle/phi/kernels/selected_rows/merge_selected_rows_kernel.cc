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

#include "paddle/phi/kernels/selected_rows/merge_selected_rows_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

#include "paddle/fluid/operators/math/selected_rows_functor.h"

namespace phi {
namespace sr {

template <typename T, typename Context>
void MergeSelectedRowsKernel(const Context& dev_ctx,
                             const SelectedRows& x,
                             SelectedRows* out) {
  paddle::operators::math::scatter::MergeAdd<Context, T> merge_func;
  merge_func(dev_ctx, x, out);
}

}  // namespace sr
}  // namespace phi

PD_REGISTER_KERNEL(merge_selected_rows,
                   CPU,
                   ALL_LAYOUT,
                   phi::sr::MergeSelectedRowsKernel,
                   float,
                   double) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(merge_selected_rows,
                   GPU,
                   ALL_LAYOUT,
                   phi::sr::MergeSelectedRowsKernel,
                   float,
                   double) {}
#endif
