/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/average_accumulates_kernel.h"
#include "paddle/phi/kernels/impl/average_accumulates_kernel_impl.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <>
void GetAccumulators<phi::CPUContext>(const phi::CPUContext& dev_ctx,
                                      const DenseTensor& in_num_accumulates,
                                      const DenseTensor& in_old_num_accumulates,
                                      const DenseTensor& in_num_updates,
                                      int64_t* num_updates,
                                      int64_t* num_accumulates,
                                      int64_t* old_num_accumulates) {
  *old_num_accumulates = in_old_num_accumulates.data<int64_t>()[0];
  *num_accumulates = in_num_accumulates.data<int64_t>()[0];
  *num_updates = in_num_updates.data<int64_t>()[0];
}

template <>
void SetAccumulators<phi::CPUContext>(const phi::CPUContext& dev_ctx,
                                      int64_t num_updates,
                                      int64_t num_accumulates,
                                      int64_t old_num_accumulates,
                                      DenseTensor* out_num_accumulates,
                                      DenseTensor* out_old_num_accumulates,
                                      DenseTensor* out_num_updates) {
  out_old_num_accumulates->data<int64_t>()[0] = old_num_accumulates;
  out_num_accumulates->data<int64_t>()[0] = num_accumulates;
  out_num_updates->data<int64_t>()[0] = num_updates;
}

}  // namespace phi

PD_REGISTER_KERNEL(average_accumulates,
                   CPU,
                   ALL_LAYOUT,
                   phi::AverageAccumulatesKernel,
                   float,
                   double) {}
