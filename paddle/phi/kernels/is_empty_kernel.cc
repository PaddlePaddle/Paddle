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

#include "paddle/phi/kernels/is_empty_kernel.h"

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void IsEmptyKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   DenseTensor* out) {
  // Note: is_empty is always executed on CPU and the output data should
  // always be allocated for CPUPlace. We reigister CUDA kernel for this op to
  // avoid the unnecessary data transform.
  bool* out_data = dev_ctx.template HostAlloc<bool>(out);
  out_data[0] = phi::product(x.dims()) == 0;
}

}  // namespace phi

PD_REGISTER_KERNEL(is_empty,
                   CPU,
                   ALL_LAYOUT,
                   phi::IsEmptyKernel,
                   float,
                   double,
                   int,
                   int64_t) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(is_empty,
                   GPU,
                   ALL_LAYOUT,
                   phi::IsEmptyKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
#endif
