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

#include "paddle/phi/kernels/tensor_array_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/layout.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
template <typename T, typename Context>
void ArrayLengthKernel(const Context& ctx,
                       const TensorArray& x,
                       DenseTensor* out) {
  out->Resize({1});
  ctx.template Alloc<int64_t>(out);
  *out->data<int64_t>() = static_cast<int64_t>(x.size());
}

}  // namespace phi

PD_REGISTER_KERNEL(array_length,
                   CPU,
                   ALL_LAYOUT,
                   phi::ArrayLengthKernel,
                   float,
                   double,
                   bool) {}
