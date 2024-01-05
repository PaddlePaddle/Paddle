// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/copysign_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
namespace phi {
template <typename T, typename Context>
void CopySignKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& y,
                    DenseTensor* out) {
  std::vector<const DenseTensor*> inputs = {&x, &y};
  std::vector<DenseTensor*> outputs = {out};
  dev_ctx.template Alloc<T>(out);
  funcs::BroadcastKernel<T>(
      dev_ctx, inputs, &outputs, phi::CopySignFunctor<T>());
}
}  // namespace phi

PD_REGISTER_KERNEL(copysign,
                   GPU,
                   ALL_LAYOUT,
                   phi::CopySignKernel,
                   bool,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
