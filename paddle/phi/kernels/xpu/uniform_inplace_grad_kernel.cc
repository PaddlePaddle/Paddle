// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/generator.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
template <typename T, typename Context>
void XPUUniformRandomInplaceGradKernel(const Context& ctx,
                                       const DenseTensor& out_grad,
                                       float min UNUSED,
                                       float max UNUSED,
                                       int seed UNUSED,
                                       int diag_num UNUSED,
                                       int diag_step UNUSED,
                                       float diag_val UNUSED,
                                       DenseTensor* x_grad) {
  auto* dx = x_grad;
  if (dx) {
    T* data = ctx.template Alloc<T>(dx);
    int64_t size = dx->numel();
    std::unique_ptr<T[]> data_cpu(new T[size]);
    for (int64_t i = 0; i < size; ++i) {
      data_cpu[i] = T(0);
    }
    phi::memory_utils::Copy(ctx.GetPlace(),
                            data,
                            phi::CPUPlace(),
                            reinterpret_cast<void*>(data_cpu.get()),
                            size * sizeof(T));
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(uniform_inplace_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::XPUUniformRandomInplaceGradKernel,
                   float) {}
