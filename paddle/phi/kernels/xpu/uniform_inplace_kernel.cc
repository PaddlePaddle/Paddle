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
void XPUUniformRandomInplaceKernel(const Context& ctx,
                                   const DenseTensor& x,
                                   float min,
                                   float max,
                                   int seed_in,
                                   int diag_num_in,
                                   int diag_step_in,
                                   float diag_val,
                                   DenseTensor* out) {
  T* data = ctx.template Alloc<T>(out);
  int64_t size = out->numel();
  std::unique_ptr<T[]> data_cpu(new T[size]);
  std::uniform_real_distribution<T> dist(static_cast<T>(min),
                                         static_cast<T>(max));
  unsigned int seed = static_cast<unsigned int>(seed_in);
  auto engine = phi::GetCPURandomEngine(seed);
  for (int64_t i = 0; i < size; ++i) {
    data_cpu[i] = dist(*engine);
  }

  unsigned int diag_num = static_cast<unsigned int>(diag_num_in);
  unsigned int diag_step = static_cast<unsigned int>(diag_step_in);
  if (diag_num > 0) {
    PADDLE_ENFORCE_GT(
        size,
        (diag_num - 1) * (diag_step + 1),
        common::errors::InvalidArgument(
            "ShapeInvalid: the diagonal's elements is equal (num-1) "
            "* (step-1) with num %d, step %d,"
            "It should be smaller than %d, but received %d",
            diag_num,
            diag_step,
            (diag_num - 1) * (diag_step + 1),
            size));
    for (int64_t i = 0; i < diag_num; ++i) {
      int64_t pos = i * diag_step + i;
      data_cpu[pos] = diag_val;
    }
  }
  phi::memory_utils::Copy(ctx.GetPlace(),
                          data,
                          phi::CPUPlace(),
                          reinterpret_cast<void*>(data_cpu.get()),
                          size * sizeof(T));
}
}  // namespace phi

PD_REGISTER_KERNEL(uniform_inplace,
                   XPU,
                   ALL_LAYOUT,
                   phi::XPUUniformRandomInplaceKernel,
                   float) {}
