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

#include "paddle/phi/kernels/rrelu_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/generator.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void RReluKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 const float lower,
                 const float upper,
                 bool is_test,
                 DenseTensor* out,
                 DenseTensor* noise) {
  const T* x_ptr = x.data<T>();
  T* o_ptr = dev_ctx.template Alloc<T>(out);
  T* n_ptr = dev_ctx.template Alloc<T>(noise);
  T zero = static_cast<T>(0);
  int numel = static_cast<int>(x.numel());
  int i = 0;

  if (is_test) {
    T mid_val = static_cast<T>((lower + upper) / 2.0);
    for (i = 0; i < numel; i++) {
      if (x_ptr[i] < zero) {
        o_ptr[i] = mid_val * x_ptr[i];
        n_ptr[i] = mid_val;
      } else {
        o_ptr[i] = x_ptr[i];
        n_ptr[i] = 1.0;
      }
    }

    return;
  }

  auto engine = dev_ctx.GetGenerator()->GetCPUEngine();

  std::uniform_real_distribution<float> dist(lower, upper);

  for (i = 0; i < numel; i++) {
    if (x_ptr[i] < zero) {
      T scale = static_cast<T>(dist(*engine));
      o_ptr[i] = scale * x_ptr[i];
      n_ptr[i] = scale;
    } else {
      o_ptr[i] = x_ptr[i];
      n_ptr[i] = 1.0;
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(rrelu,
                   CPU,
                   ALL_LAYOUT,
                   phi::RReluKernel,
                   float,
                   phi::dtype::float16,
                   double) {}
