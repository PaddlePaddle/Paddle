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
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void RReluKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 const float lower,
                 const float upper,
                 DenseTensor* out,
                 DenseTensor* noise) {
  const T* x_ptr = x.data<T>();
  T* o_ptr = dev_ctx.template Alloc<T>(out);
  T* n_ptr = dev_ctx.template Alloc<T>(noise);

  std::uniform_real_distribution<T> dist(lower, upper);
  auto gen_ptr = dev_ctx.GetGenerator();
  auto engine = gen_ptr->GetCPUEngine();

  int numel = x.numel();
  int i = 0;
  for (i = 0; i < numel; i++) {
    if (x_ptr[i] < 0) {
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

PD_REGISTER_KERNEL(rrelu, CPU, ALL_LAYOUT, phi::RReluKernel, float, double) {}
