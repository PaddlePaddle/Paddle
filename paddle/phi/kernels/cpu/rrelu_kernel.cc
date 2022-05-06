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

#include <random>

#include "paddle/phi/kernels/rrelu_kernel.h"
#include "paddle/fluid/framework/generator.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void RReluKernel(const Context& ctx,
                 const DenseTensor& x,
                 const float lower,
                 const float upper,
                 bool is_test,
                //  bool fix_seed,
                //  int seed,
                 DenseTensor* out,
                 DenseTensor* mask) {
  // auto* y = out;
  const T* x_data = x.data<T>();
  // you may try the following 2 lines(what is the difference?)
  T* out_data = ctx.template Alloc<T>(out);
  T* mask_data = ctx.template Alloc<T>(mask);
  // auto* y_data = y->mutable_data<T>(dev_ctx.GetPlace());
  // auto* mask_data = mask->mutable_data<T>(dev_ctx.GetPlace());
  uint64_t size = x.numel();
  auto zero = static_cast<T>(0);
  auto one = static_cast<T>(1);

  if (!is_test) {
    // int seed_data = fix_seed ? seed : 0;
    // auto engine = paddle::framework::GetCPURandomEngine(seed_data);
    // std::uniform_real_distribution<float> dist(lower, upper);
    
    auto gen = ctx.GetGenerator();
    auto engine = gen->GetCPUEngine();
    std::uniform_real_distribution<float> dist(lower, upper);

    for (uint64_t i = 0; i < size; ++i) {
      if (x_data[i] >= zero) {
        mask_data[i] = one;
        out_data[i] = x_data[i];
      } else {
        auto ramdom_sampled_value = static_cast<T>(dist(*engine));
        mask_data[i] = ramdom_sampled_value;
        out_data[i] = x_data[i] * ramdom_sampled_value;
      }
    }
  } else {
    auto middle_value = static_cast<T>((lower + upper) / 2.0f);
    for (uint64_t i = 0; i < size; ++i) {
      if (x_data[i] >= zero) {
        out_data[i] = x_data[i];
        mask_data[i] = one;
      } else {
        out_data[i] = x_data[i] * middle_value;
        mask_data[i] = middle_value;
      }
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(rrelu,
                   CPU,
                   ALL_LAYOUT,
                   phi::RReluKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
