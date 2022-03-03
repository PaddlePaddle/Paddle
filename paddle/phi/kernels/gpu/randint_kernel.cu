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

#include "paddle/phi/kernels/randint_kernel.h"

#include <random>

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/distribution_helper.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/memory/memcpy.h"

DECLARE_bool(use_curand);

namespace phi {

template <typename T, typename Context>
void RandintRawKernel(const Context& dev_ctx,
                      int low,
                      int high,
                      const ScalarArray& shape,
                      DataType dtype,
                      int seed,
                      DenseTensor* out) {
  out->Resize(phi::make_ddim(shape.GetData()));
  T* data = dev_ctx.template Alloc<T>(out);
  if (FLAGS_use_curand) {
    funcs::uniform_distribution<uint32_t> dist;
    funcs::uniform_int_transform<T, uint32_t> trans(low, high);
    funcs::distribution_and_transform<T>(dev_ctx, out, dist, trans);
  } else {
    DenseTensor tmp;
    tmp.Resize(phi::make_ddim(shape.GetData()));
    T* tmp_data = dev_ctx.template HostAlloc<T>(&tmp);

    std::shared_ptr<std::mt19937_64> engine;
    if (seed) {
      engine = std::make_shared<std::mt19937_64>();
      engine->seed(seed);
    } else {
      engine = dev_ctx.GetHostGenerator()->GetCPUEngine();
    }

    std::uniform_int_distribution<T> dist(low, high - 1);
    auto numel = out->numel();
    for (int64_t i = 0; i < numel; ++i) {
      tmp_data[i] = dist(*engine);
    }

    paddle::memory::Copy<phi::GPUPlace, phi::Place>(
        out->place(),
        data,
        tmp.place(),
        tmp_data,
        numel * paddle::experimental::SizeOf(out->dtype()),
        0);
  }
}

template <typename T, typename Context>
void RandintKernel(const Context& dev_ctx,
                   int low,
                   int high,
                   const ScalarArray& shape,
                   DataType dtype,
                   DenseTensor* out) {
  RandintRawKernel<T>(dev_ctx, low, high, shape, dtype, 0, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(
    randint_raw, GPU, ALL_LAYOUT, phi::RandintRawKernel, int, int64_t) {}

PD_REGISTER_KERNEL(randint, GPU, ALL_LAYOUT, phi::RandintKernel, int, int64_t) {
}
