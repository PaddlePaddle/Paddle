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

#include <random>

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/generator.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void RandintWithSeedKernel(const Context& dev_ctx,
                           int low,
                           int high,
                           const IntArray& shape,
                           DataType dtype,
                           int seed,
                           DenseTensor* out) {
  int64_t size = out->numel();
  out->Resize(common::make_ddim(shape.GetData()));
  T* data = dev_ctx.template Alloc<T>(out);
  auto numel = out->numel();
  std::shared_ptr<std::mt19937_64> engine;
  if (seed) {
    engine = std::make_shared<std::mt19937_64>();
    engine->seed(seed);
  } else {
    engine = dev_ctx.GetGenerator()->GetCPUEngine();
  }
  std::unique_ptr<T[]> data_cpu(new T[size]);
  std::uniform_int_distribution<T> dist(low, high - 1);
  for (int64_t i = 0; i < numel; ++i) {
    data_cpu[i] = dist(*engine);
  }
  memory_utils::Copy(dev_ctx.GetPlace(),
                     data,
                     phi::CPUPlace(),
                     reinterpret_cast<void*>(data_cpu.get()),
                     size * sizeof(T));
}

}  // namespace phi

PD_REGISTER_KERNEL(randint_with_seed,
                   XPU,
                   ALL_LAYOUT,
                   phi::RandintWithSeedKernel,
                   int,
                   int64_t) {}
