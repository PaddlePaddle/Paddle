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

#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/impl/seed_kernel_impl.h"

namespace phi {
template <typename T, typename Context>
void GPUSeedKernel(const Context &dev_ctx,
                   int seed_in,
                   bool deterministic,
                   const std::string &rng_name,
                   bool force_cpu,
                   DenseTensor *out) {
  int seed = get_seed(seed_in, deterministic, rng_name);

  bool cpu_place = force_cpu || dev_ctx.GetPlace() == phi::CPUPlace();
  if (cpu_place) {
    phi::DeviceContextPool &pool = phi::DeviceContextPool::Instance();
    auto &dev_ctx_cpu = *pool.Get(phi::CPUPlace());
    dev_ctx_cpu.Alloc<T>(out);
    phi::funcs::SetConstant<phi::CPUContext, T> functor;
    functor(reinterpret_cast<const phi::CPUContext &>(dev_ctx_cpu),
            out,
            static_cast<T>(seed));
  } else {
    auto *out_data = dev_ctx.template Alloc<T>(out);
    auto stream = dev_ctx.stream();
    phi::memory_utils::Copy(dev_ctx.GetPlace(),
                            out_data,
                            phi::CPUPlace(),
                            &seed,
                            sizeof(int),
                            stream);
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(seed, GPU, ALL_LAYOUT, phi::GPUSeedKernel, int) {}
