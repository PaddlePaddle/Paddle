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

#include "paddle/phi/kernels/randperm_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void RandpermKernel(const Context& dev_ctx,
                    int n,
                    DataType dtype,
                    DenseTensor* out) {
  std::shared_ptr<std::mt19937_64> engine;
  int seed = 0;
  if (seed) {
    engine = std::make_shared<std::mt19937_64>();
    engine->seed(seed);
  } else {
    engine = dev_ctx.GetGenerator()->GetCPUEngine();
  }

  if (dev_ctx.GetPlace().GetType() == phi::AllocationType::CPU) {
    T* out_data = dev_ctx.template HostAlloc<T>(out);
    for (int i = 0; i < n; ++i) {
      out_data[i] = static_cast<T>(i);
    }
    std::shuffle(out_data, out_data + n, *engine);
  } else {
    dev_ctx.template Alloc<T>(out);
    phi::DenseTensor tmp_tensor;
    tmp_tensor.Resize(common::make_ddim({n}));
    T* tmp_data = dev_ctx.template HostAlloc<T>(&tmp_tensor);
    for (int i = 0; i < n; ++i) {
      tmp_data[i] = static_cast<T>(i);
    }
    std::shuffle(tmp_data, tmp_data + n, *engine);
    Copy(dev_ctx, tmp_tensor, dev_ctx.GetPlace(), true, out);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(randperm,
                   XPU,
                   ALL_LAYOUT,
                   phi::RandpermKernel,
                   int,
                   int64_t,
                   float,
                   double) {}
