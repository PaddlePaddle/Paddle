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

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/kernels/randperm_kernel.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void RandpermKernel(const Context& ctx,
                    int n,
                    DataType dtype,
                    DenseTensor* out) {
  DenseTensor tmp;
  tmp.Resize(phi::make_ddim({n}));
  T* tmp_data = ctx.template HostAlloc<T>(&tmp);

  auto gen_ptr = ctx.GetHostGenerator();
  auto engine = gen_ptr->GetCPUEngine();

  for (int i = 0; i < n; ++i) {
    tmp_data[i] = static_cast<T>(i);
  }
  std::shuffle(tmp_data, tmp_data + n, *engine);

  T* out_data = ctx.template Alloc<T>(out);
  auto size = out->numel() * paddle::experimental::SizeOf(out->dtype());
  paddle::memory::Copy(out->place(), out_data, tmp.place(), tmp_data, size);
}

}  // namespace phi

PD_REGISTER_KERNEL(randperm,
                   GPU,
                   ALL_LAYOUT,
                   phi::RandpermKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
