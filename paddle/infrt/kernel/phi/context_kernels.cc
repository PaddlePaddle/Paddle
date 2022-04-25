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

#include "paddle/infrt/kernel/phi/context_kernels.h"

namespace infrt {
namespace kernel {
namespace phi {

::phi::CPUContext CreateCPUContext() {
  ::phi::CPUContext ctx{};
  ctx.Init();
  auto allocator = new backends::CpuPhiAllocator{};
  ctx.SetAllocator(allocator);
  ctx.SetHostAllocator(allocator);
  ctx.SetZeroAllocator(allocator);
  return ctx;
}

#ifdef INFRT_WITH_GPU
::phi::GPUContext CreateGPUContext() {
  ::phi::GPUContext context;
  context.PartialInitWithoutAllocator();
  context.SetAllocator(new ::infrt::backends::GpuPhiAllocator{});
  context.SetHostAllocator(new backends::CpuPhiAllocator{});
  context.PartialInitWithAllocator();
  return context;
}
#endif

}  // namespace phi
}  // namespace kernel
}  // namespace infrt
