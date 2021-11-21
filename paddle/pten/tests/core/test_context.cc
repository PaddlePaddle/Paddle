/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <algorithm>
#include <vector>

#include "gtest/gtest.h"
#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/pten/core/context.h"
#include "paddle/pten/tests/core/allocator.h"
#include "paddle/pten/tests/core/random.h"
#include "paddle/pten/tests/core/timer.h"

#include "paddle/fluid/platform/place.h"

namespace pten {
namespace tests {

TEST(context, cpu_context) {
  paddle::platform::CPUPlace place;
  CPUContext context(place);

  auto& pool = paddle::platform::DeviceContextPool::Instance();
  auto* fluid_ctx = pool.GetByPlace(place);

  context.SetEigenDevice(fluid_ctx->eigen_device());
  context.GetPlace();
  context.eigen_device();

  std::shared_ptr<Allocator> fancy_allocator(new FancyAllocator);
  context.SetAllocator(fancy_allocator.get());
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
TEST(context, gpu_context) {
  paddle::platform::CUDAPlace place;
  CUDAContext context(place);

  auto& pool = paddle::platform::DeviceContextPool::Instance();
  auto* fluid_ctx = pool.GetByPlace(place);

  context.SetStream(fluid_ctx->stream());
  context.SetSMCount(fluid_ctx->GetSMCount());
  context.SetComputeCapability(fluid_ctx->GetComputeCapability());
  context.SetCUDAMaxGridDimX(fluid_ctx->GetCUDAMaxGridDimSize().x);
  context.SetCUDAMaxGridDimY(fluid_ctx->GetCUDAMaxGridDimSize().y);
  context.SetCUDAMaxGridDimZ(fluid_ctx->GetCUDAMaxGridDimSize().z);
  context.SetMaxThreadsPerBlock(fluid_ctx->GetMaxThreadsPerBlock());

  context.SetCublasHandle(fluid_ctx->cublas_handle());
  context.cublas_handle();

  // only cuda has eigen device.
  // context.SetEigenDevice(fluid_ctx->eigen_device());
  // context.eigen_device();
}
#endif

#ifdef PADDLE_WITH_XPU
TEST(context, xpu_context) {
  paddle::platform::XPUPlace place;
  XPUContext context(place);

  auto& pool = paddle::platform::DeviceContextPool::Instance();
  auto* fluid_ctx = pool.GetByPlace(place);

  context.SetContext(fluid_ctx->x_context());
  context.x_context();
}
#endif

}  // namespace tests
}  // namespace pten
