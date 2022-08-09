/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include "cuda.h"          // NOLINT
#include "cuda_runtime.h"  // NOLINT
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "paddle/fluid/memory/allocation/allocator_facade.h"
#include "paddle/fluid/platform/cuda_graph_with_memory_pool.h"
#include "paddle/fluid/platform/device_context.h"

TEST(Device, DeviceContextWithCUDAGraph) {
  using paddle::platform::CUDAPlace;
  using paddle::platform::DeviceContext;
  using paddle::platform::DeviceContextPool;
  using paddle::platform::Place;
  using phi::GPUContext;

  DeviceContextPool& pool = DeviceContextPool::Instance();
  Place place = CUDAPlace(0);
  auto* dev_ctx = pool.Get(place);

  paddle::platform::BeginCUDAGraphCapture(
      place, cudaStreamCaptureMode::cudaStreamCaptureModeThreadLocal, 0);
  ASSERT_EQ(dev_ctx->IsCUDAGraphAllocatorValid(), true);
  dev_ctx->GetCUDAGraphAllocator();
  paddle::platform::EndCUDAGraphCapture();
  ASSERT_EQ(dev_ctx->IsCUDAGraphAllocatorValid(), false);
}
