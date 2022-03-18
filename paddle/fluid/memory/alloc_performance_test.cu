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

#include <thread>  // NOLINT
#include <vector>

#include "gtest/gtest.h"
#include "paddle/fluid/memory/allocation/allocator_facade.h"
#include "paddle/fluid/memory/memory.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/phi/core/stream.h"

#ifdef PADDLE_WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include "paddle/fluid/platform/cuda_graph_with_memory_pool.h"
#endif

#ifdef PADDLE_WITH_HIP
#include <hip/hip_runtime.h>
#endif

#include "gperftools/profiler.h"

namespace paddle {
namespace memory {

TEST(StreamSafeCUDAAllocInterfaceTest, AllocInterfaceTest) {
  VLOG(1) << "Begin Run";
  std::shared_ptr<Allocation> pre_allocation =
      AllocShared(platform::CUDAPlace(), 100000000);
  pre_allocation.reset();
  ProfilerStart("alloc.prof");
  for (int i = 0; i < 10000000; ++i) {
    size_t alloc_size = rand() % 10000000;
    std::shared_ptr<Allocation> allocation_implicit_stream =
        AllocShared(platform::CUDAPlace(), alloc_size);
    EXPECT_GE(allocation_implicit_stream->size(), alloc_size);
    allocation_implicit_stream.reset();
  }
  ProfilerStop();
}
}  // namespace memory
}  // namespace paddle
