// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/platform/temporary_allocator.h"
#include <gtest/gtest.h>
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/create_tensor_with_allocationptr.h"
DECLARE_double(limit_of_temporary_allocation);

namespace paddle {
namespace platform {

TEST(temporary_allocator, temporary_allocator) {
  platform::CPUPlace cpu_place;
  TemporaryAllocator alloc(cpu_place);
  alloc.Allocate(100);

#ifdef PADDLE_WITH_CUDA
  platform::CUDAPlace gpu_place(0);
  TemporaryAllocator gpu_alloc(gpu_place);

  auto allocation = gpu_alloc.Allocate(101);
  PADDLE_ENFORCE_EQ(gpu_alloc.TemporaryAllocationQueueSize(), 0);
  gpu_alloc.Release([]() {});
  PADDLE_ENFORCE_EQ(gpu_alloc.TemporaryAllocationQueueSize(), 0);

  {
    auto allocation = gpu_alloc.Allocate(102);
    PADDLE_ENFORCE_EQ(gpu_alloc.TemporaryAllocationQueueSize(), 0);
  }
  PADDLE_ENFORCE_EQ(gpu_alloc.TemporaryAllocationQueueSize(), 1);
  gpu_alloc.Release([]() {});
  PADDLE_ENFORCE_EQ(gpu_alloc.TemporaryAllocationQueueSize(), 0);
#endif
}

TEST(temporary_allocator, add_callback) {
#ifdef PADDLE_WITH_CUDA
  FLAGS_limit_of_temporary_allocation = 10;
  platform::CUDAPlace gpu_place(0);
  TemporaryAllocator gpu_alloc(gpu_place);

  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto* dev_ctx =
      static_cast<platform::CUDADeviceContext*>(pool.Get(gpu_place));
  auto stream = dev_ctx->stream();
  bool deleted = false;
  gpu_alloc.SetCallback([stream, &deleted]() {
    PADDLE_ENFORCE(cudaStreamSynchronize(stream));
    PADDLE_ENFORCE(cudaGetLastError());
    deleted = true;
  });
  { gpu_alloc.Allocate(100); }
  PADDLE_ENFORCE(deleted);
  FLAGS_limit_of_temporary_allocation = -1;
#endif
}

TEST(temporary_allocator, create_tensor_with_allocationptr) {
  platform::CPUPlace cpu_place;
  TemporaryAllocator cpu_alloc(cpu_place);
  {
    size_t memory_size = 200;
    auto allocation = cpu_alloc.Allocate(memory_size);
    void* address = allocation->ptr();
    int numel = memory_size / sizeof(float);
    framework::Tensor tensor =
        GetTensor<float>(std::move(allocation), framework::make_ddim({numel}));
    PADDLE_ENFORCE_EQ(address, tensor.data<float>());
    PADDLE_ENFORCE_EQ(tensor.numel(), numel);
  }

#ifdef PADDLE_WITH_CUDA
  platform::CUDAPlace gpu_place(0);
  TemporaryAllocator gpu_alloc(gpu_place);

  {
    size_t memory_size = 300;
    auto allocation = gpu_alloc.Allocate(memory_size);
    void* address = allocation->ptr();
    int numel = memory_size / sizeof(float);
    framework::Tensor tensor =
        GetTensor<float>(std::move(allocation), framework::make_ddim({numel}));
    PADDLE_ENFORCE_EQ(address, tensor.data<float>());
    PADDLE_ENFORCE_EQ(tensor.numel(), numel);
  }

  // The allocation is not holded now, it should be placed to
  // TemporaryAllocationQueue.
  PADDLE_ENFORCE_EQ(gpu_alloc.TemporaryAllocationQueueSize(), 1);
  gpu_alloc.Release([]() {});
  PADDLE_ENFORCE_EQ(gpu_alloc.TemporaryAllocationQueueSize(), 0);
#endif
}

TEST(temporary_allocator, create_tensor_with_allocationptr2) {
  platform::CPUPlace cpu_place;
  TemporaryAllocator cpu_alloc(cpu_place);
  {
    size_t memory_size = 400;
    int numel = memory_size / sizeof(float);

    framework::Tensor out_side_tensor;
    void* address;
    {
      auto allocation = cpu_alloc.Allocate(memory_size);
      address = allocation->ptr();
      framework::Tensor tensor = GetTensor<float>(
          std::move(allocation), framework::make_ddim({numel}));
      PADDLE_ENFORCE_EQ(address, tensor.data<float>());
      PADDLE_ENFORCE_EQ(tensor.numel(), numel);

      out_side_tensor.ShareDataWith(tensor);
    }
    PADDLE_ENFORCE_EQ(address, out_side_tensor.data<float>());
    PADDLE_ENFORCE_EQ(out_side_tensor.numel(), numel);
  }

#ifdef PADDLE_WITH_CUDA
  platform::CUDAPlace gpu_place(0);
  TemporaryAllocator gpu_alloc(gpu_place);
  {
    void* address;
    size_t memory_size = 500;
    int numel = memory_size / sizeof(float);
    framework::Tensor out_side_tensor;
    {
      auto allocation = gpu_alloc.Allocate(memory_size);
      address = allocation->ptr();
      framework::Tensor tensor = GetTensor<float>(
          std::move(allocation), framework::make_ddim({numel}));
      PADDLE_ENFORCE_EQ(address, tensor.data<float>());
      PADDLE_ENFORCE_EQ(tensor.numel(), numel);

      out_side_tensor.ShareDataWith(tensor);
    }
    PADDLE_ENFORCE_EQ(address, out_side_tensor.data<float>());
    PADDLE_ENFORCE_EQ(out_side_tensor.numel(), numel);
    // The allocation is holded by out_side_tensor.
    PADDLE_ENFORCE_EQ(gpu_alloc.TemporaryAllocationQueueSize(), 0);
    gpu_alloc.Release([]() {});
    PADDLE_ENFORCE_EQ(gpu_alloc.TemporaryAllocationQueueSize(), 0);
  }

  // The allocation is not holded now, it should be placed to
  // TemporaryAllocationQueue.
  PADDLE_ENFORCE_EQ(gpu_alloc.TemporaryAllocationQueueSize(), 1);
  gpu_alloc.Release([]() {});
  PADDLE_ENFORCE_EQ(gpu_alloc.TemporaryAllocationQueueSize(), 0);
#endif
}

}  //  namespace platform
}  //  namespace paddle
