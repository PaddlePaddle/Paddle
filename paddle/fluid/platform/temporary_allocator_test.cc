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
#include <string>
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/tensor_util.h"

DECLARE_double(limit_of_temporary_allocation);

namespace paddle {
namespace platform {

class DummyOp : public framework::OperatorBase {
 public:
  DummyOp(const std::string& type, const framework::VariableNameMap& inputs,
          const framework::VariableNameMap& outputs,
          const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

 protected:
  void RunImpl(const framework::Scope& scope,
               const platform::Place& place) const override {}
};

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
  const double limit = FLAGS_limit_of_temporary_allocation;
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
  FLAGS_limit_of_temporary_allocation = limit;
#endif
}

TEST(temporary_allocator, create_tensor_with_allocationptr) {
  framework::VariableNameMap dummy_vars;
  framework::AttributeMap dummy_attrs;
  DummyOp op("dummy", dummy_vars, dummy_vars, dummy_attrs);
  framework::Scope scope;
  framework::VariableValueMap vars;
  framework::RuntimeContext run_ctx(vars, vars);
  size_t memory_size = 300;
  {
    platform::CPUPlace cpu_place;
    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    auto* dev_ctx =
        static_cast<platform::CPUDeviceContext*>(pool.Get(cpu_place));
    framework::ExecutionContext ctx(op, scope, *dev_ctx, run_ctx);

    int numel = memory_size / sizeof(float);
    framework::Tensor tensor =
        ctx.AllocateTmpTensor<float, platform::CPUDeviceContext>(
            framework::make_ddim({numel}), *dev_ctx);
    PADDLE_ENFORCE_EQ(tensor.numel(), numel);
  }

#ifdef PADDLE_WITH_CUDA
  {
    platform::CUDAPlace gpu_place(0);
    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    auto* dev_ctx =
        static_cast<platform::CUDADeviceContext*>(pool.Get(gpu_place));
    framework::ExecutionContext ctx(op, scope, *dev_ctx, run_ctx);
    int numel = memory_size / sizeof(float);
    framework::Tensor tensor =
        ctx.AllocateTmpTensor<float, platform::CUDADeviceContext>(
            framework::make_ddim({numel}), *dev_ctx);
    PADDLE_ENFORCE_EQ(tensor.numel(), numel);
  }
#endif
}

TEST(temporary_allocator, create_tensor_with_allocationptr2) {
  framework::VariableNameMap dummy_vars;
  framework::AttributeMap dummy_attrs;
  DummyOp op("dummy", dummy_vars, dummy_vars, dummy_attrs);
  framework::Scope scope;
  framework::VariableValueMap vars;
  framework::RuntimeContext run_ctx(vars, vars);
  size_t memory_size = 400;
  {
    platform::CPUPlace cpu_place;
    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    auto* dev_ctx =
        static_cast<platform::CPUDeviceContext*>(pool.Get(cpu_place));
    framework::ExecutionContext ctx(op, scope, *dev_ctx, run_ctx);
    int numel = memory_size / sizeof(float);

    framework::Tensor out_side_tensor;
    {
      framework::Tensor tensor =
          ctx.AllocateTmpTensor<float, platform::CPUDeviceContext>(
              framework::make_ddim({numel}), *dev_ctx);
      PADDLE_ENFORCE_EQ(tensor.numel(), numel);

      out_side_tensor.ShareDataWith(tensor);
    }
    PADDLE_ENFORCE_EQ(out_side_tensor.numel(), numel);
  }

#ifdef PADDLE_WITH_CUDA
  {
    platform::CUDAPlace gpu_place(0);
    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    auto* dev_ctx =
        static_cast<platform::CUDADeviceContext*>(pool.Get(gpu_place));
    framework::ExecutionContext ctx(op, scope, *dev_ctx, run_ctx);

    size_t memory_size = 500;
    int numel = memory_size / sizeof(float);
    framework::Tensor out_side_tensor;
    {
      framework::Tensor tensor =
          ctx.AllocateTmpTensor<float, platform::CUDADeviceContext>(
              framework::make_ddim({numel}), *dev_ctx);
      PADDLE_ENFORCE_EQ(tensor.numel(), numel);

      out_side_tensor.ShareDataWith(tensor);
    }
    PADDLE_ENFORCE_EQ(out_side_tensor.numel(), numel);
  }
#endif
}

}  //  namespace platform
}  //  namespace paddle
