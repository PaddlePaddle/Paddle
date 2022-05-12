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

#include <gtest/gtest.h>

#include "paddle/infrt/kernel/phi/infershaped/infershaped_kernel_launcher.h"
#include "paddle/infrt/kernel/phi/infershaped/infershaped_kernel_launchers.h"
#include "paddle/infrt/kernel/phi/infershaped/infershaped_utils.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/meta_tensor.h"

namespace infrt {
namespace kernel {

namespace {
static void ElementwiseAddTest(const ::phi::DenseTensor& a,
                               const ::phi::DenseTensor& b,
                               ::phi::DenseTensor* c);
}

TEST(utils, registry) {
  constexpr uint8_t count =
      InferShapeHelper<decltype(&ElementwiseAddTest)>::count;
  CHECK_EQ(count, 2U);
}

class FancyAllocator : public ::phi::Allocator {
 public:
  static void Delete(::phi::Allocation* allocation) {
    ::operator delete(allocation->ptr());
  }

  AllocationPtr Allocate(size_t bytes_size) override {
    void* data = ::operator new(bytes_size);
    auto* allocation =
        new ::phi::Allocation(data, bytes_size, ::phi::CPUPlace());
    return AllocationPtr(allocation, Delete);
  }
};

TEST(ElementwiseAdd, launcher_registry) {
  host_context::KernelRegistry registry;
  RegisterInferShapeLaunchers(&registry);
  ASSERT_GE(registry.size(), 1UL);
  auto creator = registry.GetKernel("phi_cpu.add.float32.any");

  const ::phi::DDim dims({1, 2});
  const ::phi::DataType dtype{::phi::DataType::FLOAT32};
  const ::phi::DataLayout layout{::phi::DataLayout::NHWC};
  const ::phi::LoD lod{};
  ::phi::DenseTensorMeta meta(dtype, dims, layout, lod);

  auto fancy_allocator = std::unique_ptr<::phi::Allocator>(new FancyAllocator);
  auto* alloc = fancy_allocator.get();

  ::phi::DenseTensor a(alloc, meta);
  ::phi::DenseTensor b(alloc, meta);
  ::phi::DenseTensor c(alloc, meta);

  auto place = ::phi::CPUPlace();
  float* a_data = a.mutable_data<float>(place);
  float* b_data = b.mutable_data<float>(place);
  float* c_data = c.mutable_data<float>(place);
  for (size_t i = 0; i < 2; ++i) {
    a_data[i] = 1.f;
    b_data[i] = 2.f;
  }

  ::phi::CPUContext context;
  context.SetAllocator(alloc);
  context.Init();

  host_context::KernelFrameBuilder kernel_frame_builder;
  kernel_frame_builder.AddArgument(new host_context::Value(std::move(context)));
  kernel_frame_builder.AddArgument(new host_context::Value(std::move(a)));
  kernel_frame_builder.AddArgument(new host_context::Value(std::move(b)));
  kernel_frame_builder.SetResults({new host_context::Value(std::move(c))});

  creator(&kernel_frame_builder);

  for (size_t i = 0; i < 2; ++i) {
    CHECK_EQ(c_data[i], 3.f);
  }
}

}  // namespace kernel
}  // namespace infrt
