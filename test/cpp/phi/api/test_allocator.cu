/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <gtest/gtest.h>

#include "paddle/phi/api/include/context_pool.h"

#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/common/transform.h"
#include "paddle/phi/core/allocator.h"
#include "paddle/phi/core/device_context.h"

using phi::memory_utils::Copy;

template <typename T>
class Scale {
 public:
  explicit Scale(const T& scale) : scale_(scale) {}
  HOSTDEVICE T operator()(const T& a) const { return a * scale_; }

 private:
  T scale_;
};

TEST(Allocator, CPU) {
  phi::Allocator* allocator = paddle::GetAllocator(phi::CPUPlace());
  auto cpu_allocation = allocator->Allocate(sizeof(float) * 4);
  float* cpu_buf = static_cast<float*>(cpu_allocation->ptr());
  ASSERT_NE(cpu_buf, nullptr);
  cpu_buf[0] = 1.0f;
  cpu_buf[1] = 2.0f;
  cpu_buf[2] = 3.0f;
  cpu_buf[3] = 4.0f;
  for (size_t i = 0; i < 4; ++i) {
    cpu_buf[i] = cpu_buf[i] + 1;
  }
  for (size_t i = 0; i < 4; ++i) {
    ASSERT_NEAR(cpu_buf[i], static_cast<float>(2.0 + i), 1e-5);
  }
}

TEST(Allocator, GPU) {
  phi::GPUPlace gpu0(0);
  float cpu_buf[4] = {0.1, 0.2, 0.3, 0.4};
  phi::Allocator* allocator = paddle::GetAllocator(gpu0);
  auto gpu_allocation = allocator->Allocate(sizeof(cpu_buf));
  float* gpu_buf = static_cast<float*>(gpu_allocation->ptr());

  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
  auto* ctx = reinterpret_cast<phi::GPUContext*>(pool.Get(gpu0));
  Copy(gpu0, gpu_buf, phi::CPUPlace(), cpu_buf, sizeof(cpu_buf), ctx->stream());
  phi::Transform<phi::GPUContext> trans;
  trans(*ctx, gpu_buf, gpu_buf + 4, gpu_buf, Scale<float>(10));
  ctx->Wait();
  Copy(phi::CPUPlace(), cpu_buf, gpu0, gpu_buf, sizeof(cpu_buf), ctx->stream());
  for (int i = 0; i < 4; ++i) {
    ASSERT_NEAR(cpu_buf[i], static_cast<float>(i + 1), 1e-5);
  }
}
