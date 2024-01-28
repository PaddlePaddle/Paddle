/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/phi/common/transform.h"

#include "paddle/common/hostdevice.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/common/memory_utils.h"

template <typename T>
class Scale {
 public:
  explicit Scale(const T& scale) : scale_(scale) {}
  HOSTDEVICE T operator()(const T& a) const { return a * scale_; }

 private:
  T scale_;
};

template <typename T>
class Multiply {
 public:
  HOSTDEVICE T operator()(const T& a, const T& b) const { return a * b; }
};

using phi::CPUContext;
using phi::CPUPlace;
using phi::GPUContext;
using phi::GPUPlace;

using phi::Transform;

TEST(Transform, CPUUnary) {
  CPUContext ctx;
  float buf[4] = {0.1, 0.2, 0.3, 0.4};
  Transform<CPUContext> trans;
  trans(ctx, buf, buf + 4, buf, Scale<float>(10));
  for (int i = 0; i < 4; ++i) {
    ASSERT_NEAR(buf[i], static_cast<float>(i + 1), 1e-5);
  }
}

TEST(Transform, GPUUnary) {
  GPUPlace gpu0(0);
  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
  auto* ctx = reinterpret_cast<phi::GPUContext*>(pool.Get(phi::GPUPlace()));

  float cpu_buf[4] = {0.1, 0.2, 0.3, 0.4};
  auto gpu_allocation = phi::memory_utils::Alloc(gpu0, sizeof(float) * 4);
  float* gpu_buf = static_cast<float*>(gpu_allocation->ptr());
  phi::memory_utils::Copy(
      gpu0, gpu_buf, CPUPlace(), cpu_buf, sizeof(cpu_buf), ctx->stream());
  Transform<phi::GPUContext> trans;
  trans(*ctx, gpu_buf, gpu_buf + 4, gpu_buf, Scale<float>(10));
  ctx->Wait();
  phi::memory_utils::Copy(
      CPUPlace(), cpu_buf, gpu0, gpu_buf, sizeof(cpu_buf), ctx->stream());
  for (int i = 0; i < 4; ++i) {
    ASSERT_NEAR(cpu_buf[i], static_cast<float>(i + 1), 1e-5);
  }
}

TEST(Transform, CPUBinary) {
  int buf[4] = {1, 2, 3, 4};
  Transform<phi::CPUContext> trans;
  phi::CPUContext ctx;
  trans(ctx, buf, buf + 4, buf, buf, Multiply<int>());
  for (int i = 0; i < 4; ++i) {
    ASSERT_EQ((i + 1) * (i + 1), buf[i]);
  }
}

TEST(Transform, GPUBinary) {
  int buf[4] = {1, 2, 3, 4};
  GPUPlace gpu0(0);
  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
  auto* ctx = reinterpret_cast<phi::GPUContext*>(pool.Get(phi::GPUPlace()));

  auto gpu_allocation = phi::memory_utils::Alloc(gpu0, sizeof(buf));
  int* gpu_buf = static_cast<int*>(gpu_allocation->ptr());
  phi::memory_utils::Copy(
      gpu0, gpu_buf, CPUPlace(), buf, sizeof(buf), ctx->stream());
  Transform<phi::GPUContext> trans;
  trans(*ctx, gpu_buf, gpu_buf + 4, gpu_buf, gpu_buf, Multiply<int>());
  ctx->Wait();
  phi::memory_utils::Copy(
      CPUPlace(), buf, gpu0, gpu_buf, sizeof(buf), ctx->stream());
  for (int i = 0; i < 4; ++i) {
    ASSERT_EQ((i + 1) * (i + 1), buf[i]);
  }
}
