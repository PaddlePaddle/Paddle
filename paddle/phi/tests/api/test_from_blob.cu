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

#include "paddle/phi/api/include/from_blob.h"

#include "paddle/phi/api/include/api.h"
#include "paddle/phi/api/include/context_pool.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"

PD_DECLARE_KERNEL(pow, CPU, ALL_LAYOUT);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_DECLARE_KERNEL(pow, GPU, ALL_LAYOUT);
#endif

namespace paddle {
namespace tests {

using phi::memory_utils::Copy;

TEST(from_blob, CPU) {
  // 1. create data
  int64_t data[] = {4, 3, 2, 1};

  // 2. test API
  auto test_tesnor = experimental::from_blob(
      data, {1, 2, 2}, phi::DataType::INT64, phi::CPUPlace());

  // 3. check result
  // 3.1 check tensor attributes
  ASSERT_EQ(test_tesnor.dims().size(), 3);
  ASSERT_EQ(test_tesnor.dims()[0], 1);
  ASSERT_EQ(test_tesnor.dims()[1], 2);
  ASSERT_EQ(test_tesnor.dims()[2], 2);
  ASSERT_EQ(test_tesnor.numel(), 4);
  ASSERT_EQ(test_tesnor.is_cpu(), true);
  ASSERT_EQ(test_tesnor.dtype(), phi::DataType::INT64);
  ASSERT_EQ(test_tesnor.layout(), phi::DataLayout::NCHW);
  ASSERT_EQ(test_tesnor.is_dense_tensor(), true);

  // 3.2 check tensor values
  auto* test_tensor_data = test_tesnor.template data<int64_t>();
  for (int64_t i = 0; i < 4; i++) {
    ASSERT_EQ(test_tensor_data[i], 4 - i);
  }

  // 3.3 check whether memory is shared
  ASSERT_EQ(data, test_tensor_data);

  // 3.4 test other API
  auto test_tensor_pow = pow(test_tesnor, 2);
  auto* test_tensor_pow_data = test_tensor_pow.template data<int64_t>();
  for (int64_t i = 0; i < 4; i++) {
    ASSERT_EQ(test_tensor_pow_data[i],
              static_cast<int64_t>(std::pow(4 - i, 2)));
  }
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
TEST(from_blob, GPU) {
  // 1. create data
  float cpu_data[6] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
  phi::GPUPlace gpu0(0);
  phi::Allocator* allocator = paddle::GetAllocator(gpu0);
  auto gpu_allocation = allocator->Allocate(sizeof(cpu_data));
  float* gpu_data = static_cast<float*>(gpu_allocation->ptr());
  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
  auto* ctx = reinterpret_cast<phi::GPUContext*>(pool.Get(gpu0));
  Copy(gpu0,
       gpu_data,
       phi::CPUPlace(),
       cpu_data,
       sizeof(cpu_data),
       ctx->stream());

  // 2. test API
  auto gpu_tesnor = experimental::from_blob(
      gpu_data, {2, 3}, phi::DataType::FLOAT32, phi::GPUPlace());

  // 3. check result
  // 3.1 check tensor attributes
  ASSERT_EQ(gpu_tesnor.dims().size(), 2);
  ASSERT_EQ(gpu_tesnor.dims()[0], 2);
  ASSERT_EQ(gpu_tesnor.dims()[1], 3);
  ASSERT_EQ(gpu_tesnor.numel(), 6);
  ASSERT_EQ(gpu_tesnor.is_gpu(), true);
  ASSERT_EQ(gpu_tesnor.dtype(), phi::DataType::FLOAT32);

  // 3.2 check tensor values
  auto* gpu_tesnor_data = gpu_tesnor.template data<float>();
  float gpu_tesnor_data_cpu[6];
  Copy(phi::CPUPlace(),
       gpu_tesnor_data_cpu,
       gpu0,
       gpu_tesnor_data,
       sizeof(cpu_data),
       ctx->stream());
  for (int64_t i = 0; i < 6; i++) {
    ASSERT_NEAR(
        gpu_tesnor_data_cpu[i], static_cast<float>((i + 1) * 0.1f), 1e-5);
  }

  // 3.3 check whether memory is shared
  ASSERT_EQ(gpu_data, gpu_tesnor_data);

  // 3.4 test other API
  auto gpu_tesnor_pow = pow(gpu_tesnor, 2);
  auto* gpu_tesnor_pow_data = gpu_tesnor_pow.template data<float>();
  float gpu_tesnor_pow_data_cpu[6];
  Copy(phi::CPUPlace(),
       gpu_tesnor_pow_data_cpu,
       gpu0,
       gpu_tesnor_pow_data,
       sizeof(cpu_data),
       ctx->stream());
  for (int64_t i = 0; i < 6; i++) {
    ASSERT_NEAR(gpu_tesnor_pow_data_cpu[i],
                static_cast<float>(std::pow(i + 1, 2) * 0.01f),
                1e-5);
  }
}
#endif

TEST(from_blob, Option) {
  // 1. create data
  auto data = new int64_t[8];
  for (int64_t i = 0; i < 8; i++) {
    data[i] = i;
  }

  // 2. test API
  int isdelete = 0;
  auto deleter = [&isdelete](void* data) {
    delete[] static_cast<int64_t*>(data);
    isdelete++;
  };
  {
    size_t offset = sizeof(int64_t) * 4;
    auto test_tesnor = experimental::from_blob(data,
                                               {1, 2, 2, 1},
                                               phi::DataType::INT64,
                                               phi::CPUPlace(),
                                               DataLayout::NHWC,
                                               offset,
                                               deleter);

    // check tensor attributes
    ASSERT_EQ(test_tesnor.dims().size(), 4);
    ASSERT_EQ(test_tesnor.dims()[1], 2);
    ASSERT_EQ(test_tesnor.dims()[2], 2);
    ASSERT_EQ(test_tesnor.numel(), 4);
    ASSERT_EQ(test_tesnor.is_cpu(), true);
    ASSERT_EQ(test_tesnor.dtype(), phi::DataType::INT64);
    ASSERT_EQ(test_tesnor.layout(), phi::DataLayout::NHWC);  // check layout

    // check tensor values
    auto* test_tensor_data = test_tesnor.template data<int64_t>();
    for (int64_t i = 0; i < 4; i++) {
      ASSERT_EQ(test_tensor_data[i], i + 4);
    }

    // check storage_offset
    ASSERT_EQ(data + 4, test_tensor_data);

    // test other API
    auto test_tensor_pow = pow(test_tesnor, 2);
    auto* test_tensor_pow_data = test_tensor_pow.template data<int64_t>();
    for (int64_t i = 0; i < 4; i++) {
      ASSERT_EQ(test_tensor_pow_data[i],
                static_cast<int64_t>(std::pow(i + 4, 2)));
    }

    // check deleter
    ASSERT_EQ(isdelete, 0);
  }
  ASSERT_EQ(isdelete, 1);
}

}  // namespace tests
}  // namespace paddle
