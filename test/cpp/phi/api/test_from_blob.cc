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

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "paddle/phi/api/include/api.h"
#include "paddle/phi/api/include/tensor_utils.h"
#include "paddle/phi/core/kernel_registry.h"

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include "paddle/phi/api/include/context_pool.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/common/memory_utils.h"
#endif

PD_DECLARE_KERNEL(pow, CPU, ALL_LAYOUT);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_DECLARE_KERNEL(pow, GPU, ALL_LAYOUT);
#endif

using paddle::from_blob;
using phi::DataType;

namespace paddle {
phi::Place GetPlaceFromPtr(void* data);
}  // namespace paddle

TEST(from_blob, CPU) {
  // 1. create data
  int64_t data[] = {4, 3, 2, 1};  // NOLINT

  ASSERT_EQ(paddle::GetPlaceFromPtr(data), phi::CPUPlace());

  // 2. test API
  auto test_tensor = from_blob(data, {1, 2, 2}, DataType::INT64);

  // 3. check result
  // 3.1 check tensor attributes
  ASSERT_EQ(test_tensor.dims().size(), 3);
  ASSERT_EQ(test_tensor.dims()[0], 1);
  ASSERT_EQ(test_tensor.dims()[1], 2);
  ASSERT_EQ(test_tensor.dims()[2], 2);
  ASSERT_EQ(test_tensor.numel(), 4);
  ASSERT_EQ(test_tensor.is_cpu(), true);
  ASSERT_EQ(test_tensor.dtype(), DataType::INT64);
  ASSERT_EQ(test_tensor.layout(), phi::DataLayout::NCHW);
  ASSERT_EQ(test_tensor.is_dense_tensor(), true);

  // 3.2 check tensor values
  auto* test_tensor_data = test_tensor.template data<int64_t>();
  for (int64_t i = 0; i < 4; i++) {
    ASSERT_EQ(test_tensor_data[i], 4 - i);
  }

  // 3.3 check whether memory is shared
  ASSERT_EQ(data, test_tensor_data);

  // 3.4 test other API
  auto test_tensor_pow = paddle::experimental::pow(test_tensor, 2);
  auto* test_tensor_pow_data = test_tensor_pow.template data<int64_t>();
  for (int64_t i = 0; i < 4; i++) {
    ASSERT_EQ(test_tensor_pow_data[i],
              static_cast<int64_t>(std::pow(4 - i, 2)));
  }
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

using phi::memory_utils::Copy;

TEST(GetPlaceFromPtr, GPU) {
  using paddle::GetPlaceFromPtr;

  std::array<float, 6> cpu_data = {};
  auto cpu_data_place = GetPlaceFromPtr(cpu_data.data());
  ASSERT_EQ(cpu_data_place, phi::CPUPlace());
  std::cout << "cpu_data_place: " << cpu_data_place << std::endl;

  auto alloc_ptr =
      paddle::GetAllocator(phi::GPUPlace(0))->Allocate(sizeof(cpu_data));
  float* gpu0_data = static_cast<float*>(alloc_ptr->ptr());
  auto gpu0_data_place = GetPlaceFromPtr(gpu0_data);
  ASSERT_EQ(gpu0_data_place, phi::GPUPlace(0));
  std::cout << "gpu0_data_place: " << gpu0_data_place << std::endl;
  alloc_ptr.release();

  if (phi::backends::gpu::GetGPUDeviceCount() > 1) {
    float* gpu1_data =
        static_cast<float*>(paddle::GetAllocator(phi::GPUPlace(1))
                                ->Allocate(sizeof(cpu_data))
                                ->ptr());
    auto gpu1_data_place = GetPlaceFromPtr(gpu1_data);
    ASSERT_EQ(gpu1_data_place, phi::GPUPlace(1));
    std::cout << "gpu1_data_place: " << gpu1_data_place << std::endl;
  }
}

TEST(from_blob, GPU) {
  // 1. create data
  std::array<float, 6> cpu_data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
  phi::GPUPlace gpu0(0);
  phi::Allocator* allocator = paddle::GetAllocator(gpu0);
  auto gpu_allocation = allocator->Allocate(sizeof(cpu_data));
  float* gpu_data = static_cast<float*>(gpu_allocation->ptr());
  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
  auto* ctx = reinterpret_cast<phi::GPUContext*>(pool.Get(gpu0));
  Copy(gpu0,
       gpu_data,
       phi::CPUPlace(),
       cpu_data.data(),
       sizeof(cpu_data),
       ctx->stream());

  // 2. test API
  auto gpu_tensor = from_blob(gpu_data, {2, 3}, DataType::FLOAT32);

  // 3. check result
  // 3.1 check tensor attributes
  ASSERT_EQ(gpu_tensor.dims().size(), 2);
  ASSERT_EQ(gpu_tensor.dims()[0], 2);
  ASSERT_EQ(gpu_tensor.dims()[1], 3);
  ASSERT_EQ(gpu_tensor.numel(), 6);
  // ASSERT_EQ(gpu_tensor.is_gpu(), true);
  ASSERT_EQ(gpu_tensor.dtype(), DataType::FLOAT32);

  // 3.2 check tensor values
  auto* gpu_tensor_data = gpu_tensor.template data<float>();
  std::array<float, 6> gpu_tensor_data_cpu = {};
  Copy(phi::CPUPlace(),
       gpu_tensor_data_cpu.data(),
       gpu0,
       gpu_tensor_data,
       sizeof(cpu_data),
       ctx->stream());
  for (int64_t i = 0; i < 6; i++) {
    ASSERT_NEAR(
        gpu_tensor_data_cpu[i], static_cast<float>((i + 1) * 0.1f), 1e-5);
  }

  // 3.3 check whether memory is shared
  ASSERT_EQ(gpu_data, gpu_tensor_data);

  // 3.4 test other API
  auto gpu_tensor_pow = paddle::experimental::pow(gpu_tensor, 2);
  auto* gpu_tensor_pow_data = gpu_tensor_pow.template data<float>();
  std::array<float, 6> gpu_tensor_pow_data_cpu = {};
  Copy(phi::CPUPlace(),
       gpu_tensor_pow_data_cpu.data(),
       gpu0,
       gpu_tensor_pow_data,
       sizeof(cpu_data),
       ctx->stream());
  for (int64_t i = 0; i < 6; i++) {
    ASSERT_NEAR(gpu_tensor_pow_data_cpu[i],
                static_cast<float>(std::pow(i + 1, 2) * 0.01f),
                1e-5);
  }
}
#endif

TEST(from_blob, Option) {
  int delete_count = 0, f_delete_count = 0;
  auto deleter = [&delete_count](void* data) {
    delete[] static_cast<int64_t*>(data);
    delete_count++;
  };
  auto f_deleter = [&f_delete_count](void* ptr) {
    delete[] static_cast<float*>(ptr);
    f_delete_count++;
  };
  {
    auto data = new int64_t[8];
    for (int64_t i = 0; i < 8; i++) {
      data[i] = i;
    }
    auto test_tensor = from_blob(data,
                                 {1, 2, 2, 2},
                                 DataType::INT64,
                                 phi::DataLayout::NHWC,
                                 phi::CPUPlace(),
                                 deleter);
    ASSERT_EQ(test_tensor.layout(), phi::DataLayout::NHWC);
    ASSERT_EQ(delete_count, 0);

    auto f_data = new float[8];
    for (int i = 0; i < 8; i++) {
      f_data[i] = static_cast<float>(i);
    }
    auto test_tensor_f = from_blob(f_data,
                                   {1, 2, 2, 2},
                                   DataType::FLOAT32,
                                   common::DataLayout::NHWC,
                                   phi::CPUPlace(),
                                   f_deleter);
    ASSERT_EQ(test_tensor_f.layout(), phi::DataLayout::NHWC);
    ASSERT_EQ(f_delete_count, 0);
  }
  ASSERT_EQ(delete_count, 1);
  ASSERT_EQ(f_delete_count, 1);
}

TEST(from_blob, Strides) {
  int64_t data[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  auto test_tensor =
      from_blob(data, {1, 2, 2, 1}, {0, 4, 2, 0}, DataType::INT64);
  ASSERT_EQ(test_tensor.shape()[1], 2);
  ASSERT_EQ(test_tensor.shape()[2], 2);
  ASSERT_EQ(test_tensor.strides()[1], 4);
  ASSERT_EQ(test_tensor.strides()[2], 2);
}
