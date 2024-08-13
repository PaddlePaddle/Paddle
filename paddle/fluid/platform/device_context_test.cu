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
#include <vector>

#include "glog/logging.h"
#include "gtest/gtest.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/memory/allocation/allocator_facade.h"
#include "paddle/phi/core/platform/device_context.h"

TEST(Device, Init) {
  using phi::DeviceContext;
  using phi::GPUContext;
  using phi::GPUPlace;

  int count = paddle::platform::GetGPUDeviceCount();
  for (int i = 0; i < count; i++) {
    phi::GPUContext* device_context = new phi::GPUContext(GPUPlace(i));
    device_context->SetAllocator(
        paddle::memory::allocation::AllocatorFacade::Instance()
            .GetAllocator(GPUPlace(i), device_context->stream())
            .get());
    device_context->SetHostAllocator(
        paddle::memory::allocation::AllocatorFacade::Instance()
            .GetAllocator(phi::CPUPlace())
            .get());
    device_context->SetZeroAllocator(
        paddle::memory::allocation::AllocatorFacade::Instance()
            .GetZeroAllocator(GPUPlace(i))
            .get());
    device_context->SetHostZeroAllocator(
        paddle::memory::allocation::AllocatorFacade::Instance()
            .GetZeroAllocator(phi::CPUPlace())
            .get());
    device_context->SetPinnedAllocator(
        paddle::memory::allocation::AllocatorFacade::Instance()
            .GetAllocator(phi::GPUPinnedPlace())
            .get());
    device_context->PartialInitWithAllocator();

    Eigen::GpuDevice* gpu_device = device_context->eigen_device();
    ASSERT_NE(nullptr, gpu_device);
    delete device_context;
  }
}

TEST(Device, GPUContext) {
  using phi::GPUContext;
  using phi::GPUPlace;

  int count = paddle::platform::GetGPUDeviceCount();
  for (int i = 0; i < count; i++) {
    phi::GPUContext* device_context = new phi::GPUContext(GPUPlace(i));
    device_context->SetAllocator(
        paddle::memory::allocation::AllocatorFacade::Instance()
            .GetAllocator(GPUPlace(i), device_context->stream())
            .get());
    device_context->SetHostAllocator(
        paddle::memory::allocation::AllocatorFacade::Instance()
            .GetAllocator(phi::CPUPlace())
            .get());
    device_context->SetZeroAllocator(
        paddle::memory::allocation::AllocatorFacade::Instance()
            .GetZeroAllocator(GPUPlace(i))
            .get());
    device_context->SetHostZeroAllocator(
        paddle::memory::allocation::AllocatorFacade::Instance()
            .GetZeroAllocator(phi::CPUPlace())
            .get());
    device_context->SetPinnedAllocator(
        paddle::memory::allocation::AllocatorFacade::Instance()
            .GetAllocator(phi::GPUPinnedPlace())
            .get());
    device_context->PartialInitWithAllocator();
    Eigen::GpuDevice* gpu_device = device_context->eigen_device();
    ASSERT_NE(nullptr, gpu_device);
#ifdef PADDLE_WITH_HIP
    miopenHandle_t cudnn_handle = device_context->cudnn_handle();
#else
    cudnnHandle_t cudnn_handle = device_context->cudnn_handle();
#endif
    ASSERT_NE(nullptr, cudnn_handle);
#ifdef PADDLE_WITH_HIP
    rocblas_handle cublas_handle = device_context->cublas_handle();
#else
    cublasHandle_t cublas_handle = device_context->cublas_handle();
#endif
    ASSERT_NE(nullptr, cublas_handle);
    delete device_context;
  }
}

TEST(Device, HostZeroAllocator) {
  using phi::GPUPlace;

  auto device_context = std::make_unique<phi::GPUContext>(GPUPlace(0));
  device_context->SetAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(GPUPlace(0), device_context->stream())
          .get());
  device_context->SetHostAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(phi::CPUPlace())
          .get());
  device_context->SetZeroAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetZeroAllocator(GPUPlace(0))
          .get());
  device_context->SetHostZeroAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetZeroAllocator(phi::CPUPlace())
          .get());
  device_context->SetPinnedAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(phi::GPUPinnedPlace())
          .get());
  device_context->PartialInitWithAllocator();

  phi::DenseTensor tensor;
  tensor.Resize({0});
  device_context->HostAlloc<float>(&tensor);
  ASSERT_EQ(tensor.place().GetType(), phi::AllocationType::CPU);
  ASSERT_EQ(tensor.numel(), 0);
  ASSERT_EQ(tensor.dtype(), phi::DataType::FLOAT32);

  phi::GPUContext gpu_context(GPUPlace(0));
  gpu_context.SetHostZeroAllocator(&device_context->GetHostZeroAllocator());
  gpu_context.HostAlloc<float>(&tensor);
  ASSERT_EQ(tensor.place().GetType(), phi::AllocationType::CPU);
}

TEST(Device, DeviceContextPool) {
  using phi::CPUPlace;
  using phi::DeviceContextPool;
  using phi::GPUContext;
  using phi::GPUPlace;
  using phi::Place;

  DeviceContextPool& pool = DeviceContextPool::Instance();
  auto cpu_dev_ctx1 = pool.Get(CPUPlace());
  auto cpu_dev_ctx2 = pool.Get(CPUPlace());
  ASSERT_EQ(cpu_dev_ctx2, cpu_dev_ctx1);

  std::vector<Place> gpu_places;
  int count = paddle::platform::GetGPUDeviceCount();
  for (int i = 0; i < count; ++i) {
    auto dev_ctx = pool.Get(GPUPlace(i));
    ASSERT_NE(dev_ctx, nullptr);
  }
}
