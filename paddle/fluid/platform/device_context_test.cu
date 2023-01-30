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
#include "paddle/fluid/memory/allocation/allocator_facade.h"
#include "paddle/fluid/platform/device_context.h"
<<<<<<< HEAD
#include "paddle/phi/core/dense_tensor.h"
=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

TEST(Device, Init) {
  using paddle::platform::CUDAPlace;
  using paddle::platform::DeviceContext;
  using phi::GPUContext;

  int count = paddle::platform::GetGPUDeviceCount();
  for (int i = 0; i < count; i++) {
    phi::GPUContext* device_context = new phi::GPUContext(CUDAPlace(i));
    device_context->SetAllocator(
        paddle::memory::allocation::AllocatorFacade::Instance()
            .GetAllocator(CUDAPlace(i), device_context->stream())
            .get());
    device_context->SetHostAllocator(
        paddle::memory::allocation::AllocatorFacade::Instance()
            .GetAllocator(paddle::platform::CPUPlace())
            .get());
    device_context->SetZeroAllocator(
        paddle::memory::allocation::AllocatorFacade::Instance()
            .GetZeroAllocator(CUDAPlace(i))
            .get());
<<<<<<< HEAD
    device_context->SetHostZeroAllocator(
        paddle::memory::allocation::AllocatorFacade::Instance()
            .GetZeroAllocator(paddle::platform::CPUPlace())
            .get());
=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    device_context->SetPinnedAllocator(
        paddle::memory::allocation::AllocatorFacade::Instance()
            .GetAllocator(paddle::platform::CUDAPinnedPlace())
            .get());
    device_context->PartialInitWithAllocator();

    Eigen::GpuDevice* gpu_device = device_context->eigen_device();
    ASSERT_NE(nullptr, gpu_device);
    delete device_context;
  }
}

TEST(Device, GPUContext) {
  using paddle::platform::CUDAPlace;
  using phi::GPUContext;

  int count = paddle::platform::GetGPUDeviceCount();
  for (int i = 0; i < count; i++) {
    phi::GPUContext* device_context = new phi::GPUContext(CUDAPlace(i));
    device_context->SetAllocator(
        paddle::memory::allocation::AllocatorFacade::Instance()
            .GetAllocator(CUDAPlace(i), device_context->stream())
            .get());
    device_context->SetHostAllocator(
        paddle::memory::allocation::AllocatorFacade::Instance()
            .GetAllocator(paddle::platform::CPUPlace())
            .get());
    device_context->SetZeroAllocator(
        paddle::memory::allocation::AllocatorFacade::Instance()
            .GetZeroAllocator(CUDAPlace(i))
            .get());
<<<<<<< HEAD
    device_context->SetHostZeroAllocator(
        paddle::memory::allocation::AllocatorFacade::Instance()
            .GetZeroAllocator(paddle::platform::CPUPlace())
            .get());
=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    device_context->SetPinnedAllocator(
        paddle::memory::allocation::AllocatorFacade::Instance()
            .GetAllocator(paddle::platform::CUDAPinnedPlace())
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

<<<<<<< HEAD
TEST(Device, HostZeroAllocator) {
  using paddle::platform::CUDAPlace;

  auto device_context = std::make_unique<phi::GPUContext>(CUDAPlace(0));
  device_context->SetAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(CUDAPlace(0), device_context->stream())
          .get());
  device_context->SetHostAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(paddle::platform::CPUPlace())
          .get());
  device_context->SetZeroAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetZeroAllocator(CUDAPlace(0))
          .get());
  device_context->SetHostZeroAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetZeroAllocator(paddle::platform::CPUPlace())
          .get());
  device_context->SetPinnedAllocator(
      paddle::memory::allocation::AllocatorFacade::Instance()
          .GetAllocator(paddle::platform::CUDAPinnedPlace())
          .get());
  device_context->PartialInitWithAllocator();

  phi::DenseTensor tensor;
  tensor.Resize({0});
  device_context->HostAlloc<float>(&tensor);
  ASSERT_EQ(tensor.place().GetType(), phi::AllocationType::CPU);
  ASSERT_EQ(tensor.numel(), 0);
  ASSERT_EQ(tensor.dtype(), phi::DataType::FLOAT32);

  phi::GPUContext gpu_context(CUDAPlace(0));
  gpu_context.SetHostZeroAllocator(&device_context->GetHostZeroAllocator());
  gpu_context.HostAlloc<float>(&tensor);
  ASSERT_EQ(tensor.place().GetType(), phi::AllocationType::CPU);
}

=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
TEST(Device, DeviceContextPool) {
  using paddle::platform::CPUPlace;
  using paddle::platform::CUDAPlace;
  using paddle::platform::DeviceContextPool;
  using paddle::platform::Place;
  using phi::GPUContext;

  DeviceContextPool& pool = DeviceContextPool::Instance();
  auto cpu_dev_ctx1 = pool.Get(CPUPlace());
  auto cpu_dev_ctx2 = pool.Get(CPUPlace());
  ASSERT_EQ(cpu_dev_ctx2, cpu_dev_ctx1);

  std::vector<Place> gpu_places;
  int count = paddle::platform::GetGPUDeviceCount();
  for (int i = 0; i < count; ++i) {
    auto dev_ctx = pool.Get(CUDAPlace(i));
    ASSERT_NE(dev_ctx, nullptr);
  }
}
