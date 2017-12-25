/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "gtest/gtest.h"
#include "paddle/platform/device_context.h"

#include "glog/logging.h"

TEST(Device, Init) {
  using paddle::platform::DeviceContext;
  using paddle::platform::CUDADeviceContext;
  using paddle::platform::CUDAPlace;

  int count = paddle::platform::GetCUDADeviceCount();
  for (int i = 0; i < count; i++) {
    CUDADeviceContext* device_context = new CUDADeviceContext(CUDAPlace(i));
    Eigen::GpuDevice* gpu_device = device_context->eigen_device();
    ASSERT_NE(nullptr, gpu_device);
    delete device_context;
  }
}

TEST(Device, CUDADeviceContext) {
  using paddle::platform::CUDADeviceContext;
  using paddle::platform::CUDAPlace;

  int count = paddle::platform::GetCUDADeviceCount();
  for (int i = 0; i < count; i++) {
    CUDADeviceContext* device_context = new CUDADeviceContext(CUDAPlace(i));
    Eigen::GpuDevice* gpu_device = device_context->eigen_device();
    ASSERT_NE(nullptr, gpu_device);
    cudnnHandle_t cudnn_handle = device_context->cudnn_handle();
    ASSERT_NE(nullptr, cudnn_handle);
    cublasHandle_t cublas_handle = device_context->cublas_handle();
    ASSERT_NE(nullptr, cublas_handle);
    ASSERT_NE(nullptr, device_context->stream());
    delete device_context;
  }
}

TEST(Device, CUDNNDeviceContext) {
  using paddle::platform::CUDNNDeviceContext;
  using paddle::platform::CUDAPlace;
  if (paddle::platform::dynload::HasCUDNN()) {
    int count = paddle::platform::GetCUDADeviceCount();
    for (int i = 0; i < count; ++i) {
      CUDNNDeviceContext* device_context = new CUDNNDeviceContext(CUDAPlace(i));
      cudnnHandle_t cudnn_handle = device_context->cudnn_handle();
      ASSERT_NE(nullptr, cudnn_handle);
      ASSERT_NE(nullptr, device_context->stream());
      delete device_context;
    }
  }
}

TEST(Device, DeviceContextPool) {
  using paddle::platform::DeviceContextPool;
  using paddle::platform::CUDADeviceContext;
  using paddle::platform::Place;
  using paddle::platform::CPUPlace;
  using paddle::platform::CUDAPlace;

  DeviceContextPool& pool = DeviceContextPool::Get();
  auto cpu_dev_ctx1 = pool.Borrow(CPUPlace());
  auto cpu_dev_ctx2 = pool.Borrow(CPUPlace());
  EXPECT_TRUE(cpu_dev_ctx2 == cpu_dev_ctx1);

  std::vector<Place> gpu_places;
  int count = paddle::platform::GetCUDADeviceCount();
  for (int i = 0; i < count; ++i) {
    gpu_places.emplace_back(CUDAPlace(i));
  }
  auto dev_ctxs = pool.Borrow(gpu_places);
  for (size_t i = 0; i < dev_ctxs.size(); ++i) {
    auto* dev_ctx = static_cast<const CUDADeviceContext*>(dev_ctxs[i]);

    // check same as CUDAPlace(i)
    CUDAPlace place = boost::get<CUDAPlace>(dev_ctx->GetPlace());
    EXPECT_EQ(place.GetDeviceId(), static_cast<int>(i));
  }
}

int main(int argc, char** argv) {
  int dev_count = paddle::platform::GetCUDADeviceCount();
  if (dev_count <= 1) {
    LOG(WARNING) << "Cannot test multi-gpu DeviceContextPool, because the CUDA "
                    "device count is "
                 << dev_count;
    return 0;
  }

  std::vector<paddle::platform::Place> places;

  places.emplace_back(paddle::platform::CPUPlace());
  int count = paddle::platform::GetCUDADeviceCount();
  for (int i = 0; i < count; ++i) {
    places.emplace_back(paddle::platform::CUDAPlace(i));
  }

  VLOG(0) << " DeviceCount " << count;
  paddle::platform::DeviceContextPool::Create(places);

  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
