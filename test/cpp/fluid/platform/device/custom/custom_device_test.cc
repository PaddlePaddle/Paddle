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

#include <array>
#include <string>

#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/memory/allocation/allocator_facade.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/init.h"
#include "paddle/phi/backends/custom/fake_cpu_device.h"
#include "paddle/phi/backends/device_manager.h"
#include "paddle/phi/common/memory_utils.h"

void RegisterDevice() {
  CustomRuntimeParams runtime_params;
  runtime_params.size = sizeof(CustomRuntimeParams);
  auto device_interface = std::make_unique<C_DeviceInterface>();
  runtime_params.interface = device_interface.get();
  std::memset(runtime_params.interface, 0, sizeof(C_DeviceInterface));
  runtime_params.interface->size = sizeof(C_DeviceInterface);

  InitFakeCPUDevice(&runtime_params);
  phi::LoadCustomRuntimeLib(
      runtime_params, std::move(device_interface), "", nullptr);
}

void InitDevice() {
  RegisterDevice();
  EXPECT_GT(static_cast<int>(phi::DeviceManager::GetAllDeviceTypes().size()),
            0);
  auto place = phi::CustomPlace(DEVICE_TYPE, 0);
  auto device = phi::DeviceManager::GetDeviceWithPlace(place);
  EXPECT_NE(device, nullptr);

  std::vector<phi::Place> places;
  auto device_types = phi::DeviceManager::GetAllDeviceTypes();
  for (auto dev_type : device_types) {
    auto devices = phi::DeviceManager::GetDeviceList(dev_type);
    for (auto dev_id : devices) {
      places.push_back(phi::PlaceHelper::CreatePlace(dev_type, dev_id));
    }
  }
  EXPECT_GT(static_cast<int>(places.size()), 0);

  phi::DeviceContextPool::Init(places);
}

void TestDeviceInterface(const phi::Place& place) {
  std::cout << "TestDeviceInterface on " << place << std::endl;
  if (paddle::platform::is_custom_place(place)) {
    auto device = phi::DeviceManager::GetDeviceWithPlace(place);
    auto dev_type = phi::PlaceHelper::GetDeviceType(place);
    auto p1 =
        device->MemoryAllocate(phi::DeviceManager::GetMinChunkSize(place));
    EXPECT_NE(p1, nullptr);

    phi::DeviceManager::SetDevice(place);
    auto dev_id = phi::DeviceManager::GetDevice(dev_type);
    EXPECT_EQ(dev_id, place.GetDeviceId());
  }
}

void TestTensorMutableData(const phi::Place& place) {
  std::cout << "TestTensorInitialization on " << place << std::endl;
  phi::DenseTensor src_tensor;
  float* p1 = nullptr;
  float* p2 = nullptr;
  // initialization
  p1 = src_tensor.mutable_data<float>(common::make_ddim({1, 2, 3}), place);
  auto p1_holder = src_tensor.Holder();
  EXPECT_NE(p1, nullptr);
  // set src_tensor a new dim with large size
  // memory is supposed to be re-allocated
  p2 = src_tensor.mutable_data<float>(common::make_ddim({3, 1024}), place);
  auto p2_holder = src_tensor.Holder();
  EXPECT_NE(p2, nullptr);
  EXPECT_NE(p1_holder.get(), p2_holder.get());
  // set src_tensor a new dim with same size
  // memory block is supposed to be unchanged
  p1 = src_tensor.mutable_data<float>(common::make_ddim({2, 2, 3}), place);
  EXPECT_EQ(p1, p2);
  // set src_tensor a new dim with smaller size
  // memory block is supposed to be unchanged
  p2 = src_tensor.mutable_data<float>(common::make_ddim({2, 2}), place);
  EXPECT_EQ(p1, p2);
}

void TestTensorShareDataWith(const phi::Place& place) {
  std::cout << "TestTensorShareDataWith on " << place << std::endl;
  phi::DenseTensor src_tensor;
  phi::DenseTensor dst_tensor;
  src_tensor.mutable_data<int>(common::make_ddim({2, 3, 4}), place);
  dst_tensor.ShareDataWith(src_tensor);
  ASSERT_EQ(src_tensor.data<int>(), dst_tensor.data<int>());
}

void TestTensorUtils(const phi::Place& place) {
  std::cout << "TestTensorUtils on " << place << std::endl;
  if (paddle::platform::is_custom_place(place) == false) {
    return;
  }
  phi::DenseTensor src_tensor;
  phi::DenseTensor gpu_tensor;
  phi::DenseTensor dst_tensor;

  int* src_ptr =
      src_tensor.mutable_data<int>(common::make_ddim({3, 3}), phi::CPUPlace());

  std::array<int, 9> arr = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  memcpy(src_ptr, arr.data(), 9 * sizeof(int));

  // CPU Tensor to GPU Tensor
  paddle::platform::CustomDeviceContext gpu_ctx(place);
  paddle::framework::TensorCopy(src_tensor, place, gpu_ctx, &gpu_tensor);
#if 0
  // GPU Tensor to CPU Tensor
  auto cpu_place = new phi::CPUPlace();
  paddle::framework::TensorCopy(gpu_tensor, *cpu_place, gpu_ctx, &dst_tensor);

  // Sync before Compare Tensors
  gpu_ctx.Wait();
  const int* dst_ptr = dst_tensor.data<int>();
  EXPECT_NE(src_ptr, dst_ptr);
  for (size_t i = 0; i < 9; ++i) {
    EXPECT_EQ(src_ptr[i], dst_ptr[i]);
  }

  // Copy the same tensor
  paddle::framework::TensorCopy(gpu_tensor, place, gpu_ctx, &gpu_tensor);
  gpu_ctx.Wait();
  const int* dst_ptr_tmp = dst_tensor.data<int>();
  EXPECT_NE(src_ptr, dst_ptr_tmp);
  for (size_t i = 0; i < 9; ++i) {
    EXPECT_EQ(src_ptr[i], dst_ptr_tmp[i]);
  }

  phi::DenseTensor slice_tensor = src_tensor.Slice(1, 2);

  // CPU Slice Tensor to GPU Tensor
  paddle::framework::TensorCopy(slice_tensor, place, gpu_ctx, &gpu_tensor);

  // GPU Tensor to CPU Tensor
  paddle::framework::TensorCopy(gpu_tensor, *cpu_place, gpu_ctx, &dst_tensor);

  // Sync before Compare Slice Tensors
  gpu_ctx.Wait();
  const int* slice_ptr = slice_tensor.data<int>();
  dst_ptr = dst_tensor.data<int>();
  EXPECT_NE(dst_ptr, slice_ptr);
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(dst_ptr[i], slice_ptr[i]);
  }

  EXPECT_TRUE(dst_tensor.layout() == src_tensor.layout());
#endif
}

void TestCustomCCL(const phi::Place& place) {
  std::cout << "TestCustomCCL on " << place << std::endl;
  if (paddle::platform::is_custom_place(place) == false) {
    return;
  }
  std::string dev_type = place.GetDeviceType();
  phi::ccl::CCLComm comm;
  phi::stream::Stream stream(place, nullptr);
  phi::ccl::CCLRootId root_id;

  phi::DeviceManager::CCLDestroyComm(dev_type, nullptr);
  phi::DeviceManager::CCLGetUniqueId(dev_type, &root_id);
  phi::DeviceManager::CCLCommInitRank(dev_type, 0, &root_id, 0, nullptr);
  phi::DeviceManager::CCLBroadcast(
      dev_type, nullptr, 0, phi::DataType::FLOAT32, 0, comm, stream);
  phi::DeviceManager::CCLAllReduce(dev_type,
                                   nullptr,
                                   nullptr,
                                   0,
                                   phi::DataType::FLOAT32,
                                   phi::ccl::CCLReduceOp::SUM,
                                   comm,
                                   stream);
  phi::DeviceManager::CCLReduce(dev_type,
                                nullptr,
                                nullptr,
                                0,
                                phi::DataType::FLOAT32,
                                phi::ccl::CCLReduceOp::SUM,
                                0,
                                comm,
                                stream);
  phi::DeviceManager::CCLAllGather(
      dev_type, nullptr, nullptr, 0, phi::DataType::FLOAT32, comm, stream);
  phi::DeviceManager::CCLReduceScatter(dev_type,
                                       nullptr,
                                       nullptr,
                                       0,
                                       phi::DataType::FLOAT32,
                                       phi::ccl::CCLReduceOp::SUM,
                                       comm,
                                       stream);
  phi::DeviceManager::CCLGroupStart(dev_type);
  phi::DeviceManager::CCLGroupEnd(dev_type);
  phi::DeviceManager::CCLSend(
      dev_type, nullptr, 0, phi::DataType::FLOAT32, 0, comm, stream);
  phi::DeviceManager::CCLRecv(
      dev_type, nullptr, 0, phi::DataType::FLOAT32, 0, comm, stream);
}

TEST(CustomDevice, Tensor) {
  paddle::framework::InitMemoryMethod();
  InitDevice();
  auto dev_types = phi::DeviceManager::GetAllDeviceTypes();
  for (const auto& dev_type : dev_types) {
    std::cout << "Test on " << dev_type << std::endl;
    EXPECT_GT(static_cast<int>(phi::DeviceManager::GetDeviceCount(dev_type)),
              0);
    auto place = phi::PlaceHelper::CreatePlace(dev_type);

    TestDeviceInterface(place);
    TestTensorMutableData(place);
    TestTensorShareDataWith(place);
    TestTensorUtils(place);
    TestCustomCCL(place);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
