// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#endif
#ifdef PADDLE_WITH_XPU
#include "paddle/phi/core/platform/device/xpu/xpu_info.h"
#endif

#include <sstream>
#include <unordered_map>

#include "gtest/gtest.h"

TEST(DeviceTypeCompatTest, DeviceTypeConversionAndStreamOperator) {
  EXPECT_EQ(c10::DeviceTypeToPhi(c10::DeviceType::CPU),
            phi::AllocationType::CPU);
  EXPECT_EQ(c10::DeviceTypeToPhi(c10::DeviceType::CUDA),
            phi::AllocationType::GPU);
  EXPECT_EQ(c10::DeviceTypeToPhi(c10::DeviceType::XPU),
            phi::AllocationType::XPU);
  EXPECT_EQ(c10::DeviceTypeToPhi(c10::DeviceType::IPU),
            phi::AllocationType::IPU);
  EXPECT_EQ(c10::DeviceTypeToPhi(c10::DeviceType::CUSTOM),
            phi::AllocationType::CUSTOM);
  EXPECT_EQ(c10::DeviceTypeToPhi(static_cast<c10::DeviceType>(-1)),
            phi::AllocationType::UNDEFINED);

  EXPECT_EQ(c10::PhiToDeviceType(phi::AllocationType::CPU),
            c10::DeviceType::CPU);
  EXPECT_EQ(c10::PhiToDeviceType(phi::AllocationType::GPU),
            c10::DeviceType::CUDA);
  EXPECT_EQ(c10::PhiToDeviceType(phi::AllocationType::XPU),
            c10::DeviceType::XPU);
  EXPECT_EQ(c10::PhiToDeviceType(phi::AllocationType::IPU),
            c10::DeviceType::IPU);
  EXPECT_EQ(c10::PhiToDeviceType(phi::AllocationType::CUSTOM),
            c10::DeviceType::CUSTOM);
  EXPECT_EQ(c10::PhiToDeviceType(phi::AllocationType::UNDEFINED),
            c10::DeviceType::CPU);

  EXPECT_TRUE(c10::isValidDeviceType(c10::DeviceType::CPU));
  EXPECT_TRUE(c10::isValidDeviceType(c10::DeviceType::CUSTOM));
  EXPECT_FALSE(c10::isValidDeviceType(static_cast<c10::DeviceType>(-9)));

  std::ostringstream cpu_os;
  cpu_os << c10::DeviceType::CPU;
  EXPECT_EQ(cpu_os.str(), "cpu");
  std::ostringstream cuda_os;
  cuda_os << c10::DeviceType::CUDA;
  EXPECT_EQ(cuda_os.str(), "cuda");
  std::ostringstream xpu_os;
  xpu_os << c10::DeviceType::XPU;
  EXPECT_EQ(xpu_os.str(), "xpu");
  std::ostringstream ipu_os;
  ipu_os << c10::DeviceType::IPU;
  EXPECT_EQ(ipu_os.str(), "ipu");
  std::ostringstream custom_os;
  custom_os << c10::DeviceType::CUSTOM;
  EXPECT_EQ(custom_os.str(), "privateuseone");
  std::ostringstream invalid_os;
  invalid_os << static_cast<c10::DeviceType>(99);
  EXPECT_TRUE(invalid_os.str().empty());

  EXPECT_EQ(c10::DeviceType::PrivateUse1, c10::DeviceType::CUSTOM);
  EXPECT_EQ(c10::kPrivateUse1, c10::DeviceType::PrivateUse1);
}

TEST(DeviceCompatTest, DeviceParseAndPlaceBranches) {
  c10::Device cpu("cpu");
  EXPECT_TRUE(cpu.is_cpu());
  EXPECT_FALSE(cpu.has_index());
  EXPECT_EQ(cpu.str(), "cpu");

  c10::Device cuda("cuda:3");
  EXPECT_TRUE(cuda.is_cuda());
  EXPECT_TRUE(cuda.has_index());
  EXPECT_EQ(cuda.index(), 3);
  EXPECT_EQ(cuda.str(), "cuda:3");

  c10::Device xpu("xpu:1");
  EXPECT_EQ(xpu.type(), c10::DeviceType::XPU);
  EXPECT_EQ(xpu.index(), 1);
  EXPECT_EQ(xpu.str(), "xpu:1");

  c10::Device ipu("ipu:2");
  EXPECT_EQ(ipu.type(), c10::DeviceType::IPU);
  EXPECT_EQ(ipu.index(), 2);
  EXPECT_EQ(ipu.str(), "ipu:2");

  EXPECT_THROW(c10::Device(""), ::std::exception);
  EXPECT_THROW(c10::Device("npu:0"), ::std::exception);
  EXPECT_THROW(c10::Device("cuda:abc"), ::std::exception);
  EXPECT_THROW(c10::Device("cuda:9999999999999999999999"), ::std::exception);

  c10::Device custom(c10::DeviceType::CUSTOM, 5, "npu");
  phi::Place custom_place = custom._PD_GetInner();
  EXPECT_EQ(custom_place.GetType(), phi::AllocationType::CUSTOM);
  EXPECT_EQ(custom_place.GetDeviceId(), 5);
  EXPECT_EQ(custom.str(), "privateuseone:5");

  c10::Device cuda_no_index(c10::DeviceType::CUDA);
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  auto device_count = c10::cuda::device_count();
  if (device_count == 0) {
    return;
  }
  EXPECT_EQ(cuda_no_index._PD_GetInner().GetType(), phi::AllocationType::GPU);
  if (device_count >= 2) {
    c10::cuda::CUDAGuard guard(1);
    EXPECT_EQ(c10::Device(c10::DeviceType::CUDA)._PD_GetInner().GetDeviceId(),
              1);
  } else {
    EXPECT_EQ(cuda_no_index._PD_GetInner().GetDeviceId(), 0);
  }
#else
  EXPECT_EQ(cuda_no_index._PD_GetInner().GetType(), phi::AllocationType::GPU);
  EXPECT_EQ(cuda_no_index._PD_GetInner().GetDeviceId(), 0);
#endif
  c10::Device xpu_no_index(c10::DeviceType::XPU);
#ifdef PADDLE_WITH_XPU
  auto xpu_device_count = paddle::platform::GetXPUDeviceCount();
  if (xpu_device_count > 0) {
    EXPECT_EQ(xpu_no_index._PD_GetInner().GetType(), phi::AllocationType::XPU);
  }
  if (xpu_device_count >= 2) {
    paddle::platform::XPUDeviceGuard guard(1);
    EXPECT_EQ(c10::Device(c10::DeviceType::XPU)._PD_GetInner().GetDeviceId(),
              1);
  } else if (xpu_device_count == 1) {
    EXPECT_EQ(xpu_no_index._PD_GetInner().GetDeviceId(), 0);
  }
#else
  EXPECT_EQ(xpu_no_index._PD_GetInner().GetType(), phi::AllocationType::XPU);
  EXPECT_EQ(xpu_no_index._PD_GetInner().GetDeviceId(), 0);
#endif
  c10::Device ipu_no_index(c10::DeviceType::IPU);
  EXPECT_EQ(ipu_no_index._PD_GetInner().GetType(), phi::AllocationType::IPU);
  EXPECT_EQ(ipu_no_index._PD_GetInner().GetDeviceId(), 0);

  c10::Device invalid(static_cast<c10::DeviceType>(-1), 0);
  phi::Place fallback_place = invalid._PD_GetInner();
  EXPECT_EQ(fallback_place.GetType(), phi::AllocationType::CPU);
  EXPECT_EQ(invalid.str(), "cpu:0");

  std::ostringstream os;
  os << cuda;
  EXPECT_EQ(os.str(), "cuda:3");
}

TEST(DeviceCompatTest, DeviceInterfaceParity) {
  c10::Device cpu(c10::kCPU);
  c10::Device cuda(c10::kCUDA, 0);
  c10::Device xpu(c10::kXPU, 1);
  c10::Device ipu(c10::kIPU, 2);
  c10::Device privateuse(c10::kPrivateUse1, 4);

  EXPECT_TRUE(cpu.is_cpu());
  EXPECT_TRUE(cuda.is_cuda());
  EXPECT_TRUE(xpu.is_xpu());
  EXPECT_TRUE(ipu.is_ipu());
  EXPECT_TRUE(privateuse.is_privateuseone());
  EXPECT_FALSE(privateuse.is_mps());
  EXPECT_FALSE(privateuse.is_hip());
  EXPECT_FALSE(privateuse.is_meta());
  EXPECT_TRUE(cpu.supports_as_strided());
  EXPECT_FALSE(ipu.supports_as_strided());

  c10::Device cpu_with_index(c10::kCPU);
  cpu_with_index.set_index(0);
  EXPECT_EQ(cpu_with_index.index(), 0);
  EXPECT_EQ(cpu_with_index.str(), "cpu:0");

  c10::Device cuda_with_index(c10::kCUDA);
  cuda_with_index.set_index(2);
  EXPECT_EQ(cuda_with_index.index(), 2);
  EXPECT_EQ(cuda_with_index.str(), "cuda:2");

  EXPECT_NE(cpu, cuda);
  EXPECT_EQ(cuda, c10::Device(c10::kCUDA, 0));
  EXPECT_EQ(privateuse.str(), "privateuseone:4");
  EXPECT_TRUE(c10::Device("privateuseone:7").is_privateuseone());

  EXPECT_THROW(c10::Device("cuda:-1"), ::std::exception);
  EXPECT_THROW(c10::Device("cuda:01"), ::std::exception);
  EXPECT_THROW(c10::Device("cuda:1:2"), ::std::exception);

  std::unordered_map<c10::Device, int> device_map;
  device_map.emplace(c10::Device(c10::kCUDA, 0), 7);
  device_map.emplace(c10::Device(c10::kCPU), 3);
  EXPECT_EQ(device_map.at(c10::Device(c10::kCUDA, 0)), 7);
  EXPECT_EQ(device_map.at(c10::Device(c10::kCPU)), 3);
}
