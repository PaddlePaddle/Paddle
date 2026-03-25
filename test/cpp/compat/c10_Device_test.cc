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

#include <sstream>

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
  std::ostringstream custom_os;
  custom_os << c10::DeviceType::CUSTOM;
  EXPECT_EQ(custom_os.str(), "privateuseone");
  std::ostringstream invalid_os;
  invalid_os << static_cast<c10::DeviceType>(99);
  EXPECT_TRUE(invalid_os.str().empty());
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

  c10::Device ipu("ipu:2");
  EXPECT_EQ(ipu.type(), c10::DeviceType::IPU);
  EXPECT_EQ(ipu.index(), 2);

  EXPECT_THROW(c10::Device(""), ::std::exception);
  EXPECT_THROW(c10::Device("npu:0"), ::std::exception);
  EXPECT_THROW(c10::Device("cuda:abc"), ::std::exception);
  EXPECT_THROW(c10::Device("cuda:9999999999999999999999"), ::std::exception);

  c10::Device custom(c10::DeviceType::CUSTOM, 5, "npu");
  phi::Place custom_place = custom._PD_GetInner();
  EXPECT_EQ(custom_place.GetType(), phi::AllocationType::CUSTOM);
  EXPECT_EQ(custom_place.GetDeviceId(), 5);

  c10::Device invalid(static_cast<c10::DeviceType>(-1), 0);
  phi::Place fallback_place = invalid._PD_GetInner();
  EXPECT_EQ(fallback_place.GetType(), phi::AllocationType::CPU);

  std::ostringstream os;
  os << cuda;
  EXPECT_EQ(os.str(), "cuda:3");
}
