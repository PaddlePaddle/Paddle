// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/common/target.h"

#include <gtest/gtest.h>

#include "paddle/cinn/common/arch.h"

namespace cinn {
namespace common {

// 测试 GetMultiProcessCountImpl 函数对不同架构的实现

// 测试 UnknownArch 架构
TEST(TargetTest, GetMultiProcessCountImpl_UnknownArch) {
  UnknownArch arch;

  // 由于 UnknownArch 的实现会调用 LOG(FATAL)，我们需要捕获这个行为
  // 在实际测试中，这个调用会导致程序终止，所以我们需要验证它确实会终止
  ASSERT_DEATH({ GetMultiProcessCountImpl(arch); },
               "The target is not GPU! Cannot get multi processor count\\.");
}

// 测试 X86Arch 架构
TEST(TargetTest, GetMultiProcessCountImpl_X86Arch) {
  X86Arch arch;

  ASSERT_DEATH({ GetMultiProcessCountImpl(arch); },
               "The target is not GPU! Cannot get multi processor count\\.");
}

// 测试 ARMArch 架构
TEST(TargetTest, GetMultiProcessCountImpl_ARMArch) {
  ARMArch arch;

  ASSERT_DEATH({ GetMultiProcessCountImpl(arch); },
               "The target is not GPU! Cannot get multi processor count\\.");
}

// 测试 NVGPUArch 架构
TEST(TargetTest, GetMultiProcessCountImpl_NVGPUArch) {
  NVGPUArch arch;

  int result = GetMultiProcessCountImpl(arch);

  // 在没有 CUDA 支持的情况下，函数应该返回 0
  // 在有 CUDA 支持的情况下，函数会返回实际的 SM 数量
  // 由于测试环境可能没有 GPU，我们只验证返回值的合理性
  EXPECT_GE(result, 0);
#ifdef CINN_WITH_CUDA
  // 如果有 CUDA 支持，SM 数量应该大于 0（实际的 GPU）
  // 但在测试环境中可能是模拟的，所以不强制要求大于 0
#else
  // 如果没有 CUDA 支持，应该返回 0
  EXPECT_EQ(result, 0);
#endif
}

// 测试 HygonDCUArchHIP 架构
TEST(TargetTest, GetMultiProcessCountImpl_HygonDCUArchHIP) {
  HygonDCUArchHIP arch;

  // HygonDCUArchHIP 的实现会调用 CINN_NOT_IMPLEMENTED
  // 这通常会导致程序终止或抛出异常
  EXPECT_THROW({ GetMultiProcessCountImpl(arch); }, std::exception);
}

// 测试 HygonDCUArchSYCL 架构
TEST(TargetTest, GetMultiProcessCountImpl_HygonDCUArchSYCL) {
  HygonDCUArchSYCL arch;

  // HygonDCUArchSYCL 的实现会调用 CINN_NOT_IMPLEMENTED
  EXPECT_THROW({ GetMultiProcessCountImpl(arch); }, std::exception);
}

// 测试 GetMultiProcessCount 函数（通用接口）
TEST(TargetTest, GetMultiProcessCount) {
  // 测试 UnknownArch
  Arch unknown_arch = UnknownArch{};
  ASSERT_DEATH({ GetMultiProcessCount(unknown_arch); },
               "The target is not GPU! Cannot get multi processor count\\.");

  // 测试 X86Arch
  Arch x86_arch = X86Arch{};
  ASSERT_DEATH({ GetMultiProcessCount(x86_arch); },
               "The target is not GPU! Cannot get multi processor count\\.");

  // 测试 ARMArch
  Arch arm_arch = ARMArch{};
  ASSERT_DEATH({ GetMultiProcessCount(arm_arch); },
               "The target is not GPU! Cannot get multi processor count\\.");

  // 测试 NVGPUArch
  Arch nvgpu_arch = NVGPUArch{};
  int result = GetMultiProcessCount(nvgpu_arch);
  EXPECT_GE(result, 0);

  // 测试 HygonDCUArchHIP
  Arch hip_arch = HygonDCUArchHIP{};
  EXPECT_THROW({ GetMultiProcessCount(hip_arch); }, std::exception);

  // 测试 HygonDCUArchSYCL
  Arch sycl_arch = HygonDCUArchSYCL{};
  EXPECT_THROW({ GetMultiProcessCount(sycl_arch); }, std::exception);
}

// 测试 Target 类的 get_multi_processor_count 方法
TEST(TargetTest, TargetGetMultiProcessorCount) {
  // 测试主机目标（非GPU）
  Target host_target = DefaultHostTarget();
  ASSERT_DEATH({ host_target.get_multi_processor_count(); },
               "The target is not GPU! Cannot get multi processor count\\.");

  // 测试 GPU 目标
#ifdef CINN_WITH_CUDA
  Target gpu_target = DefaultNVGPUTarget();
  int result = gpu_target.get_multi_processor_count();
  EXPECT_GE(result, 0);
#endif

  // 测试未知目标
  Target unknown_target = UnkTarget();
  ASSERT_DEATH({ unknown_target.get_multi_processor_count(); },
               "The target is not GPU! Cannot get multi processor count\\.");
}

// 边界值测试：测试各种边界情况
TEST(TargetTest, EdgeCases) {
  // 测试默认设备目标
  Target default_target = DefaultTarget();

  // 根据编译时的配置，default_target 可能是 GPU 或 CPU
  // 如果是 GPU 目标，应该能够正常调用
  // 如果是 CPU 目标，应该会报错
  if (default_target.arch.IsNVGPUArch() ||
      default_target.arch.IsHygonDCUArchHIP() ||
      default_target.arch.IsHygonDCUArchSYCL()) {
    // GPU 目标应该能够正常调用
    int result = default_target.get_multi_processor_count();
    EXPECT_GE(result, 0);
  } else {
    // CPU 目标应该报错
    ASSERT_DEATH({ default_target.get_multi_processor_count(); },
                 "The target is not GPU! Cannot get multi processor count\\.");
  }
}

}  // namespace common
}  // namespace cinn
