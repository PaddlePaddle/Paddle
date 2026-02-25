// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <ostream>

#include "paddle/phi/common/place.h"

namespace c10 {

// 使用与 PyTorch 兼容的 DeviceType 枚举值
// PyTorch: CPU=0, CUDA=1
enum class DeviceType : int {
  CPU = 0,
  CUDA = 1,
  IPU = 2,
  XPU = 3,
  FPGA = 4,
  IDEEP = 5,
  MKLDNN = 6,
  CPU_Pinned = 7,
  Custom = 8,
  GPU = 9,  // Paddle uses GPU instead of CUDA

  // 占位符
  Undefined = -1
};

// 常量定义 - 使用新枚举
constexpr DeviceType kCPU = DeviceType::CPU;
constexpr DeviceType kCUDA = DeviceType::CUDA;
constexpr DeviceType kCUSTOM = DeviceType::Custom;
constexpr DeviceType kGPU = DeviceType::GPU;

// 辅助函数：从 phi::AllocationType 转换到 c10::DeviceType
inline DeviceType phiAllocationTypeToC10DeviceType(phi::AllocationType type) {
  switch (type) {
    case phi::AllocationType::CPU:
      return DeviceType::CPU;
    case phi::AllocationType::GPU:
      return DeviceType::CUDA;  // Map to CUDA for PyTorch compatibility
    case phi::AllocationType::XPU:
      return DeviceType::XPU;
    case phi::AllocationType::IPU:
      return DeviceType::IPU;
    case phi::AllocationType::CUSTOM:
      return DeviceType::Custom;
    default:
      return DeviceType::Undefined;
  }
}

// 辅助函数：从 c10::DeviceType 转换到 phi::AllocationType
inline phi::AllocationType c10DeviceTypeToPhiAllocationType(DeviceType type) {
  switch (type) {
    case DeviceType::CPU:
      return phi::AllocationType::CPU;
    case DeviceType::CUDA:
    case DeviceType::GPU:
      return phi::AllocationType::GPU;
    case DeviceType::XPU:
      return phi::AllocationType::XPU;
    case DeviceType::IPU:
      return phi::AllocationType::IPU;
    case DeviceType::Custom:
      return phi::AllocationType::CUSTOM;
    default:
      return phi::AllocationType::UNDEFINED;
  }
}

}  // namespace c10

namespace at {
using c10::DeviceType;
using c10::kCPU;
using c10::kCUDA;
using c10::kCUSTOM;
using c10::kGPU;
}  // namespace at

namespace torch {
using c10::DeviceType;
using c10::kCPU;
using c10::kCUDA;
using c10::kCUSTOM;
using c10::kGPU;
}  // namespace torch
