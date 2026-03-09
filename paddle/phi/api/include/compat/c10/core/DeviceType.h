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

// DeviceType 枚举基于 Paddle 实际支持的 phi::AllocationType 设计，
// 并对外暴露 PyTorch 兼容的命名（如用 CUDA 代替 GPU）。
// 只包含 Paddle 后端有实际实现的设备类型，确保转换函数不会走 Undefined 分支。
//
// Paddle phi::AllocationType 对应关系：
//   CPU=1, GPU/CUDA=2, GPUPINNED=3, XPU=4, XPUPINNED=5, IPU=7, CUSTOM=9
//
// 注意：CPU=0、CUDA=1 与 PyTorch 保持一致，保证 ABI 层面的基本兼容。
enum class DeviceType : int8_t {
  CPU = 0,   // phi::AllocationType::CPU
  CUDA = 1,  // phi::AllocationType::GPU  (PyTorch 规范命名)
  GPUPINNED = 2,  // phi::AllocationType::GPUPINNED (GPU 固定内存，位于 CPU 侧)
  XPU = 3,        // phi::AllocationType::XPU  (百度昆仑 XPU)
  XPUPINNED = 4,  // phi::AllocationType::XPUPINNED (XPU 固定内存)
  IPU = 5,        // phi::AllocationType::IPU  (Graphcore IPU)
  CUSTOM = 6,     // phi::AllocationType::CUSTOM (自定义/扩展设备)

  Undefined = -1,
};

// ── 常量定义 ─────────────────────────────────────────────────────────
constexpr DeviceType kCPU = DeviceType::CPU;
constexpr DeviceType kCUDA = DeviceType::CUDA;
constexpr DeviceType kGPUPINNED = DeviceType::GPUPINNED;
constexpr DeviceType kXPU = DeviceType::XPU;
constexpr DeviceType kXPUPINNED = DeviceType::XPUPINNED;
constexpr DeviceType kIPU = DeviceType::IPU;
constexpr DeviceType kCUSTOM = DeviceType::CUSTOM;

// ── phi::AllocationType → c10::DeviceType ────────────────────────────
inline DeviceType phiAllocationTypeToC10DeviceType(phi::AllocationType type) {
  switch (type) {
    case phi::AllocationType::CPU:
      return DeviceType::CPU;
    case phi::AllocationType::GPU:  // GPU 与 CUDA 在 Paddle 内部等价
      return DeviceType::CUDA;
    case phi::AllocationType::GPUPINNED:
      return DeviceType::GPUPINNED;
    case phi::AllocationType::XPU:
      return DeviceType::XPU;
    case phi::AllocationType::XPUPINNED:
      return DeviceType::XPUPINNED;
    case phi::AllocationType::IPU:
      return DeviceType::IPU;
    case phi::AllocationType::CUSTOM:
      return DeviceType::CUSTOM;
    default:
      return DeviceType::Undefined;
  }
}

// ── c10::DeviceType → phi::AllocationType ────────────────────────────
inline phi::AllocationType c10DeviceTypeToPhiAllocationType(DeviceType type) {
  switch (type) {
    case DeviceType::CPU:
      return phi::AllocationType::CPU;
    case DeviceType::CUDA:
      return phi::AllocationType::GPU;
    case DeviceType::GPUPINNED:
      return phi::AllocationType::GPUPINNED;
    case DeviceType::XPU:
      return phi::AllocationType::XPU;
    case DeviceType::XPUPINNED:
      return phi::AllocationType::XPUPINNED;
    case DeviceType::IPU:
      return phi::AllocationType::IPU;
    case DeviceType::CUSTOM:
      return phi::AllocationType::CUSTOM;
    default:
      return phi::AllocationType::UNDEFINED;
  }
}

// ── phi::Place → c10::Device 辅助：判断是否携带显式 device index ─────
// CPU、GPUPINNED、XPUPINNED 均为无索引设备（固定内存无设备编号语义）。
inline bool phiPlaceHasC10DeviceIndex(phi::AllocationType type,
                                      int index) noexcept {
  switch (type) {
    case phi::AllocationType::CPU:
    case phi::AllocationType::GPUPINNED:
    case phi::AllocationType::XPUPINNED:
      return false;
    default:
      return index != -1;
  }
}

}  // namespace c10

namespace at {
using c10::DeviceType;
using c10::kCPU;
using c10::kCUDA;
using c10::kCUSTOM;
using c10::kGPUPINNED;
using c10::kIPU;
using c10::kXPU;
using c10::kXPUPINNED;
}  // namespace at

namespace torch {
using c10::DeviceType;
using c10::kCPU;
using c10::kCUDA;
using c10::kCUSTOM;
using c10::kGPUPINNED;
using c10::kIPU;
using c10::kXPU;
using c10::kXPUPINNED;
}  // namespace torch
