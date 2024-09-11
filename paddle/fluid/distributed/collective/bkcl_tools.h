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

#pragma once

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/distributed/types.h"
#include "paddle/phi/core/platform/device_context.h"

namespace paddle {
namespace distributed {
using XPUContext = phi::XPUContext;
using phi::distributed::ReduceOp;

#define BKCLCHECK(cmd)                                                  \
  do {                                                                  \
    BKCLResult_t r = cmd;                                               \
    if (r != BKCL_SUCCESS) {                                            \
      printf("Failed, BKCL error %s:%d '%d'\n", __FILE__, __LINE__, r); \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  } while (0)

class XPUEventManager {
 public:
  XPUEventManager() {}

  ~XPUEventManager() {
    if (is_created_) {
      phi::backends::xpu::XPUDeviceGuard guard(device_index_);
      xpu_event_destroy(event_);
    }
  }

  XPUEventManager(const XPUEventManager&) = delete;
  XPUEventManager& operator=(const XPUEventManager&) = delete;

  XPUEventManager(XPUEventManager&& other) {
    std::swap(is_created_, other.is_created_);
    std::swap(device_index_, other.device_index_);
    std::swap(event_, other.event_);
  }

  XPUEventManager& operator=(XPUEventManager&& other) {
    std::swap(is_created_, other.is_created_);
    std::swap(device_index_, other.device_index_);
    std::swap(event_, other.event_);
    return *this;
  }

  bool IsCreated() const { return is_created_; }
  bool DeviceId() const { return device_index_; }
  xpuEventHandle GetRawXpuEvent() const { return event_; }

  void Record(const XPUContext& ctx) {
    auto device_index = ctx.GetPlace().device;
    if (!is_created_) {
      CreateEvent(device_index);
    }
    PADDLE_ENFORCE_EQ(device_index,
                      device_index_,
                      common::errors::PreconditionNotMet(
                          "XPUContext's device %d does not match"
                          "Event's device %d",
                          device_index,
                          device_index_));

    phi::backends::xpu::XPUDeviceGuard guard(device_index_);
    // TODO(zhangxiaoci) temporary solution: xpu::event seems buggy
    PADDLE_ENFORCE_XPU_SUCCESS(xpu_wait(ctx.stream()));
  }

  void Block(const XPUContext& ctx) const {}

 private:
  bool is_created_{false};
  xpuEventHandle event_{};
  int8_t device_index_{0};

 private:
  void CreateEvent(int device_index) {
    device_index_ = device_index;
    phi::backends::xpu::XPUDeviceGuard guard(device_index);

    PADDLE_ENFORCE_XPU_SUCCESS(xpu_event_create(&event_));

    is_created_ = true;
  }
};

BKCLOp ToBKCLRedType(ReduceOp reduction);
std::string SerializeBKCLUniqueId(const BKCLUniqueId& bkclId);
std::string BKCLDTypeToString(BKCLDataType dtype);
std::string BKCLRedTypeToString(BKCLOp op);

}  // namespace distributed
}  // namespace paddle
