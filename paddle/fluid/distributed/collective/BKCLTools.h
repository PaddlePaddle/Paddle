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

#include "paddle/fluid/distributed/collective/Types.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/backends/xpu/xpu_context.h"

namespace paddle {
namespace distributed {
using XPUContext = phi::XPUContext;

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
      platform::XPUDeviceGuard guard(device_index_);
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
  xpuEventHandle GetRawXPUEvent() const { return event_; }

  void Record(const XPUContext& ctx) {
    auto device_index = ctx.GetPlace().device;
    if (!is_created_) {
      CreateEvent(device_index);
    }
    PADDLE_ENFORCE_EQ(device_index,
                      device_index_,
                      platform::errors::PreconditionNotMet(
                          "XPUContext's device %d does not match"
                          "Event's device %d",
                          device_index,
                          device_index_));

    platform::XPUDeviceGuard guard(device_index_);
    PADDLE_ENFORCE_XPU_SUCCESS(xpu_event_record(event_, ctx.stream()));
  }

  void Block(const XPUContext& ctx) const {
    if (is_created_) {
      auto device_index = ctx.GetPlace().device;
      PADDLE_ENFORCE_EQ(device_index,
                        device_index_,
                        platform::errors::PreconditionNotMet(
                            "XPUContext's device %d does not match"
                            "Event's device %d",
                            device_index,
                            device_index_));
      platform::XPUDeviceGuard guard(device_index_);
      PADDLE_ENFORCE_XPU_SUCCESS(xpu_stream_wait_event(ctx.stream(), event_));
    }
  }

 private:
  bool is_created_{false};
  xpuEventHandle event_{};
  int8_t device_index_{0};

 private:
  void CreateEvent(int device_index) {
    device_index_ = device_index;
    platform::XPUDeviceGuard guard(device_index);

    PADDLE_ENFORCE_XPU_SUCCESS(xpu_event_create(&event_));

    is_created_ = true;
  }
};

class BKCLCommManager {
 public:
  explicit BKCLCommManager(BKCLContext_t bkclComm) : bkcl_comm_(bkclComm) {}

  BKCLCommManager() : BKCLCommManager(nullptr) {}

  ~BKCLCommManager() noexcept {
    std::unique_lock<std::mutex> lock(mutex_);
    if (bkcl_comm_) {
      bkcl_destroy_context(bkcl_comm_);
    }
  }

  static std::shared_ptr<BKCLCommManager> Create(int num_ranks,
                                                 int rank,
                                                 BKCLUniqueId comm_id) {
    auto bkcl_manager = std::make_shared<BKCLCommManager>();
    BKCLCHECK(
        bkcl_init_rank(&(bkcl_manager->bkcl_comm_), rank, num_ranks, &comm_id));

    bkcl_manager->bkcl_id_ = comm_id;
    bkcl_manager->rank_ = rank;
    return bkcl_manager;
  }

  BKCLUniqueId GetBkclId() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return bkcl_id_;
  }

  BKCLContext_t GetBkclComm() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return bkcl_comm_;
  }

  BKCLCommManager(const BKCLCommManager&) = delete;
  BKCLCommManager& operator=(const BKCLCommManager&) = delete;
  BKCLCommManager& operator=(BKCLCommManager&& other) = delete;

  BKCLCommManager(BKCLCommManager&& other) {
    std::unique_lock<std::mutex> lock(other.mutex_);
    std::swap(bkcl_comm_, other.bkcl_comm_);
  }

 protected:
  BKCLContext_t bkcl_comm_;
  BKCLUniqueId bkcl_id_;
  int rank_;
  mutable std::mutex mutex_;
};

BKCLOp ToBKCLRedType(ReduceOp reduction);
std::string SerializeBKCLUniqueId(const BKCLUniqueId& bkclId);

}  // namespace distributed
}  // namespace paddle
