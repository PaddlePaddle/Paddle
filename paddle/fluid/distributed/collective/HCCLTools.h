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

#include <error.h>
#include <string>

#include "boost/variant.hpp"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/npu/enforce_npu.h"
#include "paddle/fluid/platform/device/npu/npu_info.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace distributed {

class NPUEventManager {
 public:
  NPUEventManager() = default;

  ~NPUEventManager() {
    if (is_created_) {
      platform::NPUDeviceGuard guard(device_index_);
      platform::NPUEventDestroy(event_);
    }
  }

  NPUEventManager(const NPUEventManager&) = delete;
  NPUEventManager& operator=(const NPUEventManager&) = delete;

  NPUEventManager(NPUEventManager&& other) {
    std::swap(is_created_, other.is_created_);
    std::swap(device_index_, other.device_index_);
    std::swap(event_, other.event_);
  }

  NPUEventManager& operator=(NPUEventManager&& other) {
    std::swap(is_created_, other.is_created_);
    std::swap(device_index_, other.device_index_);
    std::swap(event_, other.event_);
    return *this;
  }

  bool IsCreated() const { return is_created_; }
  bool DeviceId() const { return device_index_; }
  aclrtEvent GetRawNPUEvent() const { return event_; }

  void Record(const paddle::platform::NPUDeviceContext& ctx) {
    auto device_index = ctx.GetPlace().device;
    if (!is_created_) {
      CreateEvent(device_index);
    }
    PADDLE_ENFORCE_EQ(device_index, device_index_,
                      platform::errors::PreconditionNotMet(
                          "NPUDeviceContext's device %d does not match"
                          "Event's device %d",
                          device_index, device_index_));

    platform::NPUDeviceGuard guard(device_index_);
    platform::NPUEventRecord(event_, ctx.stream());
  }

  bool Query() const {
    aclrtEventStatus status = ACL_EVENT_STATUS_COMPLETE;
    platform::NPUEventQuery(event_, &status);
    if (status == ACL_EVENT_STATUS_COMPLETE) {
      return true;
    }
    return false;
  }

  void Block(const paddle::platform::NPUDeviceContext& ctx) const {
    if (is_created_) {
      auto device_index = ctx.GetPlace().device;
      PADDLE_ENFORCE_EQ(device_index, device_index_,
                        platform::errors::PreconditionNotMet(
                            "CUDADeviceContext's device %d does not match"
                            "Event's device %d",
                            device_index, device_index_));
      platform::NPUDeviceGuard guard(device_index_);
      platform::NPUStreamWaitEvent(ctx.stream(), event_);
    }
  }

 private:
  bool is_created_{false};
  aclrtEvent event_{};
  int8_t device_index_{0};

 private:
  void CreateEvent(int device_index) {
    device_index_ = device_index;
    platform::NPUDeviceGuard guard(device_index);
    platform::NPUEventCreate(&event_);
    is_created_ = true;
  }
};

class HCCLCommManager {
 public:
  explicit HCCLCommManager(HcclComm hcclComm) : hccl_comm_(hcclComm) {}

  HCCLCommManager() : HCCLCommManager(nullptr) {}

  ~HCCLCommManager() noexcept {
    std::unique_lock<std::mutex> lock(mutex_);
    if (hccl_comm_) {
      platform::dynload::HcclCommDestroy(hccl_comm_);
    }
  }

  static std::shared_ptr<HCCLCommManager> Create(int num_ranks, int rank,
                                                 HcclRootInfo* comm_id,
                                                 HcclComm hccl_comm) {
    auto hccl_manager = std::make_shared<HCCLCommManager>();
    auto ret = platform::dynload::HcclCommInitRootInfo(num_ranks, comm_id, rank,
                                                       &hccl_comm);
    using __NPU_STATUS_TYPE__ = decltype(ret);
    constexpr auto __success_type__ =
        platform::details::NPUStatusType<__NPU_STATUS_TYPE__>::kSuccess;
    if (UNLIKELY(ret != __success_type__)) {
      VLOG(0) << "Error: create hccl_id error.";
      exit(-1);
    }

    hccl_manager->hccl_id_ = comm_id;
    hccl_manager->rank_ = rank;
    hccl_manager->hccl_comm_ = hccl_comm;
    return hccl_manager;
  }

  HcclRootInfo* GetHcclId() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return hccl_id_;
  }

  HcclComm GetHcclComm() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return hccl_comm_;
  }

  HCCLCommManager(const HCCLCommManager&) = delete;
  HCCLCommManager& operator=(const HCCLCommManager&) = delete;
  HCCLCommManager& operator=(HCCLCommManager&& other) = delete;

  HCCLCommManager(HCCLCommManager&& other) {
    std::unique_lock<std::mutex> lock(other.mutex_);
    std::swap(hccl_comm_, other.hccl_comm_);
  }

 protected:
  HcclComm hccl_comm_;
  HcclRootInfo* hccl_id_;
  int rank_;
  mutable std::mutex mutex_;
};

}  // namespace distributed
}  // namespace paddle
