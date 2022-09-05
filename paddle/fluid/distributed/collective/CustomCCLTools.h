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

#include "paddle/fluid/distributed/collective/Types.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/backends/device_guard.h"
#include "paddle/phi/backends/device_manager.h"

namespace paddle {
namespace distributed {

class CustomEventManager {
 public:
  CustomEventManager() = default;

  ~CustomEventManager() {
    if (is_created_) {
      event_->Destroy();
    }
  }

  CustomEventManager(const CustomEventManager&) = delete;
  CustomEventManager& operator=(const CustomEventManager&) = delete;

  CustomEventManager(CustomEventManager&& other) {
    std::swap(is_created_, other.is_created_);
    std::swap(device_index_, other.device_index_);
    std::swap(device_type_, other.device_type_);
    std::swap(event_, other.event_);
  }

  CustomEventManager& operator=(CustomEventManager&& other) {
    std::swap(is_created_, other.is_created_);
    std::swap(device_index_, other.device_index_);
    std::swap(device_type_, other.device_type_);
    std::swap(event_, other.event_);
    return *this;
  }

  bool IsCreated() const { return is_created_; }
  int8_t DeviceId() const { return device_index_; }
  std::string DeviceType() const { return device_type_; }
  phi::event::event_t GetRawCustomEvent() const { return event_->raw_event(); }
  phi::event::Event* GetCustomEvent() const { return event_.get(); }

  void Record(const paddle::platform::CustomDeviceContext& ctx) {
    auto place = ctx.GetPlace();
    auto device_type = place.GetDeviceType();
    auto device_index = place.GetDeviceId();
    if (!is_created_) {
      CreateEvent(place);
    }
    PADDLE_ENFORCE_EQ(device_index,
                      device_index_,
                      platform::errors::PreconditionNotMet(
                          "CustomDeviceContext's device %d does not match"
                          "Event's device %d",
                          device_index,
                          device_index_));
    PADDLE_ENFORCE_EQ(device_type,
                      device_type_,
                      platform::errors::PreconditionNotMet(
                          "CustomDeviceContext's device %d does not match"
                          "Event's device type %d",
                          device_type,
                          device_type_));

    phi::DeviceGuard guard(place);
    phi::stream::Stream stream(place, ctx.stream());
    event_->Record(&stream);
  }

  bool Query() const { return event_->Query(); }

  void Block(const paddle::platform::CustomDeviceContext& ctx) const {
    if (is_created_) {
      auto place = ctx.GetPlace();
      auto device_type = place.GetDeviceType();
      auto device_index = place.GetDeviceId();
      PADDLE_ENFORCE_EQ(device_index,
                        device_index_,
                        platform::errors::PreconditionNotMet(
                            "CustomDeviceContext's device %d does not match"
                            "Event's device %d",
                            device_index,
                            device_index_));
      PADDLE_ENFORCE_EQ(device_type,
                        device_type_,
                        platform::errors::PreconditionNotMet(
                            "CustomDeviceContext's device %d does not match"
                            "Event's device type %d",
                            device_type,
                            device_type_));
      phi::DeviceGuard guard(place);
      phi::stream::Stream stream(place, ctx.stream());
      stream.WaitEvent(event_.get());
    }
  }

 private:
  bool is_created_{false};
  std::shared_ptr<phi::event::Event> event_{nullptr};
  int8_t device_index_{0};
  std::string device_type_;

 private:
  void CreateEvent(const platform::Place& place) {
    device_index_ = place.GetDeviceId();
    device_type_ = place.GetDeviceType();
    event_.reset(new phi::event::Event);
    event_->Init(place);
    is_created_ = true;
  }
};

class CustomCCLCommManager {
 public:
  CustomCCLCommManager(const std::string& device_type,
                       phi::ccl::CCLComm ccl_comm)
      : device_type_(device_type), ccl_comm_(ccl_comm) {}

  CustomCCLCommManager() : CustomCCLCommManager("", nullptr) {}

  ~CustomCCLCommManager() noexcept {
    std::unique_lock<std::mutex> lock(mutex_);
    if (ccl_comm_) {
      phi::DeviceManager::CCLDestroyComm(device_type_, ccl_comm_);
    }
  }

  static std::shared_ptr<CustomCCLCommManager> Create(
      const std::string& device_type,
      int num_ranks,
      int rank,
      phi::ccl::CCLRootId* comm_id,
      phi::ccl::CCLComm* ccl_comm) {
    auto custom_ccl_manager = std::make_shared<CustomCCLCommManager>();
    phi::DeviceManager::CCLCommInitRank(
        device_type, num_ranks, comm_id, rank, ccl_comm);
    custom_ccl_manager->device_type_ = device_type;
    custom_ccl_manager->ccl_id_ = comm_id;
    custom_ccl_manager->rank_ = rank;
    custom_ccl_manager->ccl_comm_ = *ccl_comm;
    return custom_ccl_manager;
  }

  phi::ccl::CCLRootId* GetCustomCCLId() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return ccl_id_;
  }

  phi::ccl::CCLComm GetCustomCCLComm() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return ccl_comm_;
  }

  CustomCCLCommManager(const CustomCCLCommManager&) = delete;
  CustomCCLCommManager& operator=(const CustomCCLCommManager&) = delete;
  CustomCCLCommManager& operator=(CustomCCLCommManager&& other) = delete;

  CustomCCLCommManager(CustomCCLCommManager&& other) {
    std::unique_lock<std::mutex> lock(other.mutex_);
    std::swap(ccl_comm_, other.ccl_comm_);
  }

 protected:
  std::string device_type_;
  phi::ccl::CCLComm ccl_comm_;
  phi::ccl::CCLRootId* ccl_id_;
  int rank_;
  mutable std::mutex mutex_;
};

phi::ccl::CCLReduceOp ToCustomCCLRedType(ReduceOp reduction);
std::string SerializeCustomCCLUniqueId(const phi::ccl::CCLRootId& ccl_id);

}  // namespace distributed
}  // namespace paddle
