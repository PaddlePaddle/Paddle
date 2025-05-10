// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#ifndef PADDLE_PHI_CORE_PLATFORM_DEVICE_EVENT_DEFS_H
#define PADDLE_PHI_CORE_PLATFORM_DEVICE_EVENT_DEFS_H

#pragma once
#include <memory>

#include "paddle/phi/core/platform/device_type.h"

namespace paddle {
namespace platform {

class DeviceOption;
class DeviceEvent;

constexpr int MaxDeviceTypes =
    static_cast<int>(platform::DeviceType::MAX_DEVICE_TYPES);

typedef void (*EventCreateFunction)(DeviceEvent*,
                                    const phi::Place&,
                                    unsigned int flag);
typedef void (*EventRecordFunction)(DeviceEvent*, const DeviceContext*);
typedef bool (*EventQueryFunction)(const DeviceEvent*);
typedef void (*EventFinishFunction)(const DeviceEvent*);
typedef void (*EventSetFinishedFunction)(const DeviceEvent*);
typedef void (*EventWaitFunction)(const DeviceEvent*, const DeviceContext*);
typedef void (*EventResetFunction)(const DeviceEvent*);

inline int DeviceTypeToId(const DeviceType& device_type) {
  return static_cast<int>(device_type);
}

unsigned int GenerateDeviceEventFlag(bool enable_timing = false,
                                     bool blocking = false,
                                     bool interprocess = false);

enum EventStatus {
  INITIALIZED = 0,
  SCHEDULED = 1,
  SUCCESS = 2,
  FAILED = 3,
};

class DeviceEvent {
 public:
  explicit DeviceEvent(const phi::Place& place, unsigned int flag);
  ~DeviceEvent() {}

  void Record(const DeviceContext* dev_ctx);
  bool Query();
  void Finish() const;
  void SetFinished();
  void Reset();
  void Wait(const DeviceType& waiter_type, const DeviceContext* context) const;

  void InitEvent(std::shared_ptr<void> event) { event_ = event; }
  std::shared_ptr<void> GetEvent() const { return event_; }

 private:
  std::shared_ptr<void> event_;
  phi::Place place_;
  int type_id_;
  unsigned int flag_;
  bool recorded_{false};

  // Static function pointers for event operations
  static EventCreateFunction event_creator_[MaxDeviceTypes];
  static EventRecordFunction event_recorder_[MaxDeviceTypes];
  static EventQueryFunction event_querier_[MaxDeviceTypes];
  static EventFinishFunction event_finisher_[MaxDeviceTypes];
  static EventSetFinishedFunction event_finished_setter_[MaxDeviceTypes];
  static EventWaitFunction event_waiter_[MaxDeviceTypes][MaxDeviceTypes];
  static EventResetFunction event_resetter_[MaxDeviceTypes];

  // Template friends for function registerers
  template <DeviceType device_typ>
  friend struct EventCreateFunctionRegisterer;
  template <DeviceType device_typ>
  friend struct EventRecordFunctionRegisterer;
  template <DeviceType device_typ>
  friend struct EventQueryFunctionRegisterer;
  template <DeviceType device_typ>
  friend struct EventFinishFunctionRegisterer;
  template <DeviceType device_typ>
  friend struct EventSetFinishedFunctionRegisterer;
  template <DeviceType waiter_typ, DeviceType event_type>
  friend struct EventWaitFunctionRegisterer;
  template <DeviceType device_typ>
  friend struct EventResetFunctionRegisterer;
};

}  // namespace platform
}  // namespace paddle
#endif  // PADDLE_PHI_CORE_PLATFORM_DEVICE_EVENT_DEFS_H
