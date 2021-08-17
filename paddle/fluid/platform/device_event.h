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
#pragma once
#include <memory>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace platform {

class DeviceOption;
class DeviceEvent;

constexpr int MaxDeviceTypes =
    static_cast<int>(platform::DeviceType::MAX_DEVICE_TYPES);

typedef void (*EventCreateFunction)(DeviceEvent*, const DeviceOption&);
typedef void (*EventRecordFunction)(DeviceEvent*, const platform::Place&,
                                    const void*);
typedef bool (*EventQueryFunction)(const DeviceEvent*);

class DeviceOption {
 public:
  explicit DeviceOption(int device_type) : device_type_(device_type) {}

  DeviceOption(int device_type, int device_id)
      : device_type_(device_type), device_id_(device_id) {}

  int device_type() const { return device_type_; }

  int device_id() const { return device_id_; }

 private:
  int device_type_;
  int device_id_;
};

class DeviceEvent {
 public:
  explicit DeviceEvent(const DeviceOption& device_option)
      : event_(),
        type_(device_option.device_type()),
        device_option_(device_option) {
    PADDLE_ENFORCE_LT(type_, MaxDeviceTypes,
                      platform::errors::PreconditionNotMet(
                          "Required type < %d, but received type = %d",
                          MaxDeviceTypes, type_));
    PADDLE_ENFORCE_NOT_NULL(
        event_creator_[type_],
        platform::errors::Unavailable(
            "event_creator_[%d] shall not be nullptr.", type_));
    event_creator_[type_](this, device_option_);
  }

  ~DeviceEvent() {}

  void Record(const platform::Place& place, const void* dev_ctx) {
    PADDLE_ENFORCE_NOT_NULL(
        event_recorder_[type_],
        platform::errors::Unavailable(
            "event_recorder_[%d] shall not be nullptr.", type_));
    event_recorder_[type_](this, place, dev_ctx);
  }

  bool Query() {
    PADDLE_ENFORCE_NOT_NULL(
        event_querier_[type_],
        platform::errors::Unavailable(
            "event_querier_[%d] shall not be nullptr.", type_));
    return event_querier_[type_](this);
  }

  void InitEvent(std::shared_ptr<void> event) { event_ = event; }

  std::shared_ptr<void> GetEvent() const { return event_; }

 private:
  std::shared_ptr<void> event_;
  int type_;
  DeviceOption device_option_;

  static EventCreateFunction event_creator_[MaxDeviceTypes];
  static EventRecordFunction event_recorder_[MaxDeviceTypes];
  static EventQueryFunction event_querier_[MaxDeviceTypes];

  template <DeviceType device_typ>
  friend struct EventCreateFunctionRegisterer;

  template <DeviceType device_typ>
  friend struct EventRecordFunctionRegisterer;

  template <DeviceType device_typ>
  friend struct EventQueryFunctionRegisterer;
};

inline int DeviceTypeToId(const DeviceType& device_type) {
  return static_cast<int>(device_type);
}

template <DeviceType device_type>
struct EventCreateFunctionRegisterer {
  explicit EventCreateFunctionRegisterer(EventCreateFunction func) {
    auto type_idx = DeviceTypeToId(device_type);
    DeviceEvent::event_creator_[type_idx] = func;
    VLOG(2) << "register creator " << type_idx << " with "
            << DeviceEvent::event_creator_[type_idx];
  }
  void Touch() {}
};

#define REGISTER_EVENT_CREATE_FUNCTION(device_type, func)               \
  static ::paddle::platform::EventCreateFunctionRegisterer<device_type> \
      g_device_event_create_1(func);                                    \
  int touch_g_device_event_create_1() {                                 \
    g_device_event_create_1.Touch();                                    \
    return 0;                                                           \
  }

#define USE_EVENT(device_type)                \
  extern int touch_g_device_event_create_1(); \
  UNUSED static int use_event_itself_1 = touch_g_device_event_create_1();

template <DeviceType device_type>
struct EventRecordFunctionRegisterer {
  explicit EventRecordFunctionRegisterer(EventRecordFunction func) {
    auto type_idx = DeviceTypeToId(device_type);
    DeviceEvent::event_recorder_[type_idx] = func;
  }
};
#define REGISTER_EVENT_RECORD_FUNCTION(device_type, func) \
  namespace {                                             \
  static EventRecordFunctionRegisterer<device_type>       \
      g_device_event_record_##type_idx(func);             \
  }

template <DeviceType device_type>
struct EventQueryFunctionRegisterer {
  explicit EventQueryFunctionRegisterer(EventQueryFunction func) {
    auto type_idx = DeviceTypeToId(device_type);
    DeviceEvent::event_querier_[type_idx] = func;
  }
};
#define REGISTER_EVENT_QUERY_FUNCTION(device_type, func) \
  namespace {                                            \
  static EventQueryFunctionRegisterer<device_type>       \
      g_device_event_query_##type_idx(func);             \
  }

}  // namespace platform
}  // namespace paddle
