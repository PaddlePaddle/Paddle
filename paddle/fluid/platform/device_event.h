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
                                    const DeviceContext*);
typedef bool (*EventQueryFunction)(const DeviceEvent*);
typedef void (*EventFinishFunction)(const DeviceEvent*);
typedef void (*EventWaitFunction)(const DeviceEvent*, DeviceContext*);

inline int DeviceTypeToId(const DeviceType& device_type) {
  return static_cast<int>(device_type);
}

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

  void Record(const platform::Place& place, const DeviceContext* dev_ctx) {
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

  void Finish() const {
    PADDLE_ENFORCE_NOT_NULL(
        event_finisher_[type_],
        platform::errors::Unavailable(
            "event_finisher_[%d] shall not be nullptr.", type_));
    event_finisher_[type_](this);
  }

  void Wait(const DeviceType& waiter_type, DeviceContext* context) const {
    auto waiter_idx = DeviceTypeToId(waiter_type);
    PADDLE_ENFORCE_NOT_NULL(
        event_waiter_[waiter_idx][type_],
        platform::errors::Unavailable(
            "event_waiter_[%d][%d] shall not be nullptr.", waiter_idx, type_));
    event_waiter_[waiter_idx][type_](this, context);
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
  static EventFinishFunction event_finisher_[MaxDeviceTypes];
  static EventWaitFunction event_waiter_[MaxDeviceTypes][MaxDeviceTypes];

  template <DeviceType device_typ>
  friend struct EventCreateFunctionRegisterer;

  template <DeviceType device_typ>
  friend struct EventRecordFunctionRegisterer;

  template <DeviceType device_typ>
  friend struct EventQueryFunctionRegisterer;

  template <DeviceType device_typ>
  friend struct EventFinishFunctionRegisterer;

  template <DeviceType waiter_typ, DeviceType event_type>
  friend struct EventWaitFunctionRegisterer;
};

/**
 * check if MACRO is used in GLOBAL NAMESPACE.
 */
#define STATIC_ASSERT_GLOBAL_NAMESPACE(uniq_name, msg)                        \
  struct __test_global_namespace_##uniq_name##__ {};                          \
  static_assert(std::is_same<::__test_global_namespace_##uniq_name##__,       \
                             __test_global_namespace_##uniq_name##__>::value, \
                msg)

// =============== Register for Create ===============
template <DeviceType device_type>
struct EventCreateFunctionRegisterer : public framework::Registrar {
  explicit EventCreateFunctionRegisterer(EventCreateFunction func) {
    auto type_idx = DeviceTypeToId(device_type);
    VLOG(3) << "register event_creator with type_id :" << type_idx;
    DeviceEvent::event_creator_[type_idx] = func;
  }
};

#define REGISTER_EVENT_CREATE_FUNCTION(device_type, func)                   \
  STATIC_ASSERT_GLOBAL_NAMESPACE(                                           \
      __reg_event_creator__##device_type,                                   \
      "REGISTER_EVENT_CREATE_FUNCTION must be called in global namespace"); \
  static ::paddle::platform::EventCreateFunctionRegisterer<device_type>     \
      __reg_event_create_##device_type##__(func);                           \
  int TouchDeviceEventCreate##device_type() {                               \
    __reg_event_create_##device_type##__.Touch();                           \
    return 0;                                                               \
  }

// =============== Register for Record ===============
template <DeviceType device_type>
struct EventRecordFunctionRegisterer : public framework::Registrar {
  explicit EventRecordFunctionRegisterer(EventRecordFunction func) {
    auto type_idx = DeviceTypeToId(device_type);
    VLOG(3) << "register event_recorder with type_id :" << type_idx;
    DeviceEvent::event_recorder_[type_idx] = func;
  }
};

#define REGISTER_EVENT_RECORD_FUNCTION(device_type, func)                   \
  STATIC_ASSERT_GLOBAL_NAMESPACE(                                           \
      __reg_event_recorder__##device_type,                                  \
      "REGISTER_EVENT_RECORD_FUNCTION must be called in global namespace"); \
  static ::paddle::platform::EventRecordFunctionRegisterer<device_type>     \
      __reg_event_record_##device_type##__(func);                           \
  int TouchDeviceEventRecord##device_type() {                               \
    __reg_event_record_##device_type##__.Touch();                           \
    return 0;                                                               \
  }

// =============== Register for Query ===============
template <DeviceType device_type>
struct EventQueryFunctionRegisterer : public framework::Registrar {
  explicit EventQueryFunctionRegisterer(EventQueryFunction func) {
    auto type_idx = DeviceTypeToId(device_type);
    VLOG(3) << "register event_querier with type_id :" << type_idx;
    DeviceEvent::event_querier_[type_idx] = func;
  }
};

#define REGISTER_EVENT_QUERY_FUNCTION(device_type, func)                   \
  STATIC_ASSERT_GLOBAL_NAMESPACE(                                          \
      __reg_event_querier__##device_type,                                  \
      "REGISTER_EVENT_QUERY_FUNCTION must be called in global namespace"); \
  static ::paddle::platform::EventQueryFunctionRegisterer<device_type>     \
      __reg_event_query_##device_type##__(func);                           \
  int TouchDeviceEventQuery##device_type() {                               \
    __reg_event_query_##device_type##__.Touch();                           \
    return 0;                                                              \
  }

// =============== Register for Finish ===============
template <DeviceType device_type>
struct EventFinishFunctionRegisterer : public framework::Registrar {
  explicit EventFinishFunctionRegisterer(EventFinishFunction func) {
    auto type_idx = DeviceTypeToId(device_type);
    VLOG(3) << "register event_finisher with type_id :" << type_idx;
    DeviceEvent::event_finisher_[type_idx] = func;
  }
};

#define REGISTER_EVENT_FINISH_FUNCTION(device_type, func)                   \
  STATIC_ASSERT_GLOBAL_NAMESPACE(                                           \
      __reg_event_finishier__##device_type,                                 \
      "REGISTER_EVENT_FINISH_FUNCTION must be called in global namespace"); \
  static ::paddle::platform::EventFinishFunctionRegisterer<device_type>     \
      __reg_event_finish_##device_type##__(func);                           \
  int TouchDeviceEventFinish##device_type() {                               \
    __reg_event_finish_##device_type##__.Touch();                           \
    return 0;                                                               \
  }

// =============== Register for Wait ===============
template <DeviceType waiter_type, DeviceType event_type>
struct EventWaitFunctionRegisterer : public framework::Registrar {
  explicit EventWaitFunctionRegisterer(EventWaitFunction func) {
    auto waiter_idx = DeviceTypeToId(waiter_type);
    auto event_idx = DeviceTypeToId(event_type);
    VLOG(3) << "register event_finisher with waiter_idx : " << waiter_idx
            << ", event_idx : " << event_idx;
    DeviceEvent::event_waiter_[waiter_idx][event_idx] = func;
  }
};

#define REGISTER_EVENT_WAIT_FUNCTION(waiter_type, event_type, func)       \
  STATIC_ASSERT_GLOBAL_NAMESPACE(                                         \
      __reg_event_waiter__##waiter_type##event_type,                      \
      "REGISTER_EVENT_WAIT_FUNCTION must be called in global namespace"); \
  static ::paddle::platform::EventWaitFunctionRegisterer<waiter_type,     \
                                                         event_type>      \
      __reg_event_wait_##waiter_type##event_type##__(func);               \
  int TouchDeviceEventWait##waiter_type##event_type() {                   \
    __reg_event_wait_##waiter_type##event_type##__.Touch();               \
    return 0;                                                             \
  }

#define USE_EVENT(device_type)                         \
  extern int TouchDeviceEventCreate##device_type();    \
  extern int TouchDeviceEventRecord##device_type();    \
  extern int TouchDeviceEventQuery##device_type();     \
  extern int TouchDeviceEventFinish##device_type();    \
  UNUSED static int use_event_creator_##device_type =  \
      TouchDeviceEventCreate##device_type();           \
  UNUSED static int use_event_recorder_##device_type = \
      TouchDeviceEventRecord##device_type();           \
  UNUSED static int use_event_querier_##device_type =  \
      TouchDeviceEventQuery##device_type();            \
  UNUSED static int use_event_finisher_##device_type = \
      TouchDeviceEventFinish##device_type();

#define USE_EVENT_WAIT(waiter_type, event_type)                  \
  extern int TouchDeviceEventWait##waiter_type##event_type();    \
  UNUSED static int use_event_waiter_##waiter_type##event_type = \
      TouchDeviceEventWait##waiter_type##event_type();

}  // namespace platform
}  // namespace paddle
