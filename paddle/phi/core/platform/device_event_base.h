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
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/platform/device_context.h"
#include "paddle/utils/test_macros.h"

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
  explicit DeviceEvent(const phi::Place& place, unsigned int flag)
      : event_(), place_(place), flag_(flag) {
    type_id_ = DeviceTypeToId(platform::Place2DeviceType(place));
    PADDLE_ENFORCE_LT(type_id_,
                      MaxDeviceTypes,
                      common::errors::PreconditionNotMet(
                          "Required type < %d, but received type = %d",
                          MaxDeviceTypes,
                          type_id_));
#ifndef PADDLE_WITH_CUSTOM_DEVICE
    // TODO(Aurelius84): only support CPU/CUDA.
    PADDLE_ENFORCE_LT(type_id_,
                      3,
                      common::errors::Unavailable(
                          "Currently DeviceEvent do not support %s", place));
#endif
    PADDLE_ENFORCE_NOT_NULL(
        event_creator_[type_id_],
        common::errors::Unavailable("event_creator_[%d] shall not be nullptr.",
                                    type_id_));
    event_creator_[type_id_](this, place, flag);
  }

  ~DeviceEvent() {}

  void Record(const DeviceContext* dev_ctx) {
    PADDLE_ENFORCE_NOT_NULL(
        event_recorder_[type_id_],
        common::errors::Unavailable("event_recorder_[%d] shall not be nullptr.",
                                    type_id_));
    if (!recorded_) {
      recorded_ = true;
    }
    event_recorder_[type_id_](this, dev_ctx);
  }

  bool Query() {
    PADDLE_ENFORCE_NOT_NULL(
        event_querier_[type_id_],
        common::errors::Unavailable("event_querier_[%d] shall not be nullptr.",
                                    type_id_));
    if (!recorded_) {
      VLOG(4) << "Event " << this << " is not recorded yet, and skip query!";
      return true;
    }
    return event_querier_[type_id_](this);
  }

  void Finish() const {
    PADDLE_ENFORCE_NOT_NULL(
        event_finisher_[type_id_],
        common::errors::Unavailable("event_finisher_[%d] shall not be nullptr.",
                                    type_id_));
    event_finisher_[type_id_](this);
  }

  void SetFinished() {
    PADDLE_ENFORCE_NOT_NULL(
        event_finished_setter_[type_id_],
        common::errors::Unavailable(
            "event_finished_setter_[%d] shall not be nullptr.", type_id_));
    event_finished_setter_[type_id_](this);
  }

  void Reset() {
    PADDLE_ENFORCE_NOT_NULL(
        event_resetter_[type_id_],
        common::errors::Unavailable("event_resetter_[%d] shall not be nullptr.",
                                    type_id_));
    event_resetter_[type_id_](this);
  }

  void Wait(const DeviceType& waiter_type, const DeviceContext* context) const {
    auto waiter_idx = DeviceTypeToId(waiter_type);
    PADDLE_ENFORCE_NOT_NULL(event_waiter_[waiter_idx][type_id_],
                            common::errors::Unavailable(
                                "event_waiter_[%d][%d] shall not be nullptr.",
                                waiter_idx,
                                type_id_));
    if (!recorded_) {
      VLOG(4) << "Event " << this << " is not recorded yet, and skip wait!";
      return;
    }
    event_waiter_[waiter_idx][type_id_](this, context);
  }

  void InitEvent(std::shared_ptr<void> event) { event_ = event; }

  std::shared_ptr<void> GetEvent() const { return event_; }

 private:
  std::shared_ptr<void> event_;
  phi::Place place_;
  int type_id_;
  unsigned int flag_;

  // NOTE(chenruibiao): In cross-step stream synchronization, an event may be
  // recorded in the first step and waited in the second step. So, in the first
  // step, the WaitEvent may be called without RecordEvent.
  // On cuda device, it is ok to wait event that is not recorded yet;
  // while on npu device, it results in error.
  // So, we add flag recorded_ to handle this case uniformly.
  bool recorded_{false};

  static EventCreateFunction event_creator_[MaxDeviceTypes];
  static EventRecordFunction event_recorder_[MaxDeviceTypes];
  static EventQueryFunction event_querier_[MaxDeviceTypes];
  static EventFinishFunction event_finisher_[MaxDeviceTypes];
  static EventSetFinishedFunction event_finished_setter_[MaxDeviceTypes];
  static EventWaitFunction event_waiter_[MaxDeviceTypes][MaxDeviceTypes];
  static EventResetFunction event_resetter_[MaxDeviceTypes];

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
    DeviceEvent::event_creator_[type_idx] = func;
  }
};

#define REGISTER_EVENT_CREATE_FUNCTION(device_type, func)                   \
  STATIC_ASSERT_GLOBAL_NAMESPACE(                                           \
      __reg_event_creator__##device_type,                                   \
      "REGISTER_EVENT_CREATE_FUNCTION must be called in global namespace"); \
  static ::paddle::platform::EventCreateFunctionRegisterer<device_type>     \
      __reg_event_create_##device_type##__(func);                           \
  TEST_API int TouchDeviceEventCreate##device_type() {                      \
    __reg_event_create_##device_type##__.Touch();                           \
    return 0;                                                               \
  }

// =============== Register for Record ===============
template <DeviceType device_type>
struct EventRecordFunctionRegisterer : public framework::Registrar {
  explicit EventRecordFunctionRegisterer(EventRecordFunction func) {
    auto type_idx = DeviceTypeToId(device_type);
    DeviceEvent::event_recorder_[type_idx] = func;
  }
};

#define REGISTER_EVENT_RECORD_FUNCTION(device_type, func)                   \
  STATIC_ASSERT_GLOBAL_NAMESPACE(                                           \
      __reg_event_recorder__##device_type,                                  \
      "REGISTER_EVENT_RECORD_FUNCTION must be called in global namespace"); \
  static ::paddle::platform::EventRecordFunctionRegisterer<device_type>     \
      __reg_event_record_##device_type##__(func);                           \
  TEST_API int TouchDeviceEventRecord##device_type() {                      \
    __reg_event_record_##device_type##__.Touch();                           \
    return 0;                                                               \
  }

// =============== Register for Query ===============
template <DeviceType device_type>
struct EventQueryFunctionRegisterer : public framework::Registrar {
  explicit EventQueryFunctionRegisterer(EventQueryFunction func) {
    auto type_idx = DeviceTypeToId(device_type);
    DeviceEvent::event_querier_[type_idx] = func;
  }
};

#define REGISTER_EVENT_QUERY_FUNCTION(device_type, func)                   \
  STATIC_ASSERT_GLOBAL_NAMESPACE(                                          \
      __reg_event_querier__##device_type,                                  \
      "REGISTER_EVENT_QUERY_FUNCTION must be called in global namespace"); \
  static ::paddle::platform::EventQueryFunctionRegisterer<device_type>     \
      __reg_event_query_##device_type##__(func);                           \
  TEST_API int TouchDeviceEventQuery##device_type() {                      \
    __reg_event_query_##device_type##__.Touch();                           \
    return 0;                                                              \
  }

// =============== Register for Finish ===============
template <DeviceType device_type>
struct EventFinishFunctionRegisterer : public framework::Registrar {
  explicit EventFinishFunctionRegisterer(EventFinishFunction func) {
    auto type_idx = DeviceTypeToId(device_type);
    DeviceEvent::event_finisher_[type_idx] = func;
  }
};

#define REGISTER_EVENT_FINISH_FUNCTION(device_type, func)                   \
  STATIC_ASSERT_GLOBAL_NAMESPACE(                                           \
      __reg_event_finishier__##device_type,                                 \
      "REGISTER_EVENT_FINISH_FUNCTION must be called in global namespace"); \
  static ::paddle::platform::EventFinishFunctionRegisterer<device_type>     \
      __reg_event_finish_##device_type##__(func);                           \
  TEST_API int TouchDeviceEventFinish##device_type() {                      \
    __reg_event_finish_##device_type##__.Touch();                           \
    return 0;                                                               \
  }

// =============== Register for SetFinished ===============
template <DeviceType device_type>
struct EventSetFinishedFunctionRegisterer : public framework::Registrar {
  explicit EventSetFinishedFunctionRegisterer(EventSetFinishedFunction func) {
    auto type_idx = DeviceTypeToId(device_type);
    DeviceEvent::event_finished_setter_[type_idx] = func;
  }
};

#define REGISTER_EVENT_SET_FINISHED_FUNCTION(device_type, func)              \
  STATIC_ASSERT_GLOBAL_NAMESPACE(                                            \
      __reg_event_finished_setter__##device_type,                            \
      "REGISTER_EVENT_FINISH_FUNCTION must be called in global namespace");  \
  static ::paddle::platform::EventSetFinishedFunctionRegisterer<device_type> \
      __reg_event_finished_setter_##device_type##__(func);                   \
  TEST_API int TouchDeviceEventSetFinished##device_type() {                  \
    __reg_event_finished_setter_##device_type##__.Touch();                   \
    return 0;                                                                \
  }

// =============== Register for Wait ===============
template <DeviceType waiter_type, DeviceType event_type>
struct EventWaitFunctionRegisterer : public framework::Registrar {
  explicit EventWaitFunctionRegisterer(EventWaitFunction func) {
    auto waiter_idx = DeviceTypeToId(waiter_type);
    auto event_idx = DeviceTypeToId(event_type);
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
  TEST_API int TouchDeviceEventWait##waiter_type##event_type() {          \
    __reg_event_wait_##waiter_type##event_type##__.Touch();               \
    return 0;                                                             \
  }

// =============== Register for Reset ===============
template <DeviceType device_type>
struct EventResetFunctionRegisterer : public framework::Registrar {
  explicit EventResetFunctionRegisterer(EventResetFunction func) {
    auto type_idx = DeviceTypeToId(device_type);
    DeviceEvent::event_resetter_[type_idx] = func;
  }
};

#define REGISTER_EVENT_RESET_FUNCTION(device_type, func)                   \
  STATIC_ASSERT_GLOBAL_NAMESPACE(                                          \
      __reg_event_resetter__##device_type,                                 \
      "REGISTER_EVENT_RESET_FUNCTION must be called in global namespace"); \
  static ::paddle::platform::EventResetFunctionRegisterer<device_type>     \
      __reg_event_resetter_##device_type##__(func);                        \
  TEST_API int TouchDeviceEventReset##device_type() {                      \
    __reg_event_resetter_##device_type##__.Touch();                        \
    return 0;                                                              \
  }

#define USE_EVENT(device_type)                                \
  extern int TouchDeviceEventCreate##device_type();           \
  extern int TouchDeviceEventRecord##device_type();           \
  extern int TouchDeviceEventQuery##device_type();            \
  extern int TouchDeviceEventFinish##device_type();           \
  extern int TouchDeviceEventSetFinished##device_type();      \
  extern int TouchDeviceEventReset##device_type();            \
  UNUSED static int use_event_creator_##device_type =         \
      TouchDeviceEventCreate##device_type();                  \
  UNUSED static int use_event_recorder_##device_type =        \
      TouchDeviceEventRecord##device_type();                  \
  UNUSED static int use_event_querier_##device_type =         \
      TouchDeviceEventQuery##device_type();                   \
  UNUSED static int use_event_finisher_##device_type =        \
      TouchDeviceEventFinish##device_type();                  \
  UNUSED static int use_event_finished_setter_##device_type = \
      TouchDeviceEventSetFinished##device_type();             \
  UNUSED static int use_event_resetter_##device_type =        \
      TouchDeviceEventReset##device_type();

#define USE_EVENT_WAIT(waiter_type, event_type)                  \
  extern int TouchDeviceEventWait##waiter_type##event_type();    \
  UNUSED static int use_event_waiter_##waiter_type##event_type = \
      TouchDeviceEventWait##waiter_type##event_type();

}  // namespace platform
}  // namespace paddle
