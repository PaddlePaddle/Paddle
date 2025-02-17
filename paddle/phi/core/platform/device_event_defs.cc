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

#include "paddle/phi/core/platform/device_event_defs.h"
#include "glog/logging.h"

namespace paddle {
namespace platform {

DeviceEvent::DeviceEvent(const phi::Place& place, unsigned int flag)
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

void DeviceEvent::Record(const DeviceContext* dev_ctx) {
  PADDLE_ENFORCE_NOT_NULL(
      event_recorder_[type_id_],
      common::errors::Unavailable("event_recorder_[%d] shall not be nullptr.",
                                  type_id_));
  if (!recorded_) {
    recorded_ = true;
  }
  event_recorder_[type_id_](this, dev_ctx);
}

bool DeviceEvent::Query() {
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

void DeviceEvent::Finish() const {
  PADDLE_ENFORCE_NOT_NULL(
      event_finisher_[type_id_],
      common::errors::Unavailable("event_finisher_[%d] shall not be nullptr.",
                                  type_id_));
  event_finisher_[type_id_](this);
}

void DeviceEvent::SetFinished() {
  PADDLE_ENFORCE_NOT_NULL(
      event_finished_setter_[type_id_],
      common::errors::Unavailable(
          "event_finished_setter_[%d] shall not be nullptr.", type_id_));
  event_finished_setter_[type_id_](this);
}

void DeviceEvent::Reset() {
  PADDLE_ENFORCE_NOT_NULL(
      event_resetter_[type_id_],
      common::errors::Unavailable("event_resetter_[%d] shall not be nullptr.",
                                  type_id_));
  event_resetter_[type_id_](this);
}

void DeviceEvent::Wait(const DeviceType& waiter_type,
                       const DeviceContext* context) const {
  auto waiter_idx = DeviceTypeToId(waiter_type);
  PADDLE_ENFORCE_NOT_NULL(
      event_waiter_[waiter_idx][type_id_],
      common::errors::Unavailable(
          "event_waiter_[%d][%d] shall not be nullptr.", waiter_idx, type_id_));
  if (!recorded_) {
    VLOG(4) << "Event " << this << " is not recorded yet, and skip wait!";
    return;
  }
  event_waiter_[waiter_idx][type_id_](this, context);
}

}  // namespace platform
}  // namespace paddle
