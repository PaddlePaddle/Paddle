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
#include <atomic>
#include <condition_variable>
#include <mutex>

#include "paddle/phi/core/platform/device_event_base.h"

namespace paddle {
namespace platform {

struct CPUDeviceEventWrapper {
  explicit CPUDeviceEventWrapper(const phi::Place& place, unsigned int flag = 0)
      : status_(EventStatus::INITIALIZED) {
    PADDLE_ENFORCE_EQ(
        phi::is_cpu_place(place),
        true,
        common::errors::PreconditionNotMet(
            "Required device shall be CPUPlace, but received %d. ", place));
  }
  std::mutex mutex_;
  std::condition_variable cv_completed_;
  std::atomic<int> status_;
};

void DeviceEventCreateCPU(DeviceEvent* event, const phi::Place& place);

void DeviceEventRecordCPU(DeviceEvent* event,
                          const phi::Place& place,
                          const DeviceContext* context);

bool DeviceEventQueryCPU(const DeviceEvent* event);

void DeviceEventFinishCPU(const DeviceEvent* event);

void EventSetFinishedCPU(const DeviceEvent* event);

void DeviceEventCPUWaitCPU(const DeviceEvent* event, DeviceContext* context);

}  // namespace platform
}  // namespace paddle
