// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/new_executor/profiler.h"

namespace paddle {
namespace framework {
namespace profiler {

OpProfileEvent::OpProfileEvent(const platform::DeviceContext* device_context) {}

OpProfileEvent::~OpProfileEvent() {}

void OpProfileEvent::Record() {
  platform::Timer timer;
  timeval _now;
  gettimeofday(&_now, nullptr);
  uint64_t us_since_epoch = static_cast<uint64_t>(_now.tv_sec) * 1000 * 1000 +
                            static_cast<uint64_t>(_now.tv_usec);
  this->event_obj_cpu_.event_time_us_ = us_since_epoch;
}

std::tuple<double, double> OpProfileEvent::MeasureTimeLapseWrtOtherEvent(
    const OpProfileEvent& end_event) const {
  double device_time_lapse_us = -1.0;

  // measure CPU time cost
  uint64_t cpu_time_lapse_us = end_event.event_obj_cpu_.event_time_us_ -
                               this->event_obj_cpu_.event_time_us_;
  return std::make_tuple(static_cast<double>(cpu_time_lapse_us),
                         device_time_lapse_us);
}

OpRuntimeProfiler::OpRuntimeProfiler() {}

OpRuntimeProfiler::~OpRuntimeProfiler() {}

void OpRuntimeProfiler::RecordEvent(
    const std::string& event_name,
    const platform::DeviceContext* device_context) {
  if (this->name_to_device_profile_events_.find(event_name) !=
      this->name_to_device_profile_events_.end()) {
    VLOG(1) << "An event with name \"" << event_name
            << "\" already exists. "
               "RecordEvent will overwrite this event.";
  }
  // create a new event and record
  std::shared_ptr event_ptr = std::make_shared<OpProfileEvent>(device_context);
  event_ptr->Record();
  this->name_to_device_profile_events_.insert_or_assign(event_name, event_ptr);
}

std::tuple<double, double> OpRuntimeProfiler::MeasureTimeLapseBetweenEvents(
    const std::string& start_event_name,
    const std::string& end_event_name) const {
  if (this->name_to_device_profile_events_.find(start_event_name) ==
          this->name_to_device_profile_events_.end() ||
      this->name_to_device_profile_events_.find(end_event_name) ==
          this->name_to_device_profile_events_.end()) {
    VLOG(1) << "One of the event name is not found. " << start_event_name << " "
            << end_event_name;
    return std::make_tuple(-1.0, -1.0);
  }
  std::shared_ptr<OpProfileEvent> start_event =
      this->name_to_device_profile_events_.at(start_event_name);
  std::shared_ptr<OpProfileEvent> end_event =
      this->name_to_device_profile_events_.at(end_event_name);
  VLOG(6) << "Trying to meansure time lapse between event \""
          << start_event_name << "\" and \"" << end_event_name << "\"";
  return start_event->MeasureTimeLapseWrtOtherEvent(*end_event);
}

}  // namespace profiler
}  // namespace framework
}  // namespace paddle
