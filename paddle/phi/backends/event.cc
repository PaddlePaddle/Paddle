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

#include "paddle/phi/backends/event.h"
#include "paddle/fluid/platform/device/device_wrapper.h"
#include "paddle/phi/backends/device_guard.h"
#include "paddle/phi/backends/stream.h"

namespace phi {
namespace event {

event_t Event::raw_event() const { return event_; }

void Event::set_event(event_t event) { event_ = event; }

Event::Event(const Place& place, event_t event)
    : place_(place),
      device_(phi::DeviceManager::GetDeviceWithPlace(place)),
      event_(event),
      own_data_(false) {}

Event::~Event() { Destroy(); }

bool Event::Init(const Place& place, Flag flags) {
  place_ = place;
  DeviceGuard guard(place_);
  device_->CreateEvent(this, flags);
  VLOG(3) << "Init Event: " << event_ << ", place: " << place_
          << ", flag:" << static_cast<int>(flags);
  own_data_ = true;
  return true;
}

void Event::Destroy() {
  if (own_data_) {
    DeviceGuard guard(place_);
    device_->DestroyEvent(this);
    own_data_ = false;
  }
}

void Event::Record(const stream::Stream* stream) { stream->RecordEvent(this); }

bool Event::Query() const { return device_->QueryEvent(this); }

void Event::Synchonrize() const { device_->SynchronizeEvent(this); }

const Place& Event::GetPlace() const { return place_; }

}  // namespace event
}  // namespace phi
