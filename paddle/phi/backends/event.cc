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

#include <list>
#include <mutex>

#include "paddle/phi/backends/event.h"

#include "glog/logging.h"

#include "paddle/phi/backends/device_guard.h"
#include "paddle/phi/backends/stream.h"

namespace phi::event {

std::list<Event*> g_events;
std::mutex g_events_mutex;

void Event::ReleaseAll() {
  std::unique_lock lock(g_events_mutex);
  for (auto* event : g_events) {
    event->Destroy();
  }
}

event_t Event::raw_event() const { return event_; }

void Event::set_event(event_t event) { event_ = event; }

Event::Event(const Place& place, event_t event)
    : place_(place),
      device_(phi::DeviceManager::GetDeviceWithPlace(place)),
      event_(event),
      own_data_(false) {}

Event::~Event() {
  Synchronize();
  Destroy();
  std::unique_lock lock(g_events_mutex);
  g_events.remove(this);
}

bool Event::Init(const Place& place, Flag flags) {
  place_ = place;
  device_ = phi::DeviceManager::GetDeviceWithPlace(place);

  // note(wangran16): bind device to the current thread. fix npu plugin null
  // context bug.
  phi::DeviceManager::SetDevice(place_);
  device_->CreateEvent(this, flags);
  VLOG(3) << "Init Event: " << event_ << ", place: " << place_
          << ", flag:" << static_cast<int>(flags);
  own_data_ = true;
  std::unique_lock lock(g_events_mutex);
  g_events.push_back(this);
  return true;
}

void Event::Destroy() {
  if (device_) {
    if (own_data_ &&
        phi::DeviceManager::HasDeviceType(place_.GetDeviceType())) {
      phi::DeviceManager::SetDevice(place_);
      device_->DestroyEvent(this);
    }
    own_data_ = false;
    event_ = nullptr;
    device_ = nullptr;
    is_recorded_ = false;
  }
}

void Event::Record(const stream::Stream* stream) {
  if (device_) {
    is_recorded_ = true;  // synchronize the event during destroy
    stream->RecordEvent(this);
  }
}

bool Event::Query() const {
  if (device_ && is_recorded_) {
    bool ret = device_->QueryEvent(this);
    if (ret) {
      is_recorded_ =
          false;  // event completed, do not need to synchronize the event.
    }
    return ret;
  } else {
    return true;
  }
}

void Event::Synchronize() const {
  if (device_ && is_recorded_) {
    device_->SynchronizeEvent(this);
  }
}

const Place& Event::GetPlace() const { return place_; }

}  // namespace phi::event
