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

#include "paddle/phi/backends/stream.h"

#include "glog/logging.h"

#include "paddle/phi/backends/device_guard.h"
#include "paddle/phi/backends/event.h"

namespace phi::stream {

std::list<Stream*> g_streams;
std::mutex g_streams_mutex;

void Stream::ReleaseAll() {
  std::unique_lock lock(g_streams_mutex);
  for (auto* stream : g_streams) {
    stream->Destroy();
  }
}

Stream::~Stream() {
  Destroy();
  std::unique_lock lock(g_streams_mutex);
  g_streams.remove(this);
}

const stream_t& Stream::raw_stream() const { return stream_; }

void Stream::set_stream(stream_t stream) { stream_ = stream; }

// For compatible
Stream::Stream(const Place& place, stream_t stream)
    : place_(place),
      device_(phi::DeviceManager::GetDeviceWithPlace(place)),
      stream_(stream),
      callback_manager_(new CallbackManager(this)),
      own_data_(false) {}

bool Stream::Init(const Place& place,
                  const Priority& priority,
                  const Flag& flag) {
  place_ = place;
  device_ = phi::DeviceManager::GetDeviceWithPlace(place);

  // note(wangran16): bind device to the current thread. fix npu plugin null
  // context bug.
  phi::DeviceManager::SetDevice(place_);
  device_->CreateStream(this, priority, flag);

  callback_manager_ = std::make_unique<CallbackManager>(this);
  VLOG(3) << "Init Stream: " << stream_ << ", place: " << place_
          << ", priority: " << static_cast<int>(priority)
          << ", flag:" << static_cast<int>(flag);
  own_data_ = true;
  std::unique_lock lock(g_streams_mutex);
  g_streams.push_back(this);
  return true;
}

void Stream::RecordEvent(event::Event* event, Callback callback) const {
  callback();
  device_->RecordEvent(event, this);
}

void Stream::RecordEvent(event::Event* event) const {
  device_->RecordEvent(event, this);
}

void Stream::WaitEvent(event::Event* event) const {
  device_->StreamWaitEvent(this, event);
}

void Stream::Wait() const {
#if !defined(_WIN32)
  device_->SynchronizeStream(this);
#else
  while (1) {
    if (device_->QueryStream(this)) {
      break;
    }
  }
#endif
}

void Stream::WaitCallback() const { callback_manager_->Wait(); }

void Stream::Destroy() {
  if (device_) {
    if (own_data_ &&
        phi::DeviceManager::HasDeviceType(place_.GetDeviceType())) {
      phi::DeviceManager::SetDevice(place_);
      device_->DestroyStream(this);
    }
    own_data_ = false;
    stream_ = nullptr;
    device_ = nullptr;
  }
}

bool Stream::Query() const { return device_->QueryStream(this); }

void Stream::Synchronize() const { device_->SynchronizeStream(this); }

const Place& Stream::GetPlace() const { return place_; }

}  // namespace phi::stream
