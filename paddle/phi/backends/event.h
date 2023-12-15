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

#pragma once
#include "paddle/common/macros.h"
#include "paddle/phi/common/place.h"

namespace phi {

class Device;

namespace stream {
class Stream;
}  // namespace stream

namespace event {
using event_t = void*;

class Event {
 public:
  enum Flag {
    Default = 0x0,
    BlockingSync = 0x1,
    DisableTiming = 0x2,
    Interprocess = 0x4,
  };

  Event() = default;
  // For compatible
  Event(const Place& place, event_t event);
  ~Event();
  event_t raw_event() const;
  void set_event(event_t event);
  bool Init(const Place& place, Flag flags = Flag::Default);
  void Destroy();
  void Record(const stream::Stream* stream);
  bool Query() const;
  void Synchronize() const;
  const Place& GetPlace() const;

  static void ReleaseAll();

 private:
  DISABLE_COPY_AND_ASSIGN(Event);
  Place place_;
  Device* device_;
  event_t event_;
  bool own_data_ = true;
  mutable bool is_recorded_ = false;
};
}  // namespace event

}  // namespace phi
