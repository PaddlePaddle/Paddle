/* Copyright (c) 2023 Enflame. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <memory>

#include "paddle/fluid/platform/device/gcu/runtime/rt_context.h"

namespace paddle {
namespace platform {
namespace gcu {
namespace runtime {

struct Event {
  Context *ctx;
  topsEvent_t tops_event;

  static std::shared_ptr<Event> CreateNativeEvent(Context *ctx);

  void Synchronize();

  Event(Context *ctx, topsEvent_t event);

  ~Event();

  Event() = delete;
  RT_DISALLOW_COPY_AND_ASSIGN(Event);
};

}  // namespace runtime
}  // namespace gcu
}  // namespace platform
}  // namespace paddle
