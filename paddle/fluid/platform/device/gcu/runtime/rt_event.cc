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

#include "paddle/fluid/platform/device/gcu/runtime/rt_event.h"

#include <memory>

namespace paddle {
namespace platform {
namespace gcu {
namespace runtime {

std::shared_ptr<Event> Event::CreateNativeEvent(Context *ctx) {
  GcuDeviceGuard guard(ctx->device);
  topsEvent_t e;
  RT_CHECK(topsEventCreateWithFlags(&e, topsEventRecordOnce));
  return std::make_shared<Event>(ctx, e);
}

void Event::Synchronize() {
  GcuDeviceGuard guard(ctx->device);
  RT_CHECK(topsEventSynchronize(tops_event));
}

Event::Event(Context *ctx, topsEvent_t event) : ctx(ctx), tops_event(event) {}

Event::~Event() {
  GcuDeviceGuard guard(ctx->device);
  RT_CHECK_NO_THROW(topsEventDestroy(tops_event));
}

}  // namespace runtime
}  // namespace gcu
}  // namespace platform
}  // namespace paddle
