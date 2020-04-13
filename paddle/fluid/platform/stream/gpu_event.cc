/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/platform/stream/gpu_event.h"
#include "paddle/fluid/framework/details/stream_executor_gpu.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/stream/gpu_stream.h"

namespace paddle {
namespace platform {
namespace stream {

// namespace pf = paddle::framework;

Event::Event(pfd::StreamExecutor* pe)
    : implementation_(pe->CreateEventImplementation()), pe_(pe) {}

Event::~Event() {
  if (pe_ && implementation_) {
    pe_->DeleteEvent(this);
  }
}

bool Event::Init() {
  if (pe_) {
    PADDLE_ENFORCE_EQ(pe_->AllocateEvent(this), true, "shoule init ok");
  }
  return true;
}

Event::Status Event::PollForStatus() { return pe_->PollForStatus(this); }

}  // namespace stream
}  // namespace platform
}  // namespace paddle
