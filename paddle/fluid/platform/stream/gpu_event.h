/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

licensed under the Apache License, Version 2.0 (the "License");
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
#include "paddle/fluid/platform/stream/stream_internal.h"

namespace paddle {
namespace framework {
namespace details {
class StreamExecutor;
}
}

namespace platform {
namespace stream {

namespace internal {
class StreamInterface;
class EventInterface;
}

namespace pfd = paddle::framework::details;

class Event {
 public:
  // Potential states for an Event. If PollForStatus() returns anything aside
  // from kPending or kComplete, an error has occurred; kUnknown is a bad state.
  // Not all implementations are able to return all enumeration values. Refer to
  // the platform-specific implementation for details.
  enum class Status {
    kUnknown,
    kError,
    kPending,
    kComplete,
  };

  explicit Event(pfd::StreamExecutor* pe);  // NOLINT

  // Releases any resources held by the Event object.
  ~Event();

  // Performs any platform-specific or potentially error-generating
  // initialization.
  bool Init();

  // Returns the current Status for the event.
  Status PollForStatus();

  // Returns a pointer to the underlying platform-specific implementation.
  internal::EventInterface* implementation() { return implementation_.get(); }

  Event(Event&&) = default;
  Event& operator=(Event&&) = default;

 private:
  friend class BaseStream;

  std::unique_ptr<internal::EventInterface> implementation_;
  // StreamExecutor* stream_exec_;
  pfd::StreamExecutor* pe_;
  // DeviceContext* ctx_;

  DISALLOW_COPY_AND_ASSIGN(Event);
};

}  // namespace stream
}  // namespace platform
}  // namespace paddle
