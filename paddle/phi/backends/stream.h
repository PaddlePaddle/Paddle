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

#include "paddle/phi/backends/callback_manager.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/macros.h"

namespace phi {

class Device;

namespace event {
class Event;
}  // namespace event

namespace stream {
using stream_t = void*;
class Stream {
 public:
  enum class Priority : uint8_t {
    kNull = 0x0,
    kHigh = 0x1,
    kNormal = 0x2,
  };

  enum class Flag : uint8_t {
    kDefaultFlag = 0x0,
    kStreamNonBlocking = 0x1,
  };

  using Callback = std::function<void()>;

  Stream() = default;
  // For compatiable
  Stream(const Place& place, stream_t stream);
  ~Stream();
  const stream_t& raw_stream() const;
  void set_stream(stream_t stream);
  bool Init(const Place& place,
            const Priority& priority = Priority::kNormal,
            const Flag& flag = Flag::kDefaultFlag);
  template <typename Callback>
  void AddCallback(Callback&& callback) const {
    callback_manager_->AddCallback(callback);
  }
  void RecordEvent(event::Event* event, Callback callback) const;
  void RecordEvent(event::Event* event) const;
  void WaitEvent(event::Event* event) const;
  void Wait() const;
  void WaitCallback() const;
  void Destroy();
  bool Query() const;
  void Synchronize() const;
  const Place& GetPlace() const;

 private:
  DISABLE_COPY_AND_ASSIGN(Stream);
  Place place_;
  Device* device_;
  stream_t stream_;
  std::unique_ptr<CallbackManager> callback_manager_;
  bool own_data_ = true;
};

}  // namespace stream
}  // namespace phi
