/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include <cstdint>
#include <memory>

#include "paddle/fluid/platform/device/npu/npu_info.h"
#include "paddle/fluid/platform/macros.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/stream_callback_manager.h"

namespace paddle {
namespace platform {
namespace stream {

#ifdef PADDLE_WITH_ASCEND_CL

class NPUStream final {
 public:
  NPUStream() = default;
  explicit NPUStream(const Place& place) { Init(place); }
  virtual ~NPUStream() { Destroy(); }

  bool Init(const Place& place);

  template <typename Callback>
  void AddCallback(Callback&& callback) const {
    callback_manager_->AddCallback(callback);
  }

  template <typename Callback>
  void RecordEvent(aclrtEvent ev, Callback callback) const {
    callback();
    NPUEventRecord(ev, stream_);
  }

  void RecordEvent(aclrtEvent ev) const { NPUEventRecord(ev, stream_); }

  void WaitEvent(aclrtEvent ev) const { NPUStreamWaitEvent(stream_, ev); }

  void Wait() const;
  void WaitCallback() const { callback_manager_->Wait(); }

  aclrtStream raw_stream() const { return stream_; }
  void Destroy();

 private:
  Place place_;
  aclrtStream stream_{nullptr};
  std::unique_ptr<StreamCallbackManager<aclrtStream>> callback_manager_;

  DISABLE_COPY_AND_ASSIGN(NPUStream);
};

#endif

}  // namespace stream
}  // namespace platform
}  // namespace paddle
