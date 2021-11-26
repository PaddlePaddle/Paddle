/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/fluid/platform/mlu/mlu_info.h"
#include "paddle/fluid/platform/macros.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace platform {
namespace stream {

#ifdef PADDLE_WITH_MLU
class MLUStream final {
 public:
  MLUStream() = default;
  explicit MLUStream(const Place& place,
                     const int priority = 0) {
    // TODO(mlu): adapt init queue
  }
  virtual ~MLUStream() { Destroy(); }

  bool Init(const Place& place, const int priority = 0);

  template <typename Callback>
  void AddCallback(Callback&& callback) const {
    // TODO(mlu): support AddCallback
    // callback_manager_->AddCallback(callback);
  }

  template <typename Callback>
  void RecordEvent(mluEventHandle event, Callback callback) const {
    callback();
    // TODO(mlu): support notifier and event
    // PADDLE_ENFORCE_MLU_SUCCESS(cnPlaceNotifier(event, stream_));
  }

  void RecordEvent(mluEventHandle event) const {
    // TODO(mlu): support notifier and event
    // PADDLE_ENFORCE_MLU_SUCCESS(cnPlaceNotifier(event, stream_));
  }

  void WaitEvent(mluEventHandle event) const {
    // TODO(mlu): support notifier and event
    // PADDLE_ENFORCE_MLU_SUCCESS(cnWaitNotifier(event));
  }

  void Wait() const;
  void WaitCallback() const { /*callback_manager_->Wait();*/ }

  const mluStream& raw_stream() const { return stream_; }
  void Destroy();

  bool Query() const {
    // TODO(mlu): adapt query queue
    return true;
  }

  void Synchronize() const {
    // TODO(mlu): adapt Sync queue
    // PADDLE_ENFORCE_MLU_SUCCESS(cnQueueSync(stream_));
  }

  const Place& GetPlace() const { return place_; }

 private:
  Place place_;
  mluStream stream_{nullptr};
  int priority_{0};
  // std::unique_ptr<StreamCallbackManager<mluStream>> callback_manager_;

  DISABLE_COPY_AND_ASSIGN(MLUStream);
};

MLUStream* get_current_mlu_stream(int deviceId);
MLUStream* set_current_mlu_stream(MLUStream* stream);

#endif

}  // namespace stream
}  // namespace platform
}  // namespace paddle
