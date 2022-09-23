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

#include "paddle/fluid/platform/device/mlu/enforce.h"
#include "paddle/fluid/platform/device/mlu/mlu_info.h"
#include "paddle/fluid/platform/macros.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/stream_callback_manager.h"

namespace paddle {
namespace platform {
namespace stream {

#ifdef PADDLE_WITH_MLU
class MLUStream final {
 public:
  MLUStream() = default;
  explicit MLUStream(const MLUPlace& place, const int priority = 0) {
    Init(place, priority);
  }
  virtual ~MLUStream() { Destroy(); }

  bool Init(const MLUPlace& place, const int priority = 0);

  template <typename Callback>
  void AddCallback(Callback&& callback) const {
    callback_manager_->AddCallback(callback);
  }

  template <typename Callback>
  void RecordEvent(mluEventHandle event, Callback callback) const {
    callback();
    PADDLE_ENFORCE_MLU_SUCCESS(cnPlaceNotifier(event, stream_));
  }

  void RecordEvent(mluEventHandle event) const {
    PADDLE_ENFORCE_MLU_SUCCESS(cnPlaceNotifier(event, stream_));
  }

  void WaitEvent(mluEventHandle event) const {
    PADDLE_ENFORCE_MLU_SUCCESS(cnWaitNotifier(event));
  }

  void Wait() const;
  void WaitCallback() const { callback_manager_->Wait(); }

  const mluStream& raw_stream() const { return stream_; }

  void Destroy();

  bool Query() const {
    cnrtStatus stat = cnrtQueueQuery(stream_);
    if (stat == cnrtSuccess) {
      return true;
    }
    if (stat == cnrtErrorNotReady) {
      return false;
    }
    PADDLE_ENFORCE_MLU_SUCCESS(stat);
    return false;
  }

  void Synchronize() const {
    PADDLE_ENFORCE_MLU_SUCCESS(cnrtQueueSync(stream_));
  }

  const MLUPlace& GetPlace() const { return place_; }

 private:
  MLUPlace place_;
  mluStream stream_{nullptr};
  int priority_{0};
  std::unique_ptr<StreamCallbackManager<mluStream>> callback_manager_;

  DISABLE_COPY_AND_ASSIGN(MLUStream);
};

MLUStream* get_current_mlu_stream(int deviceId);
MLUStream* set_current_mlu_stream(MLUStream* stream);

#endif

}  // namespace stream
}  // namespace platform
}  // namespace paddle
