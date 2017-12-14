/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/platform/profiler.h"

namespace paddle {
namespace platform {

ProfilerState kState = ProfilerState::kDisabled;
uint32_t kNextThreadId = 0;
std::mutex kAllEventListsMutex;
std::list<std::shared_ptr<EventList>> kAllEventLists;
thread_local std::shared_ptr<EventList> kEventList;
thread_local int32_t kThreadId;

void EnableProfiler(ProfilerState state) {
  PADDLE_ENFORCE(state != ProfilerState::kDisabled,
                 "Can't enbale profling, since the input state is ",
                 "ProfilerState::kDisabled");
  PADDLE_ENFORCE(kState == ProfilerState::kDisabled,
                 "The profiling state should be disabled when calling ",
                 "EnableProfiler.");
  kState = state;
#ifdef PADDLE_WITH_CUDA
  auto ForEachDevice = [](std::function<void(int)> op) {
    int count = GetCUDADeviceCount();
    for (int i = 0; i < count; i++) {
      DeviceGuard dev_guard(i);
      op(i);
    }
  };
  if (kState == ProfilerState::kCUDA) {
    // Generate some dummy evenets first to reduce the startup overhead.
    for (int i = 0; i < 5; i++) {
      ForEachDevice([](int d) {
        DeviceContext* dev_ctx = new CUDADeviceContext(GPUPlace(d));
        Mark("_cuda_startup_", dev_ctx);
        dev_ctx->Wait();
      });
    }
  }
#endif
  // Mark the profiling start.
  Mark("_start_profiler_");
}

std::vector<std::vector<Event>> DisableProfiler() {
  PADDLE_ENFORCE(kState != ProfilerState::kDisabled,
                 "Can't disable profiling, since it's not starting.");
  // Mark the profiling stop.
  Mark("_stop_profiler_");
  kState = ProfilerState::kDisabled;
  std::vector<std::vector<Event>> result;
  std::lock_guard<std::mutex> guard(kAllEventListsMutex);
  for (auto it = kAllEventLists.begin(); it != kAllEventLists.end(); ++it) {
    auto& list = *it;
    result.emplace_back(list->Reduce());
  }
  return result;
}

}  // namespace platform
}  // namespace paddle
