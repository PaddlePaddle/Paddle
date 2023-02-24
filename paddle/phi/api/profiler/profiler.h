/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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
#include <forward_list>
#include <list>
#include <map>
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "gflags/gflags.h"

#include "paddle/phi/api/profiler/event_tracing.h"
#include "paddle/phi/api/profiler/supplement_tracing.h"

DECLARE_bool(enable_host_event_recorder_hook);

namespace phi {

enum class ProfilerState {
  kDisabled,  // disabled state
  kCPU,       // CPU profiling state
  kCUDA,      // GPU profiling state
  kAll,       // Profile both CPU and GPU. (Currently experimental).
};

// it is the flag to control to print the profiling result
enum class TracerOption {
  kDefault,      // print the different op type profiling result
  kOpDetail,     // print the detail profiling result of different op type
  kAllOpDetail,  // print the detail profiling result of different op name
};

template <typename T>
struct EventList {
  constexpr static size_t kMB = 1024 * 1024;
  constexpr static size_t kEventBlockSize = 16 * kMB;
  constexpr static size_t kEventSize = sizeof(T);
  constexpr static size_t kEventAlign = alignof(T);
  constexpr static size_t kNumBlock =
      kEventBlockSize /
      ((kEventSize + kEventAlign - 1) / kEventAlign * kEventAlign);

  template <typename... Args>
  T* Record(Args&&... args) {
    if (event_blocks.empty() || event_blocks.front().size() == kNumBlock) {
      event_blocks.emplace_front();
      event_blocks.front().reserve(kNumBlock);
    }
    event_blocks.front().emplace_back(std::forward<Args>(args)...);
    return &event_blocks.front().back();
  }

  std::vector<T> Reduce() {
    std::vector<T> result;
    for (auto& block : event_blocks) {
      result.insert(result.begin(),
                    std::make_move_iterator(block.begin()),
                    std::make_move_iterator(block.end()));
    }
    event_blocks.clear();
    return result;
  }

  void Clear() { event_blocks.clear(); }

  std::forward_list<std::vector<T>> event_blocks;
};

Event* PushEvent(const std::string& name,
                 const EventRole role,
                 const std::string attr = "none");
void PopEvent(const std::string& name,
              const EventRole role,
              const std::string attr = "none");

void EnableOpInfoRecorder();
void DisableOpInfoRecorder();

}  // namespace phi
