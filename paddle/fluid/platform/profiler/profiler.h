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

#include <atomic>
#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include "paddle/fluid/platform/macros.h"
#include "paddle/fluid/platform/profiler/event_node.h"
#include "paddle/fluid/platform/profiler/tracer_base.h"

namespace paddle {
namespace platform {

struct ProfilerOptions {
  uint32_t trace_level = 0;
};

class Profiler {
 public:
  static std::unique_ptr<Profiler> Create(const ProfilerOptions& options);

  void Prepare();

  void Start();

  std::unique_ptr<NodeTrees> Stop();

  ~Profiler();

 private:
  class TracerHolder {
   public:
    TracerHolder(TracerBase* tracer, bool owned)
        : tracer(tracer), owned(owned) {}
    ~TracerHolder() {
      if (owned) {
        delete tracer;
      }
    }

    TracerBase& Get() { return *tracer; }

   private:
    TracerBase* tracer;
    bool owned;
  };

  explicit Profiler(const ProfilerOptions& options);

  DISABLE_COPY_AND_ASSIGN(Profiler);

  static std::atomic<bool> alive_;
  ProfilerOptions options_;
  uint64_t start_ns_ = UINT64_MAX;
  std::list<TracerHolder> tracers_;
};

}  // namespace platform
}  // namespace paddle
