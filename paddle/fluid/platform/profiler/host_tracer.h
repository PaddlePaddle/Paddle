/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#include <cstring>
#include <mutex>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/platform/macros.h"
#include "paddle/fluid/platform/profiler/tracer_base.h"

namespace paddle {
namespace platform {

class HostTracer : public TracerBase {
 public:
  void PrepareTracing() {}

  void StartTracing() override;

  void StopTracing() override;

  void CollectTraceData(TraceEventCollector *collector) override;

  void SetTraceLevel(uint32_t trace_level) {
    trace_level_ = trace_level;
    state_ = TracerState::READY;
  }

 private:
  uint32_t trace_level_;
};

}  // namespace platform
}  // namespace paddle
