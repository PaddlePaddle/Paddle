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

#include "paddle/fluid/platform/profiler/tracer_base.h"

namespace paddle {
namespace platform {

class HostTraceLevel {
 public:
  static constexpr int64_t kDisabled = -1;

  static HostTraceLevel& GetInstance() {
    static HostTraceLevel instance;
    return instance;
  }

  bool NeedTrace(uint32_t level) {
    return trace_level_ >= static_cast<int64_t>(level);
  }

  void SetLevel(int64_t trace_level) { trace_level_ = trace_level; }

 private:
  // Verbose trace level, works like VLOG(level)
  int trace_level_ = kDisabled;
};

struct HostTracerOptions {
  uint32_t trace_level = 0;
};

class HostTracer : public TracerBase {
 public:
  explicit HostTracer(const HostTracerOptions& options) {
    trace_level_ = options.trace_level;
  }

  void StartTracing() override;

  void StopTracing() override;

  void CollectTraceData(TraceEventCollector* collector) override;

 private:
  uint32_t trace_level_;
};

}  // namespace platform
}  // namespace paddle
