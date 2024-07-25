// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <cstdint>
#include <memory>
#include <vector>

#include "paddle/common/macros.h"
#include "paddle/fluid/platform/profiler/tracer_base.h"
#include "paddle/phi/backends/dynload/xpti.h"

namespace paddle {
namespace platform {

class XPUTracer : public TracerBase {
 public:
  static XPUTracer& GetInstance() {
    static XPUTracer instance;
    return instance;
  }

  void PrepareTracing() override;

  void StartTracing() override;

  void StopTracing() override;

  void CollectTraceData(TraceEventCollector* collector) override;

  XPUTracer() {}

 private:
  DISABLE_COPY_AND_ASSIGN(XPUTracer);

  uint64_t tracing_start_ns_ = UINT64_MAX;
};

}  // namespace platform
}  // namespace paddle
