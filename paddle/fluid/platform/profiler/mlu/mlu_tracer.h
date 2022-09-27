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

#include <cstdint>
#include <vector>
#ifdef PADDLE_WITH_MLU
#include "paddle/fluid/platform/device/mlu/mlu_info.h"
#endif
#include "paddle/fluid/platform/macros.h"
#include "paddle/fluid/platform/profiler/tracer_base.h"

namespace paddle {
namespace platform {

class MluTracer : public TracerBase {
 public:
  static MluTracer& GetInstance() {
    static MluTracer instance;
    return instance;
  }

  void PrepareTracing() override;

  void StartTracing() override;

  void StopTracing() override;

  void CollectTraceData(TraceEventCollector* collector) override;

  void ProcessCnpapiActivity(uint64_t* buffer, size_t valid_size);

 private:
  MluTracer();

  DISABLE_COPY_AND_ASSIGN(MluTracer);

  void EnableCnpapiActivity();

  void DisableCnpapiActivity();

  uint64_t tracing_start_ns_ = UINT64_MAX;

  TraceEventCollector collector_;
};

}  // namespace platform
}  // namespace paddle
