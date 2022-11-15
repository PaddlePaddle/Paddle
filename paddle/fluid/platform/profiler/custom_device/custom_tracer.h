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
#include <memory>
#include <vector>

#include "paddle/fluid/platform/macros.h"
#include "paddle/fluid/platform/profiler/tracer_base.h"

namespace paddle {
namespace platform {

class CustomTracer : public TracerBase {
 public:
  static CustomTracer& GetInstance(const std::string& device_type) {
    static std::unordered_map<std::string, std::shared_ptr<CustomTracer>>
        instance;
    if (instance.find(device_type) == instance.cend()) {
      instance.insert(
          {device_type, std::make_shared<CustomTracer>(device_type)});
    }
    return *instance[device_type];
  }

  void PrepareTracing() override;

  void StartTracing() override;

  void StopTracing() override;

  void CollectTraceData(TraceEventCollector* collector) override;

  ~CustomTracer() override;

  explicit CustomTracer(const std::string& dev_type);

 private:
  DISABLE_COPY_AND_ASSIGN(CustomTracer);

  TraceEventCollector collector_;

  uint64_t tracing_start_ns_ = UINT64_MAX;

  std::string dev_type_;

  void* context_;
};

}  // namespace platform
}  // namespace paddle
