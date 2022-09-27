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
#include <bitset>
#include <cstdint>
#include <functional>
#include <list>
#include <memory>

#include "paddle/fluid/platform/macros.h"
#include "paddle/fluid/platform/profiler/cpu_utilization.h"
#include "paddle/fluid/platform/profiler/event_node.h"
#include "paddle/fluid/platform/profiler/event_python.h"
#include "paddle/fluid/platform/profiler/tracer_base.h"

DECLARE_int64(host_trace_level);

namespace paddle {
namespace platform {

static constexpr uint32_t kProfileCPUOptionBit = 0;
static constexpr uint32_t kProfileGPUOptionBit = 1;
static constexpr uint32_t kProfileMLUOptionBit = 2;
static constexpr uint32_t kProfileCustomDeviceOptionBit = 3;

struct ProfilerOptions {
  uint32_t trace_switch = 0;  // bit 0: cpu, bit 1: gpu, bit 2: mlu
  uint32_t trace_level = FLAGS_host_trace_level;
};

class Profiler {
 public:
  static uint32_t
      span_indx;  // index of profiler range, when user profiles multiple ranges
                  // such as [2,4], [6,8], the first range is index 0.
  static const char* version;  // profiler version.
  static std::unique_ptr<Profiler> Create(
      const ProfilerOptions& options,
      const std::vector<std::string>& custom_device_types = {});

  static bool IsCuptiSupported();

  static bool IsCnpapiSupported();

  void Prepare();

  void Start();

  std::unique_ptr<ProfilerResult> Stop();

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

  explicit Profiler(const ProfilerOptions& options,
                    const std::vector<std::string>& custom_device_types = {});

  DISABLE_COPY_AND_ASSIGN(Profiler);

  static std::atomic<bool> alive_;
  ProfilerOptions options_;
  uint64_t start_ns_ = UINT64_MAX;
  std::list<TracerHolder> tracers_;
  CpuUtilization cpu_utilization_;
};

}  // namespace platform
}  // namespace paddle
