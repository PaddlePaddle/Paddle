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

#include <string>
#include <vector>

#include "acl/acl_prof.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace platform {

// ACL_AICORE_ARITHMETIC_UTILIZATION = 0, record arithmetic stats
// ACL_AICORE_PIPE_UTILIZATION = 1, record pipe
// ACL_AICORE_MEMORY_BANDWIDTH = 2, record memory
// ACL_AICORE_L0B_AND_WIDTH = 3, recore internal io
// ACL_AICORE_RESOURCE_CONFLICT_RATI = 4, record conflict ratio
constexpr aclprofAicoreMetrics default_metrics =
    ACL_AICORE_ARITHMETIC_UTILIZATION;

// ACL_PROF_ACL_API, record ACL API stats
// ACL_PROF_TASK_TIME, record AI core stats
// ACL_PROF_AICORE_METRICS, must include
// ACL_PROF_AICPU_TRACE, recore AICPU, not supported yet
constexpr dataTypeConfig default_type =
    ACL_PROF_ACL_API | ACL_PROF_AICORE_METRICS | ACL_PROF_TASK_TIME;

void NPUProfilerInit(std::string output_path) {
  PADDLE_ENFORCE_NPU_SUCCESS(
      aclprofInit(output_path.c_str(), output_path.size()));
}

void NPUProfilerStart(const aclprofConfig *config)) {
  if (config == nullptr) {
    // NOTE(zhiqiu): support single device by default.
    int device_id = GetCurrentNPUDeviceId();
    std::vector<uint32_t> devices = {static_cast<uint32_t>(device_id)};
    config = NPUProfilerCreateConfig(devices, metrics, c);
  }
  PADDLE_ENFORCE_NPU_SUCCESS(aclprofStart(config));
}

void NPUProfilerStop(const aclprofConfig *config)) {
  PADDLE_ENFORCE_NPU_SUCCESS(aclprofStop(config));
  NPUProfilerDestroyConfig(config);
}

void NPUProfilerFinalize() { PADDLE_ENFORCE_NPU_SUCCESS(aclprofFinalize()); }

aclprofConfig *NPUProfilerCreateConfig(
    std::vector<int32_t> devices,
    aclprofAicoreMetrics metrics = default_metrics,
    dataTypeConfig c = default_type, p aclprofAicoreEvents *events = nullptr) {
  aclprofConfig* config = aclprofCreateConfig(devices.data(), devices.size(),
                                            metrics, events, c));
  PADDLE_ENFORCE_NOT_NULL(config, paddle::platform::errors::External(
                                      "Failed to create prof config for NPU"));
  return config;
}

void NPUProfilerDestroyConfig(const aclprofConfig *config) {
  PADDLE_ENFORCE_NPU_SUCCESS(aclprofDestroyConfig(config));
}

}  // namespace platform
}  // namespace paddle
