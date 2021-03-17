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

void NPUProfilerInit(std::string output_path, std::string output_mode,
                     std::string config_file) {
  PADDLE_ENFORCE_NPU_SUCCESS(
      aclprofInit(output_path.c_str(), output_path.size()));
}

void NPUProfilerStart(const aclprofConfig *config)) {
  PADDLE_ENFORCE_NPU_SUCCESS(aclprofStart(config));
}

void NPUProfilerStop(const aclprofConfig *config)) {
  PADDLE_ENFORCE_NPU_SUCCESS(aclprofStop(config));
}

void NPUProfilerFinalize() { PADDLE_ENFORCE_NPU_SUCCESS(aclprofFinalize()); }

void NPUProfilerCreateConfig(std::vector<int32_t> devices,
                             aclprofAicoreMetrics metrics,
                             dataTypeConfig config,
                             p aclprofAicoreEvents *events = nullptr) {
  PADDLE_ENFORCE_NPU_SUCCESS(aclprofCreateConfig(devices.data(), devices.size(),
                                                 metrics, events, config));
}

void NPUProfilerDestroyConfig(const aclprofConfig *config) {
  PADDLE_ENFORCE_NPU_SUCCESS(aclprofDestroyConfig(config));
}

}  // namespace platform
}  // namespace paddle
