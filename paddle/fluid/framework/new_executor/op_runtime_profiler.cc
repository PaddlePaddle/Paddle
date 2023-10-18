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

#include "paddle/fluid/framework/new_executor/op_runtime_profiler.h"

namespace paddle {
namespace framework {
namespace profiling {

OpRuntimeProfilingRecorder::OpRuntimeProfilingRecorder() {}
OpRuntimeProfilingRecorder::~OpRuntimeProfilingRecorder() {}

void OpRuntimeProfilingRecorder::RecordOpRuntime(const std::string& key,
                                                 double us) {
  auto it = all_ops_runtime_us_.find(key);
  if (it != all_ops_runtime_us_.end()) {
    double old_us = it->second;
    VLOG(4) << "Warning: op runtime will be overwritten! Op: \"" << key << "\""
            << ", old value: " << old_us << ", new value: " << us;
    it->second = us;
  } else {
    VLOG(4) << "Record op runtime. Op: \"" << key << "\", runtime: " << us;
    all_ops_runtime_us_.insert(std::pair<std::string, double>(key, us));
  }
}

double OpRuntimeProfilingRecorder::GetOpRuntime(const std::string& key) const {
  auto it = all_ops_runtime_us_.find(key);
  if (it == all_ops_runtime_us_.end()) {
    VLOG(1) << "Op key \"" << key << "\" not found in profiling recorder!";
    return -1.0;
  } else {
    return it->second;
  }
}

bool OpRuntimeProfilingRecorder::FindOpRuntimeRecord(
    const std::string& key) const {
  auto it = all_ops_runtime_us_.find(key);
  return it != all_ops_runtime_us_.end();
}

}  // namespace profiling
}  // namespace framework
}  // namespace paddle
