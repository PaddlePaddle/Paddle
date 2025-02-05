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

#include <set>
#include <string>

#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/common/place.h"

namespace paddle {
namespace framework {
namespace interpreter {

struct ExecutionConfig {
  bool create_local_scope{true};
  bool used_for_cinn{false};
  bool used_for_control_flow_op{false};
  bool used_for_jit{false};
  bool used_for_sot{false};
  bool used_for_inference{false};

  size_t device_num_threads{0};
  size_t host_num_threads{0};

  std::set<std::pair<int, std::string>>
      force_sync_ops;  // set{pair<op_id, name>}, -1 matches any op_id, ""
                       // matches any name

  std::set<std::string> force_root_scope_vars;
  std::set<std::string> jit_input_vars;
  std::set<std::string> skip_gc_vars;

  void AnalyzeThreadPoolConfig(const phi::Place& place, size_t op_num);
  void Log(int log_level);
};

std::set<std::pair<int, std::string>> GetForceSyncOps(
    int micro_batch_id, const std::string& job_name);

}  // namespace interpreter
}  // namespace framework
}  // namespace paddle
