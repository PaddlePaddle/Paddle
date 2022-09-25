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

#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace framework {
namespace interpreter {

struct ExecutionConfig {
  bool used_for_jit{false};
  bool create_local_scope{true};

  size_t host_num_threads;
  size_t deivce_num_threads;
  size_t prepare_num_threads;

  std::set<std::string> skip_gc_vars;

  ExecutionConfig(const phi::Place& place, size_t op_num);
  void Log(int log_level);
};

}  // namespace interpreter
}  // namespace framework
}  // namespace paddle
