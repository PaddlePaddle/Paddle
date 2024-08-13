// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/new_executor/interpreter/plan.h"
#include "paddle/fluid/framework/new_executor/interpretercore.h"
#include "paddle/fluid/framework/new_executor/new_executor_defs.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/phi/common/place.h"
#include "paddle/pir/include/core/program.h"

namespace paddle {
namespace framework {

class InterpreterCore;

class StandaloneExecutor {
 public:
  StandaloneExecutor(const phi::Place& place,
                     const interpreter::Plan& plan_,
                     Scope* scope);

  ~StandaloneExecutor() {}

  paddle::framework::FetchList Run(
      const std::vector<std::string>& feed_names,
      const bool enable_job_schedule_profiler = false);

  std::shared_ptr<framework::ProgramDesc> RunProfile(
      const std::vector<std::string>& feed_names);

 private:
  bool is_interpretercore_build_result_shared_{false};
  const phi::Place place_;
  interpreter::Plan plan_;
  std::vector<std::shared_ptr<InterpreterCore>> interpretercores_;

  Scope* scope_;
  std::vector<Scope*> micro_batch_scopes_;

  std::vector<std::string> fetch_var_names_;
  FetchUnmergedList fetch_list_;

  std::vector<std::unordered_map<std::string, std::shared_ptr<EventInter>>>
      vec_force_events_to_wait_;
};

}  // namespace framework
}  // namespace paddle
