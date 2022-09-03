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
#include "paddle/fluid/framework/new_executor/interpretercore.h"
#include "paddle/fluid/framework/new_executor/new_executor_defs.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace framework {

class InterpreterCore;

class StandaloneExecutor {
 public:
  StandaloneExecutor(const platform::Place& place, const ProgramDesc& prog);

  ~StandaloneExecutor() {}

  // NOTE(zhiqiu): feed_names are only used for caching interpretercore.
  // fetch_names are used for caching interpretercore and inserting fetch ops,
  // the latter can be moved to python side.
  paddle::framework::FetchList Run(Scope* scope,
                                   const std::vector<std::string>& feed_names,
                                   const std::vector<std::string>& fetch_names);

  framework::interpreter::CostInfo DryRun(
      Scope* scope,
      const std::vector<std::string>& feed_names,
      const std::vector<framework::LoDTensor>& feed_tensors);

 private:
  std::shared_ptr<InterpreterCore> GetInterpreterCore(
      Scope* scope,
      const ProgramDesc& prog,
      const std::vector<std::string>& feed_names,
      const std::vector<std::string>& fetch_names,
      bool add_fetch_op);

  platform::Place place_;
  const ProgramDesc& prog_;

  std::unordered_map<std::string, std::shared_ptr<InterpreterCore>>
      interpretercores_;
};

}  // namespace framework
}  // namespace paddle
