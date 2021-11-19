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

#include "paddle/fluid/framework/new_executor/interpretercore.h"

namespace paddle {
namespace framework {

class ExecutorBase {
 public:
  virtual ~ExecutorBase() {}
  virtual paddle::framework::FetchList Run(
      const std::vector<std::string>& feed_names,
      const std::vector<framework::LoDTensor>& feed_tensors,
      const std::vector<std::string>& fetch_names) = 0;
};

class StandaloneExecutor : public ExecutorBase {
 public:
  StandaloneExecutor(const platform::Place& place,
                     const ProgramDesc& startup_prog,
                     const ProgramDesc& main_prog, Scope* scope);

  ~StandaloneExecutor() {}

  paddle::framework::FetchList Run(
      const std::vector<std::string>& feed_names,
      const std::vector<framework::LoDTensor>& feed_tensors,
      const std::vector<std::string>& fetch_names);

  // NOTE(zhiqiu): feed_names are only used for caching interpretercore.
  // fetch_names are used for caching interpretercore and inserting fetch ops,
  // the latter can be moved to python side.
  paddle::framework::FetchList Run(const std::vector<std::string>& feed_names,
                                   const std::vector<std::string>& fetch_names);

  framework::interpreter::CostInfo DryRun(
      const std::vector<std::string>& feed_names,
      const std::vector<framework::LoDTensor>& feed_tensors);

 private:
  void BuildVariableScope(const framework::ProgramDesc& pdesc,
                          VariableScope* var_scope);

  std::shared_ptr<InterpreterCore> GetInterpreterCore(
      const std::vector<std::string>& feed_names,
      const std::vector<std::string>& fetch_names);

  const platform::Place& place_;
  const ProgramDesc& startup_prog_;
  const ProgramDesc& main_prog_;
  VariableScope global_scope_;

  std::unordered_map<std::string, std::shared_ptr<ProgramDesc>> programs_;
  std::unordered_map<std::string, std::shared_ptr<InterpreterCore>>
      interpretercores_;
};

}  // namespace framework
}  // namespace paddle
