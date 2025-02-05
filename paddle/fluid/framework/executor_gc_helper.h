// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/garbage_collector.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/scope.h"

namespace paddle {
namespace framework {

// Result map: op -> variable names that can be deleted after op runs
class GarbageCollector;
class OperatorBase;
class Scope;

struct OpInOutInfo {
 public:
  void Build(const OperatorBase *op);

  bool IsBuilt() const { return is_built_; }

  bool IsInArgBufferNeeded(const std::string &in_arg_name) const;

 private:
  // A set to record unused buffer input vars of op
  std::unordered_set<std::string> no_need_buffer_ins_;
  // A set to record other args of op (including in, out)
  std::unordered_set<std::string> other_args_set_;
  bool is_built_{false};
};

std::unordered_map<const OperatorBase *, std::vector<std::string>>
GetUnusedVars(const BlockDesc &block,
              const std::vector<std::unique_ptr<OperatorBase>> &ops,
              const std::vector<std::string> &skip_vars,
              const std::multiset<std::string> *unpersist_vars = nullptr,
              bool is_shard_for_thread_mode = false);

// Collect unused tensors
void DeleteUnusedTensors(const Scope &scope,
                         const std::vector<std::string> &delete_vars,
                         GarbageCollector *gc);

// Collect unused tensors after op runs
void DeleteUnusedTensors(
    const Scope &scope,
    const OperatorBase *op,
    const std::unordered_map<const OperatorBase *, std::vector<std::string>>
        &delete_vars_map,
    GarbageCollector *gc);

// Get the clean vars of GC after each op runs. This function is used for
// analysis statically.
// result is in the format: result[block_idx][op_idx][delete_var_idx]
std::vector<std::vector<std::vector<std::string>>> GetEagerDeletionCleanVars(
    const ProgramDesc &program, const std::vector<std::string> &skip_vars = {});

std::vector<std::vector<std::vector<std::string>>>
GetEagerDeletionCleanVarsForPartial(
    const ProgramDesc &program,
    const std::vector<std::string> &skip_vars = {},
    const bool &for_partial_block = false);

}  // namespace framework
}  // namespace paddle
