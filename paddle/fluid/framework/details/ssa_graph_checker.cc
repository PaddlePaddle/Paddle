// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/details/ssa_graph_checker.h"
#include <string>
#include "paddle/fluid/framework/ir/graph.h"

namespace paddle {
namespace framework {
namespace details {

bool SSAGraghBuilderWithChecker::IsValidGraph(const Graph *graph) const {
  std::unordered_map<OpHandleBase *, size_t> pending_ops;
  std::unordered_set<VarHandleBase *> pending_vars;
  std::unordered_set<VarHandleBase *> ready_vars;
  std::unordered_set<OpHandleBase *> ready_ops;

  auto insert_pending_var = [&](VarHandleBase *var) {
    pending_vars.insert(var);
    if (var->GeneratedOp() == nullptr) {
      ready_vars.emplace(var);
    }
  };

  for (auto &var_map : graph->Get<GraphVars>("vars")) {
    for (auto &name_pair : var_map) {
      for (auto &version_pair : name_pair.second) {
        insert_pending_var(version_pair.get());
      }
    }
  }

  for (auto &var : graph->Get<GraphDepVars>("dep_vars")) {
    insert_pending_var(var.get());
  }

  for (auto &op : graph->Get<GraphOps>("ops")) {
    if (op->Inputs().empty()) {
      ready_ops.insert(op.get());
    } else {
      pending_ops.insert({op.get(), op.get()->NoDupInputSize()});
    }
  }

  auto run_all_ops = [&](std::unordered_set<OpHandleBase *> &set) {
    for (auto *op : set) {
      for (auto out : op->Outputs()) {
        ready_vars.emplace(out);
      }
    }
    set.clear();
  };

  while (!pending_vars.empty()) {
    run_all_ops(ready_ops);

    if (ready_vars.empty()) {
      return false;
    }

    for (auto ready_var : ready_vars) {
      pending_vars.erase(ready_var);
      for (auto *op : ready_var->PendingOps()) {
        auto &deps = --pending_ops[op];
        if (deps == 0) {
          ready_ops.insert(op);
        }
      }
    }
    ready_vars.clear();
  }
  return true;
}
}  // namespace details
}  // namespace framework
}  // namespace paddle
