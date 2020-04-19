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

#include <string>
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_helper.h"

namespace paddle {
namespace framework {
namespace ir {

class SSAGraghBuilderWithChecker : public ir::Pass {
 protected:
  void ApplyImpl(ir::Graph *graph) const override {
    PADDLE_ENFORCE(IsValidGraph(graph));
  }

  bool IsValidGraph(const ir::Graph *graph) const {
    std::unordered_map<details::OpHandleBase *, size_t> pending_ops;
    std::unordered_set<details::VarHandleBase *> pending_vars;
    std::unordered_set<details::VarHandleBase *> ready_vars;
    std::unordered_set<details::OpHandleBase *> ready_ops;

    auto insert_pending_var = [&](details::VarHandleBase *var) {
      pending_vars.insert(var);
      if (var->GeneratedOp() == nullptr) {
        ready_vars.emplace(var);
      }
    };

    for (auto &var_map : graph->Get<details::GraphVars>(details::kGraphVars)) {
      for (auto &name_pair : var_map) {
        for (auto &version_pair : name_pair.second) {
          insert_pending_var(version_pair);
        }
      }
    }

    for (auto &var :
         graph->Get<details::GraphDepVars>(details::kGraphDepVars)) {
      insert_pending_var(var);
    }

    for (auto *op : ir::FilterByNodeWrapper<details::OpHandleBase>(*graph)) {
      if (op->Inputs().empty()) {
        ready_ops.insert(op);
      } else {
        pending_ops.insert({op, op->NoDupInputSize()});
      }
    }

    auto run_all_ops = [&](std::unordered_set<details::OpHandleBase *> &set) {
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
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(multi_devices_check_pass,
              paddle::framework::ir::SSAGraghBuilderWithChecker)
    .RequireGraphAttr(paddle::framework::details::kGraphVars)
    .RequireGraphAttr(paddle::framework::details::kGraphDepVars);
