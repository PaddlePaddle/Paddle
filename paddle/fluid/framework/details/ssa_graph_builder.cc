//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/details/ssa_graph_builder.h"

namespace paddle {
namespace framework {
namespace details {
void SSAGraphBuilder::PolishGraphToSupportDataHazards(SSAGraph *graph) {
  for (auto &var_map : graph->vars_) {
    for (auto &name_pair : var_map) {
      if (name_pair.second.size() <= 1) {
        return;
      }
      auto it_new = name_pair.second.rbegin();
      auto it_old = name_pair.second.rbegin();
      ++it_old;
      for (; it_old != name_pair.second.rend(); it_new = it_old, ++it_old) {
        auto *write_op = it_new->second.generated_op_;
        auto &read_ops = it_old->second.pending_ops_;
        auto *ex_write_op = it_old->second.generated_op_;

        if (ex_write_op == nullptr) {  // Nobody write this var.
          continue;
        }

        for (auto *read_op : read_ops) {
          // Manually add a dependency var from read_op to write_op;
          if (read_op == write_op) {
            // Read Write is the same op.
            continue;
          }

          auto *dep_var = new DummyVarHandle();
          read_op->AddOutput(dep_var);
          write_op->AddInput(dep_var);
          graph->dep_vars_.emplace(dep_var);
        }
      }
    }
  }
}

VarHandle *SSAGraphBuilder::CreateOrGetLatestVarHandle(
    SSAGraph *graph, const std::string &each_var_name,
    const platform::Place &place, size_t place_offset) {
  auto &var_holders = graph->vars_[place_offset];
  auto &var_holder = var_holders[each_var_name];
  VarHandle *var = nullptr;
  if (var_holder.empty()) {
    auto &init_var = var_holder[0];
    init_var.place_ = place;
    init_var.name_ = each_var_name;
    init_var.generated_op_ = nullptr;
    init_var.version_ = 0;
    var = &init_var;
  } else {
    var = &var_holder.rbegin()->second;
  }
  return var;
}

void SSAGraphBuilder::CreateOpOutput(SSAGraph *graph, OpHandleBase *op_handle,
                                     const std::string &each_var_name,
                                     const platform::Place &place,
                                     size_t place_offset) {
  auto &vars = graph->vars_[place_offset][each_var_name];
  size_t version = vars.size();
  auto &var = vars[version];
  var.version_ = version;
  var.name_ = each_var_name;
  var.place_ = place;
  op_handle->AddOutput(&var);
}
}  // namespace details
}  // namespace framework
}  // namespace paddle
