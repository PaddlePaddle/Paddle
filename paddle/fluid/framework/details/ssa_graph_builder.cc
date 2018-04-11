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
void SSAGraphBuilder::PolishGraphToSupportDataHazards(Context *graph) {
  for (auto &var_map : graph->vars_) {
    for (auto &name_pair : var_map) {
      if (name_pair.second.size() <= 1) {
        continue;
      }
      auto it_new = name_pair.second.rbegin();
      auto it_old = name_pair.second.rbegin();
      ++it_old;
      for (; it_old != name_pair.second.rend(); it_new = it_old, ++it_old) {
        auto *write_op = (*it_new)->generated_op_;
        auto &read_ops = (*it_old)->pending_ops_;

        for (auto *read_op : read_ops) {
          // Manually add a dependency var from read_op to write_op;
          if (read_op == write_op) {
            // Read Write is the same op.
            continue;
          }

          auto *dep_var = new DummyVarHandle();
          read_op->AddOutput(dep_var);
          write_op->AddInput(dep_var);
          graph->dep_vars_.emplace_back(dep_var);
        }
      }
    }
  }
}

VarHandle *SSAGraphBuilder::CreateOrGetLatestVarHandle(
    Context *graph, const std::string &each_var_name,
    const platform::Place &place, size_t place_offset) {
  auto &var_holders = graph->vars_[place_offset];
  auto &var_holder = var_holders[each_var_name];
  VarHandle *var = nullptr;
  if (var_holder.empty()) {
    var_holder.emplace_back(new VarHandle);
    auto &init_var = var_holder[0];
    init_var->place_ = place;
    init_var->name_ = each_var_name;
    init_var->generated_op_ = nullptr;
    init_var->version_ = 0;
    var = init_var.get();
  } else {
    var = var_holder.rbegin()->get();
  }
  return var;
}

void SSAGraphBuilder::CreateOpOutput(Context *graph, OpHandleBase *op_handle,
                                     const std::string &each_var_name,
                                     const platform::Place &place,
                                     size_t place_offset) {
  auto &vars = graph->vars_[place_offset][each_var_name];
  size_t version = vars.size();
  vars.emplace_back(new VarHandle());
  auto &var = vars.back();
  var->version_ = version;
  var->name_ = each_var_name;
  var->place_ = place;
  op_handle->AddOutput(var.get());
}

template <typename Callback>
static void IterAllVar(const SSAGraphBuilder::Context &graph,
                       Callback callback) {
  for (auto &each : graph.vars_) {
    for (auto &pair1 : each) {
      for (auto &pair2 : pair1.second) {
        callback(*pair2);
      }
    }
  }

  for (auto &var : graph.dep_vars_) {
    callback(*var);
  }
}

void SSAGraphBuilder::PrintGraphviz(const Context &graph, std::ostream &sout) {
  size_t var_id = 0;
  std::unordered_map<const VarHandleBase *, size_t> vars;

  sout << "digraph G {\n";

  IterAllVar(graph, [&](const VarHandleBase &var) {
    auto *var_ptr = &var;
    auto *var_handle_ptr = dynamic_cast<const VarHandle *>(var_ptr);
    auto *dummy_ptr = dynamic_cast<const DummyVarHandle *>(var_ptr);

    size_t cur_var_id = var_id++;
    vars[var_ptr] = cur_var_id;

    if (var_handle_ptr) {
      sout << "var_" << cur_var_id << " [label=\"" << var_handle_ptr->name_
           << "\\n"
           << var_handle_ptr->place_ << "\\n"
           << var_handle_ptr->version_ << "\"]" << std::endl;
    } else if (dummy_ptr) {
      sout << "var_" << cur_var_id << " [label=\"dummy\"]" << std::endl;
    }
  });

  size_t op_id = 0;
  for (auto &op : graph.ops_) {
    std::string op_name = "op_" + std::to_string(op_id++);
    sout << op_name << " [label=\"" << op->Name() << "\", shape=rect]"
         << std::endl;
    for (auto in : op->inputs_) {
      std::string var_name = "var_" + std::to_string(vars[in]);
      sout << var_name << " -> " << op_name << std::endl;
    }

    for (auto out : op->outputs_) {
      std::string var_name = "var_" + std::to_string(vars[out]);
      sout << op_name << " -> " << var_name << std::endl;
    }
  }

  sout << "}\n";
}

void SSAGraphBuilder::AddOutputToLeafOps(Context *graph) {
  for (auto &op : graph->ops_) {
    if (!op->outputs_.empty()) {
      continue;
    }
    auto *dummy_leaf = new DummyVarHandle();
    graph->dep_vars_.emplace_back(dummy_leaf);
    op->AddOutput(dummy_leaf);
  }
}

std::unique_ptr<SSAGraph> SSAGraphBuilder::ContextToSSAGraph(
    std::unique_ptr<Context> &&graph) {
  SSAGraph *result = new SSAGraph();

  for (auto &op : graph->ops_) {
    result->ops_.emplace_back(std::move(op));
  }

  for (auto &vars : graph->vars_) {
    for (auto &versioned_var : vars) {
      for (auto &v : versioned_var.second) {
        result->vars_.emplace_back(v.release());
      }
    }
  }

  for (auto &var : graph->dep_vars_) {
    result->vars_.emplace_back(std::move(var));
  }

  return std::unique_ptr<SSAGraph>(result);
}
}  // namespace details
}  // namespace framework
}  // namespace paddle
