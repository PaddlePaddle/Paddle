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

#include "paddle/fluid/framework/details/ssa_graph_printer.h"
#include <string>
#include "paddle/fluid/framework/details/ssa_graph.h"

namespace paddle {
namespace framework {
namespace details {

template <typename Callback>
static inline void IterAllVar(const SSAGraph &graph, Callback callback) {
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

void GraphvizSSAGraphPrinter::Print(const SSAGraph &graph,
                                    std::ostream &sout) const {
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
    for (auto in : op->Inputs()) {
      std::string var_name = "var_" + std::to_string(vars[in]);
      sout << var_name << " -> " << op_name << std::endl;
    }

    for (auto out : op->Outputs()) {
      std::string var_name = "var_" + std::to_string(vars[out]);
      sout << op_name << " -> " << var_name << std::endl;
    }
  }

  sout << "}\n";
}
}  // namespace details
}  // namespace framework
}  // namespace paddle
