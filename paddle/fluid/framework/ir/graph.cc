/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/var_desc.h"

namespace paddle {
namespace framework {
namespace ir {

Graph::Graph(const ProgramDesc &program) : program_(program) {
  auto var_nodes = InitFromProgram(program_);
  ResolveHazard(var_nodes);
}

std::map<std::string, std::vector<ir::Node *>> Graph::InitFromProgram(
    const ProgramDesc &program) {
  VLOG(3) << "block in program:" << program_.Size();
  std::unordered_map<std::string, VarDesc *> all_vars;
  // var nodes for each var name, will have multiple versions in SSA
  std::map<std::string, std::vector<ir::Node *>> var_nodes;
  for (auto *var : program.Block(0).AllVars()) {
    all_vars.emplace(var->Name(), var);
  }

  for (auto *op : program.Block(0).AllOps()) {
    ir::Node *node = CreateOpNode(op);
    // For input args, reuse the same var name if it was created before.
    // Otherwise, create a new one.
    for (auto &each_var_name : op->InputArgumentNames()) {
      ir::Node *var = nullptr;
      if (var_nodes.find(each_var_name) != var_nodes.end()) {
        var = var_nodes.at(each_var_name).back();
      } else if (all_vars.count(each_var_name) != 0) {
        var = CreateVarNode(all_vars.at(each_var_name));
        var_nodes[each_var_name].push_back(var);
      } else {
        // Operation input var can be optional (dispensable). Which means
        // the operation doesn't really need the var at runtime. In this
        // case, the no-existed var is ready at the beginning.
        var = CreateEmptyNode(each_var_name, ir::Node::Type::kVariable);
        var_nodes[each_var_name].push_back(var);
      }
      node->inputs.push_back(var);
      var->outputs.push_back(node);
    }
    // For output args, always create a new var.
    std::unordered_set<std::string> out_arg_set;
    for (auto &each_var_name : op->OutputArgumentNames()) {
      if (each_var_name != kEmptyVarName) {
        PADDLE_ENFORCE(out_arg_set.count(each_var_name) == 0,
                       "Program is wrong. %s occurs in output of %s several "
                       "times.",
                       each_var_name, op->Type());
        out_arg_set.insert(each_var_name);
      }

      ir::Node *var = nullptr;
      if (all_vars.count(each_var_name) != 0) {
        var = CreateVarNode(all_vars.at(each_var_name));
      } else {
        // Operation output vars can be @EMPTY@. For example, while_grad
        // can have multi @EMPTY@ outputs with no VarDesc.
        // TODO(panyx0718): Add a test.
        var = CreateEmptyNode(each_var_name, ir::Node::Type::kVariable);
      }
      var_nodes[each_var_name].push_back(var);
      node->outputs.push_back(var);
      var->inputs.push_back(node);
    }
  }
  Set<const std::vector<OpDesc *>>(
      details::kStaleProgramOpDescs,
      new std::vector<OpDesc *>(program.Block(0).AllOps()));
  return var_nodes;
}

void Graph::ResolveHazard(
    const std::map<std::string, std::vector<ir::Node *>> &var_nodes) {
  /**
   * We should handle write after read(WAR) and write after write(WAW) here.
   * Because some of the operators of the program can be executed parallelly.
   * So, to make the program running in the right order, we should add the
   * dependence of WAR and WAW.
   *
   *
   * https://en.wikipedia.org/wiki/Hazard_(computer_architecture)#Write_after_read_(WAR)
   */

  for (auto &var : var_nodes) {
    auto &versions = var.second;
    if (versions.size() <= 1) continue;

    auto it_new = versions.rbegin();
    auto it_old = versions.rbegin();
    ++it_old;
    for (; it_old != versions.rend(); it_new = it_old, ++it_old) {
      VLOG(3) << "deal with var: " << (*it_new)->Name();
      ir::Node *write_op =
          (*it_new)->inputs.empty() ? nullptr : (*it_new)->inputs[0];
      const auto &read_ops = (*it_old)->outputs;

      PADDLE_ENFORCE(
          write_op,
          string::Sprintf("The write_op of var %s should not be empty.",
                          (*it_new)->Name()));

      // Add write after write dependence
      ir::Node *upstream_op =
          (*it_old)->inputs.empty() ? nullptr : (*it_old)->inputs[0];
      // TODO(zcd): Add a test.
      if (upstream_op && upstream_op != write_op) {
        ir::Node *dep_var = CreateControlDepVar();
        write_op->inputs.push_back(dep_var);
        upstream_op->outputs.push_back(dep_var);
        dep_var->outputs.push_back(write_op);
        dep_var->inputs.push_back(upstream_op);
      }

      for (auto *read_op : read_ops) {
        // Manually add a dependency var from read_op to write_op;
        if (read_op == write_op) {
          // Read Write is the same op.
          continue;
        }
        // 2 ops might have been connected via other vars.
        bool has_dep = false;
        for (ir::Node *r_out : read_op->outputs) {
          for (ir::Node *w_in : write_op->inputs) {
            if (r_out == w_in) {
              has_dep = true;
              break;
            }
          }
        }
        if (has_dep) continue;

        ir::Node *dep_var = CreateControlDepVar();
        read_op->outputs.push_back(dep_var);
        dep_var->inputs.push_back(read_op);
        write_op->inputs.push_back(dep_var);
        dep_var->outputs.push_back(write_op);
      }
    }
  }
}

std::shared_ptr<Graph> Graph::Clone() {
  auto cloned_graph = std::make_shared<Graph>(this->program_);
  cloned_graph->ReleaseNodes();
  cloned_graph->num_node_created_ = 0;
  std::unordered_map<ir::Node *, ir::Node *> origin_to_cloned;
  for (auto *n : this->node_set_) {
    ir::Node *cloned_node = nullptr;
    if (n->IsCtrlVar()) {
      cloned_node = cloned_graph->CreateControlDepVar();
    } else if (!n->var_desc_ && !n->op_desc_) {  // empty node
      cloned_node = cloned_graph->CreateEmptyNode(n->Name(), n->NodeType());
    } else if (n->IsVar()) {
      cloned_node = cloned_graph->CreateVarNode(n->Var());
    } else if (n->IsOp()) {
      cloned_node = cloned_graph->CreateOpNode(n->Op());
    }
    if (cloned_node) {
      origin_to_cloned[n] = cloned_node;
    } else {
      PADDLE_THROW("The cloned node's type is not supported!");
    }
  }
  for (auto *n : this->node_set_) {
    for (auto it = n->inputs.begin(); it != n->inputs.end(); it++) {
      origin_to_cloned[n]->inputs.push_back(origin_to_cloned[*it]);
    }
    for (auto it = n->outputs.begin(); it != n->outputs.end(); it++) {
      origin_to_cloned[n]->outputs.push_back(origin_to_cloned[*it]);
    }
  }
  return cloned_graph;
}

bool IsControlDepVar(const ir::Node &var) {
  return var.Name().find(ir::Node::kControlDepVarName) != std::string::npos;
}
}  // namespace ir
}  // namespace framework
}  // namespace paddle
