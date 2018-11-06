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
#include <unordered_set>

#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/var_desc.h"

namespace paddle {
namespace framework {
namespace ir {
namespace {

void CheckProgram(const ProgramDesc &program) {
#define _INT(role) static_cast<int>(role)

  std::map<int, bool> visit;
  for (OpDesc *op : program.Block(0).AllOps()) {
    // For backward compatibility, some program doesn't have role added.
    if (!op->HasAttr(OpProtoAndCheckerMaker::OpRoleAttrName())) continue;
    int role_id =
        boost::get<int>(op->GetAttr(OpProtoAndCheckerMaker::OpRoleAttrName()));
    visit[role_id] = true;
    switch (role_id) {
      case _INT(OpRole::kForward):
        if (visit.find(_INT(OpRole::kBackward)) != visit.end()) {
          LOG(ERROR)
              << "Cannot add backward operator before forward operator %s."
              << op->Type();
        }
        break;
      case _INT(OpRole::kBackward):
      case _INT(OpRole::kBackward) | _INT(OpRole::kLoss):
        PADDLE_ENFORCE(
            visit.find(_INT(OpRole::kOptimize)) == visit.end(),
            "Cannot add backward operator %s after optimize operator.",
            op->Type());
        break;
      case _INT(OpRole::kForward) | _INT(OpRole::kLoss):
        PADDLE_ENFORCE(visit.find(_INT(OpRole::kBackward) |
                                  _INT(OpRole::kLoss)) == visit.end(),
                       "Cannot add backward|loss operator before "
                       "forward|loss operator %s.",
                       op->Type());
        PADDLE_ENFORCE(
            visit.find(_INT(OpRole::kOptimize)) == visit.end(),
            "Cannot add forward|loss operator %s after optimize operator.",
            op->Type());
        break;
      case _INT(OpRole::kOptimize):
      case _INT(OpRole::kOptimize) | _INT(OpRole::kLRSched):
        PADDLE_ENFORCE(visit.find(_INT(OpRole::kBackward)) != visit.end(),
                       "Optimize operators %s must follow backward operator.",
                       op->Type());
        break;
      case _INT(OpRole::kLRSched):
      case _INT(OpRole::kDist):
      case _INT(OpRole::kRPC):
      case _INT(OpRole::kNotSpecified):
        break;
      default:
        LOG(FATAL) << "Unknown operator role. Don't add new role because "
                      "you don't know what you are doing.";
    }
  }

#undef _INT
}
}  // namespace

Graph::Graph(const ProgramDesc &program) : program_(program) {
  CheckProgram(program_);
  // Make the nodes id start from 0.
  Node::ResetId();
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
    for (auto &each_var_name : op->OutputArgumentNames()) {
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
  return std::move(var_nodes);
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

      PADDLE_ENFORCE(write_op, "The write_op should not be empty.");

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

bool IsControlDepVar(const ir::Node &var) {
  return var.Name().find(ir::Node::kControlDepVarName) != std::string::npos;
}
}  // namespace ir
}  // namespace framework
}  // namespace paddle
