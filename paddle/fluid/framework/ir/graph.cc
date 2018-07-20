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
/*
namespace {
void SortHelper(
    const std::map<ir::Node *, std::unordered_set<ir::Node *>> &adj_list,
    ir::Node *node, std::unordered_set<ir::Node *> *visited,
    std::vector<ir::Node *> *ret) {
  visited->insert(node);

  for (auto adj : adj_list.at(node)) {
    if (visited->find(adj) == visited->end()) {
      SortHelper(adj_list, adj, visited, ret);
    }
  }

  VLOG(3) << "topology sort insert: " << node->Name()
          << reinterpret_cast<void *>(node) << " input " << node->inputs.size();
  ret->push_back(node);
}

std::vector<ir::Node*> TopologySort(
    const std::map<ir::Node *, std::unordered_set<ir::Node *>> &adj_list) {
  std::unordered_set<ir::Node *> visited;
  std::vector<ir::Node *> ret;

  for (auto adj : adj_list) {
    if (visited.find(adj.first) == visited.end()) {
      SortHelper(adj_list, adj.first, &visited, &ret);
    }
  }
  return ret;
}
}  // namespace
*/

Graph::Graph(const ProgramDesc &program) : program_(program) {
  VLOG(3) << "block in program:" << program_.Size();
  std::unordered_map<std::string, VarDesc *> all_vars;
  for (auto *var : program.Block(0).AllVars()) {
    all_vars.emplace(var->Name(), var);
  }

  std::map<std::string, std::vector<ir::Node *>> var_nodes;
  for (auto *op : program.Block(0).AllOps()) {
    ir::Node *node = CreateOpNode(op);

    for (auto &each_var_name : op->InputArgumentNames()) {
      ir::Node *var = nullptr;
      if (var_nodes.find(each_var_name) != var_nodes.end()) {
        var = var_nodes.at(each_var_name).back();
      } else if (all_vars.count(each_var_name) != 0) {
        var = CreateVarNode(all_vars.at(each_var_name));
        var_nodes[each_var_name].push_back(var);
      } else {
        // TODO(paddle-dev): Seems some assumption doesn't hold?
        VLOG(3) << op->Type()
                << " input var not in all_var list: " << each_var_name;
        var = CreateEmptyNode(each_var_name, ir::Node::Type::kVariable);
        var_nodes[each_var_name].push_back(var);
      }
      node->inputs.push_back(var);
      var->outputs.push_back(node);
    }

    for (auto &each_var_name : op->OutputArgumentNames()) {
      ir::Node *var = CreateVarNode(all_vars.at(each_var_name));
      var_nodes[each_var_name].push_back(var);
      node->outputs.push_back(var);
      var->inputs.push_back(node);
    }
  }
  for (auto &var : var_nodes) {
    auto &versions = var.second;
    if (versions.size() <= 1) continue;

    auto it_new = versions.rbegin();
    auto it_old = versions.rbegin();
    ++it_old;
    for (; it_old != versions.rend(); it_new = it_old, ++it_old) {
      ir::Node *write_op =
          (*it_new)->inputs.empty() ? nullptr : (*it_new)->inputs[0];
      const auto &read_ops = (*it_old)->outputs;

      for (auto *read_op : read_ops) {
        // Manually add a dependency var from read_op to write_op;
        if (read_op == write_op) {
          // Read Write is the same op.
          continue;
        }
        ir::Node *dep_var = CreateEmptyNode(ir::Node::kControlDepVarName,
                                            ir::Node::Type::kVariable);
        read_op->outputs.push_back(dep_var);
        dep_var->inputs.push_back(read_op);
        write_op->inputs.push_back(dep_var);
        dep_var->outputs.push_back(write_op);
      }
    }
  }
}

/*
bool HasCircleHelper(ir::Node* node,
                     const std::map<ir::Node *, std::unordered_set<ir::Node *>>
&adj_list,
                     std::unordered_set<ir::Node*>* visited,
                     std::unordered_set<ir::Node*>* in_trace) {
  if (visited->find(node) == visited->end()) {
    visited->insert(node);
    in_trace->insert(node);

    for (ir::Node *in : adj_list.at(node)) {
      if (visited->find(in) == visited->end() &&
          HasCircleHelper(in, adj_list, visited, in_trace)) {
        return true;
      } else if (in_trace->find(in) != in_trace->end()) {
        return true;
      }
    }
  }
  in_trace->erase(node);
  return false;
}

bool HasCircle(const std::map<ir::Node *, std::unordered_set<ir::Node *>>
&adj_list) {
  std::unordered_set<ir::Node*> visited;
  std::unordered_set<ir::Node*> in_trace;
  for (auto& adj : adj_list) {
    if (HasCircleHelper(adj.first, adj_list, &visited, &in_trace)) {
      return true;
    }
  }
  return false;
}

std::map<ir::Node *, std::unordered_set<ir::Node *>> BuildAdjList(
    const std::vector<ir::Node*> &nodes) {
  std::map<ir::Node *, std::unordered_set<ir::Node *>> adj_list;

  for (auto &n : nodes) {
    if (n->NodeType() != ir::Node::Type::kOperation) continue;
    if (adj_list.find(n) == adj_list.end()) {
      adj_list[n] = std::unordered_set<ir::Node *>();
    }
    for (auto &var : n->inputs) {
      for (auto &adj_n : var->inputs) {
        PADDLE_ENFORCE(adj_n->NodeType() == ir::Node::Type::kOperation);
        adj_list[n].insert(adj_n);
        LOG(ERROR) << "adj " << adj_n->Name() << reinterpret_cast<void *>(adj_n)
                   << " -> " << n->Name() << reinterpret_cast<void *>(n)
                   << "  via " << var->Name() << reinterpret_cast<void *>(var);
      }
    }
  }
  return adj_list;
}

std::vector<ir::Node *> TopologySortOperationFromInToOut(
    const std::vector<std::unique_ptr<ir::Node>> &nodes) {
  std::vector<ir::Node*> tmp;
  for (auto& n : nodes) {
    tmp.push_back(n.get());
  }
  std::map<ir::Node *, std::unordered_set<ir::Node *>> adj_list =
BuildAdjList(tmp);

  PADDLE_ENFORCE(!HasCircle(adj_list));
  std::vector<ir::Node*> ret = TopologySort(adj_list);

  ir::Node *last_backward = nullptr;
  std::vector<ir::Node *> optimize_ops;
  for (ir::Node* n : ret) {
    if (boost::get<int>(
        n->Op()->GetAttr(OpProtoAndCheckerMaker::OpRoleAttrName())) ==
        static_cast<int>(OpRole::kBackward)) {
      last_backward = n;
    } else if (boost::get<int>(
        n->Op()->GetAttr(OpProtoAndCheckerMaker::OpRoleAttrName())) ==
        static_cast<int>(OpRole::kOptimize)) {
      optimize_ops.push_back(n);
    }
  }

  if (last_backward) {
    for (ir::Node *opt_node : optimize_ops) {
      ir::Node *dep_var = CreateEmptyNode(ir::Node::kControlDepVarName,
                                          ir::Node::Type::kVariable);
      last_backward->outputs.push_back(dep_var);
      dep_var->inputs.push_back(last_backward);
      opt_node->inputs.push_back(dep_var);
      dep_var->outputs.push_back(opt_node);
      VLOG(3) << "appending connect: " << last_backward->Name()
              << reinterpret_cast<void *>(last_backward) << "->"
              << opt_node->Name() << reinterpret_cast<void *>(opt_node);
    }
  }

  PADDLE_ENFORCE(!HasCircle(adj_list));
  for (ir::Node *n : ret) {
    std::unordered_set<ir::Node *> dummy;
    n->inputs.erase(
        std::remove_if(n->inputs.begin(), n->inputs.end(),
                       [n](ir::Node *t) {
                         return t->Name() == ir::Node::kControlDepVarName; }),
        n->inputs.end());
    n->outputs.erase(
        std::remove_if(n->outputs.begin(), n->outputs.end(),
                       [n](ir::Node *t) {
                         return t->Name() == ir::Node::kControlDepVarName; }),
        n->outputs.end());
  }
  return ret;
}*/

}  // namespace framework
}  // namespace paddle
