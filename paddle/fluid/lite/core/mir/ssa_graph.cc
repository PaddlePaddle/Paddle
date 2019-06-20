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

#include "paddle/fluid/lite/core/mir/ssa_graph.h"
#include <algorithm>
#include <memory>
#include <set>
#include <unordered_map>
#include <utility>

namespace paddle {
namespace lite {
namespace mir {

bool SSAGraph::CheckBidirectionalConnection() {
  VLOG(4) << "node count " << node_storage_.size();
  for (auto &node : node_storage_) {
    if (node.IsStmt()) VLOG(4) << node.AsStmt().op_info()->Type();
    if (node.IsArg()) VLOG(4) << node.AsArg().name << " " << node.AsArg().id;
    for (auto *in : node.inlinks) {
      CHECK(in->outlinks.end() !=
            std::find(in->outlinks.begin(), in->outlinks.end(), &node));
    }
    for (auto *out : node.outlinks) {
      CHECK(out->inlinks.end() !=
            std::find(out->inlinks.begin(), out->inlinks.end(), &node));
    }
  }
  return true;
}

std::map<mir::Node *, std::set<mir::Node *>> SSAGraph::BuildOperationAdjList() {
  std::map<mir::Node *, std::set<mir::Node *>> adj_list;

  for (auto &n : mutable_nodes()) {
    if (!n.IsStmt()) continue;
    if (adj_list.find(&n) == adj_list.end()) {
      adj_list[&n] = std::set<mir::Node *>();
    }
    std::vector<mir::Node *> nodes;
    for (auto &var : n.inlinks) {
      for (auto &adj_n : var->inlinks) {
        CHECK(adj_n->IsStmt());
        nodes.push_back(adj_n);
      }
    }
    std::sort(nodes.begin(), nodes.end(),
              [](mir::Node *node1, mir::Node *node2) { return node1 > node2; });
    adj_list[&n].insert(std::make_move_iterator(nodes.begin()),
                        std::make_move_iterator(nodes.end()));
  }
  return adj_list;
}

void SSAGraph::SortHelper(
    const std::map<mir::Node *, std::set<mir::Node *>> &adj_list,
    mir::Node *node, std::set<mir::Node *> *visited,
    std::vector<mir::Node *> *ret) {
  visited->insert(node);

  for (auto adj : adj_list.at(node)) {
    if (visited->find(adj) == visited->end()) {
      SortHelper(adj_list, adj, visited, ret);
    }
  }

  ret->push_back(node);
}

std::vector<mir::Node *> SSAGraph::StmtTopologicalOrder() {
  CheckBidirectionalConnection();

  std::stack<mir::Node *> stack;
  std::set<mir::Node *> visited;
  std::vector<mir::Node *> res;

  auto adj_list = BuildOperationAdjList();

  for (auto adj : adj_list) {
    if (visited.find(adj.first) == visited.end()) {
      SortHelper(adj_list, adj.first, &visited, &res);
    }
  }

  return res;
}

Node *SSAGraph::GraphCreateInstructNode(
    const std::shared_ptr<OpLite> &op, const std::vector<Place> &valid_places) {
  node_storage_.emplace_back();
  // TODO(Superjomn) remove one valid_places here.
  op->SetValidPlaces(valid_places);
  auto &new_node = node_storage_.back();
  auto kernels = op->CreateKernels(valid_places);
  node_storage_.back().AsStmt(op->op_type_, std::move(kernels), op);

  CHECK(new_node.inlinks.empty()) << "duplicate Build found";
  CHECK(new_node.outlinks.empty()) << "duplicate Build found";
  return &node_storage_.back();
}

void SSAGraph::Build(const Program &program,
                     const std::vector<Place> &valid_places) {
  CHECK(node_storage_.empty());

  auto weights_name = program.weights();
  auto is_weights = [&](const std::string &name) -> bool {
    auto it = std::find(weights_name.begin(), weights_name.end(), name);
    if (it == weights_name.end()) return false;
    return true;
  };

  std::unordered_map<std::string, mir::Node *> arg_update_node_map_;
  for (auto &op : program.ops()) {
    VLOG(3) << op->op_info()->Type();
    auto *op_node = GraphCreateInstructNode(op, valid_places);
    for (const std::string &name : op->op_info()->input_names()) {
      mir::Node *arg_node = nullptr;
      if (arg_update_node_map_.count(name)) {
        arg_node = arg_update_node_map_.at(name);
      } else {
        node_storage_.emplace_back();
        arg_node = &node_storage_.back();
        arg_node->AsArg(name, node_storage_.size() - 1);
        arg_update_node_map_[name] = arg_node;
      }
      if (is_weights(name)) arg_node->AsArg().is_weight = true;
      CHECK(arg_node->IsRoleSet());
      DirectedLink(arg_node, op_node);
    }
    for (const std::string &name : op->op_info()->output_names()) {
      node_storage_.emplace_back();
      auto *arg_node = &node_storage_.back();
      arg_node->AsArg(name, node_storage_.size() - 1);
      arg_update_node_map_[name] = arg_node;

      if (is_weights(name)) arg_node->AsArg().is_weight = true;
      CHECK(arg_node->IsRoleSet());
      DirectedLink(op_node, arg_node);
    }
    CHECK(CheckLinksRoleSet());
  }

  CHECK(CheckNodesRoleSet());
  CheckValid();
}

void SSAGraph::RemoveNode(const mir::Node *node) {
  auto pos = std::find_if(node_storage_.begin(), node_storage_.end(),
                          [&node](mir::Node &n) { return &n == node; });
  CHECK(pos != node_storage_.end());
  node_storage_.erase(pos);
}

mir::Node *SSAGraph::Argument(const std::string &name) {
  auto it = arguments_.find(name);
  CHECK(it != arguments_.end()) << "no argument called " << name;
  return it->second;
}

std::vector<mir::Node *> SSAGraph::inputs() {
  std::vector<mir::Node *> res;
  for (auto &node : node_storage_) {
    if (node.inlinks.empty()) {
      res.push_back(&node);
    }
  }
  return res;
}

std::vector<mir::Node *> SSAGraph::outputs() {
  std::vector<mir::Node *> res;
  for (auto &node : node_storage_) {
    if (node.outlinks.empty()) {
      res.push_back(&node);
    }
  }
  return res;
}

mir::Node *SSAGraph::RetrieveArgument(const std::string &arg) {
  auto it = arguments_.find(arg);
  if (it != arguments_.end()) {
    return it->second;
  }
  return nullptr;
}

bool SSAGraph::CheckNodesRoleSet() {
  for (auto &node : mutable_nodes()) {
    CHECK_OR_FALSE(node.IsRoleSet());
  }
  return true;
}

bool SSAGraph::CheckLinksRoleSet() {
  for (auto &node : mutable_nodes()) {
    CHECK_OR_FALSE(node.IsRoleSet());
    if (!node.IsStmt()) continue;
    for (auto *x : node.inlinks) {
      CHECK_OR_FALSE(x->IsRoleSet());
      CHECK_OR_FALSE(x->IsArg());
    }
    for (auto *x : node.outlinks) {
      CHECK_OR_FALSE(x->IsRoleSet());
      CHECK_OR_FALSE(x->IsArg());
    }
  }
  return true;
}

Node *SSAGraph::NewArgumentNode(const std::string &name) {
  node_storage_.emplace_back();
  auto &arg_node = node_storage_.back();
  arg_node.AsArg(name, node_storage_.size() - 1);
  return &arg_node;
}

Node *SSAGraph::NewInstructNode() {
  node_storage_.emplace_back();
  return &node_storage_.back();
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
