// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/cinn/frontend/group_cluster/pattern_graph.h"

namespace cinn::frontend::group_cluster {

std::vector<std::vector<const pir::Operation*>> PatternGraph::ClusterOps() {
  SinkTrivialPattern();
  FuseReducePattern();
  // TODO(wuzhanfei) need sort here, or do not return from all_pattern_nodes_
  std::vector<std::vector<const pir::Operation*>> result;
  std::transform(all_pattern_nodes_.begin(),
                 all_pattern_nodes_.end(),
                 std::back_inserter(result),
                 [](const PatternNodePtr node) { return node->GetOps(); });
  return result;
}

void PatternGraph::SinkTrivialPattern() {
  // TODO(wuzhanfei): need consider Unsupport op here
  const auto FindTrivialNode =
      [](std::unordered_set<PatternNodePtr> all_nodes) -> PatternNodePtr {
    for (PatternNodePtr node : all_nodes) {
      if (node->IsTrivial() && !node->downstream_.empty()) return node;
    }
    return nullptr;
  };

  PatternNodePtr upstream;
  while ((upstream = FindTrivialNode(all_pattern_nodes_)) != nullptr) {
    std::vector<PatternNodePtr> fusion_candidate = upstream->downstream_;
    upstream->downstream_.clear();
    for (const auto& downstream : fusion_candidate) {
      PatternNodePtr new_node =
          std::make_shared<PatternNode>(upstream, downstream);
      AppendNode(new_node);
      RemoveNode(downstream);
    }
    RemoveNode(upstream);
  }
}

void PatternGraph::FuseReducePattern() {
  // TODO(wuzhanfei) reduce fusion, similar with implementation in backend
}

PatternGraph::PatternGraph(const std::vector<const pir::Operation*>& ops,
                           const policy::PolicyManager policy_manager)
    : policy_manager_(policy_manager) {
  std::unordered_map<const pir::Operation*, PatternNodePtr> op_to_node_map;

  for (int i = 0; i < ops.size(); ++i) {
    PatternNodePtr node = std::make_shared<PatternNode>(ops[i]);
    op_to_node_map[ops[i]] = node;
    all_pattern_nodes_.emplace(node);
    node->sink_op_ = ops[i];
  }

  for (const pir::Operation* op : ops) {
    PatternNodePtr cur_node = op_to_node_map[op];

    // add upstream nodes
    for (int i = 0; i < op->num_operands(); ++i) {
      ::pir::Operation* input_op = op->operand_source(i).defining_op();
      if (op_to_node_map.find(input_op) != op_to_node_map.end()) {
        PatternNodePtr upstream_node = op_to_node_map[input_op];
        cur_node->upstream_.push_back(upstream_node);
        upstream_node->downstream_.push_back(cur_node);
      }
    }

    // add downstream nodes
    for (int i = 0; i < op->num_results(); ++i) {
      pir::Value related_value = op->result(i);
      for (auto consumer_it = related_value.use_begin();
           consumer_it != related_value.use_end();
           ++consumer_it) {
        ::pir::Operation* output_op = consumer_it->owner();
        if (op_to_node_map.find(output_op) != op_to_node_map.end()) {
          PatternNodePtr downstream_node = op_to_node_map[output_op];
          cur_node->downstream_.push_back(downstream_node);
          downstream_node->upstream_.push_back(cur_node);
        }
      }
    }

    if (cur_node->upstream_.empty()) {
      entrance_nodes_.emplace(cur_node);
    }

    if (cur_node->downstream_.empty()) {
      exit_nodes_.emplace(cur_node);
    }
  }

  VLOG(4) << "PatternGraph Created, pattern node size: "
          << all_pattern_nodes_.size();
}

void PatternGraph::RemoveNode(PatternNodePtr node) {
  if (all_pattern_nodes_.find(node) != all_pattern_nodes_.end()) {
    all_pattern_nodes_.erase(node);
  }
  if (entrance_nodes_.find(node) != entrance_nodes_.end()) {
    entrance_nodes_.erase(node);
  }
  if (exit_nodes_.find(node) != exit_nodes_.end()) {
    exit_nodes_.erase(node);
  }
}

void PatternGraph::AppendNode(PatternNodePtr node) {
  all_pattern_nodes_.emplace(node);
  if (node->upstream_.empty()) {
    entrance_nodes_.emplace(node);
  }
  if (node->downstream_.empty()) {
    exit_nodes_.emplace(node);
  }
}

}  // namespace cinn::frontend::group_cluster
