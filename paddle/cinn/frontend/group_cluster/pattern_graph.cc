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
  VLOG(4) << "SinkTrivialPattern";
  SinkTrivialPattern();
  // ReducePattern -> ReduceTreePattern
  VLOG(4) << "ReduceLiftReduceTree";
  ReduceLiftReduceTree();

  VLOG(4) << "ReduceTreeGrown";
  ReduceTreeGrown();
  // ReduceTreePattern + TrivialPattern fusion.

  VLOG(4) << "ReduceTree_Trivial_Fusion";
  ReduceTree_Trivial_Fusion();

  VLOG(4) << "Start Pattern Flatten.";
  // TODO(wuzhanfei) need sort here, or do not return from all_pattern_nodes_
  std::vector<std::vector<const pir::Operation*>> result;
  std::transform(all_pattern_nodes_.begin(),
                 all_pattern_nodes_.end(),
                 std::back_inserter(result),
                 [](const PatternNodePtr node) { return node->GetOps(); });
  VLOG(4) << "ClusterOps returns " << result.size() << " Groups";
  return result;
}

void PatternGraph::SinkTrivialPattern() {
  // TODO(wuzhanfei): need consider Unsupport op here
  const auto& CanTrivialFuseIntoDownstream = [&](PatternNodePtr node) -> bool {
    for (const auto& downstream : node->downstream_) {
      return downstream->IsReduce() || downstream->IsTrivial();
    }
    return true;
  };

  const auto FindTrivialNode =
      [&](PatternNodePtrSet all_nodes) -> PatternNodePtr {
    for (PatternNodePtr node : all_nodes) {
      if (node->IsTrivial() && !node->downstream_.empty() &&
          CanTrivialFuseIntoDownstream(node)) {
        VLOG(4) << "FindTrivialNode: " << node;
        return node;
      }
    }
    return nullptr;
  };

  VLOG(4) << "Begin Graph is: ";
  PrintGraph();
  PatternNodePtr upstream;
  while ((upstream = FindTrivialNode(all_pattern_nodes_)) != nullptr) {
    VLOG(4) << "Start Finding Can Merge Trivial Node.";
    VLOG(4) << "Remain pattern node is: " << all_pattern_nodes_.size();
    PrintGraph();
    std::vector<PatternNodePtr> fusion_candidate = upstream->downstream_;
    upstream->downstream_.clear();
    for (const auto& downstream : fusion_candidate) {
      MergeNode(upstream, downstream);
      RemoveNode(downstream);
    }
    RemoveNode(upstream);
  }
  VLOG(4) << "End Graph is: ";
  PrintGraph();
}

void PatternGraph::MergeNode(const PatternNodePtr& upstream,
                             const PatternNodePtr& downstream) {
  PatternNodePtr merged_node =
      std::make_shared<PatternNode>(upstream, downstream);
  const auto RemoveFromVector = [](std::vector<PatternNodePtr>& vec,
                                   PatternNodePtr item) {
    auto iter = std::find(vec.begin(), vec.end(), item);
    if (iter != vec.end()) {
      vec.erase(iter);
    }
  };
  // deal with the reference.
  ExtendVector(&merged_node->upstream_, upstream->upstream_);
  ExtendVector(&merged_node->upstream_, downstream->upstream_);
  RemoveFromVector(merged_node->upstream_, upstream);

  ExtendVector(&merged_node->downstream_, upstream->downstream_);
  ExtendVector(&merged_node->downstream_, downstream->downstream_);
  RemoveFromVector(merged_node->downstream_, downstream);
  for (const auto& upstream_node : merged_node->upstream_) {
    RemoveFromVector(upstream_node->downstream_, upstream);
    RemoveFromVector(upstream_node->downstream_, downstream);
    upstream_node->downstream_.push_back(merged_node);
  }
  for (const auto& downstream_node : merged_node->downstream_) {
    RemoveFromVector(downstream_node->upstream_, upstream);
    RemoveFromVector(downstream_node->upstream_, downstream);
    downstream_node->upstream_.push_back(merged_node);
  }

  // deal with the graph storage.
  AppendNode(merged_node);
}

void PatternGraph::ReduceLiftReduceTree() {
  const auto FindCanLiftReducePattern =
      [](PatternNodePtrSet all_nodes) -> PatternNodePtr {
    for (PatternNodePtr node : all_nodes) {
      if (node->IsReduce() && !(node->downstream_.size() < 2)) return node;
    }
    return nullptr;
  };
  PatternNodePtr op;
  while ((op = FindCanLiftReducePattern(all_pattern_nodes_)) != nullptr) {
    const auto& reduce_pattern = ToReducePattern(op->stmt_pattern_);
    op->stmt_pattern_ = ReduceTreePattern({reduce_pattern}, reduce_pattern);
  }
}

void PatternGraph::ReduceTreeGrown() {
  const auto FindReduceTree =
      [](PatternNodePtrSet all_nodes) -> PatternNodePtr {
    for (PatternNodePtr node : all_nodes) {
      if (node->IsReduceTree() && !node->downstream_.empty() &&
          node->downstream_.at(0)->IsReduceTree())
        return node;
    }
    return nullptr;
  };
  PatternNodePtr upstream;
  while ((upstream = FindReduceTree(all_pattern_nodes_)) != nullptr) {
    CHECK_EQ(upstream->downstream_.size(), 1);
    auto downstream = upstream->downstream_.at(0);
    if (policy_manager_.CanFuse(upstream, downstream)) {
      PatternNodePtr new_node =
          std::make_shared<PatternNode>(upstream, downstream);
      AppendNode(new_node);
      RemoveNode(downstream);
      RemoveNode(upstream);
    }
  }
}

void PatternGraph::ReduceTree_Trivial_Fusion() {
  const auto FindReduceTree =
      [](PatternNodePtrSet all_nodes) -> PatternNodePtr {
    for (PatternNodePtr node : all_nodes) {
      if (node->IsReduceTree() && !node->downstream_.empty() &&
          node->downstream_.at(0)->IsTrivial())
        return node;
    }
    return nullptr;
  };
  PatternNodePtr upstream;
  while ((upstream = FindReduceTree(all_pattern_nodes_)) != nullptr) {
    CHECK_EQ(upstream->downstream_.size(), 1);
    auto downstream = upstream->downstream_.at(0);
    if (policy_manager_.CanFuse(upstream, downstream)) {
      PatternNodePtr new_node =
          std::make_shared<PatternNode>(upstream, downstream);
      AppendNode(new_node);
      RemoveNode(downstream);
      RemoveNode(upstream);
    }
  }
}

PatternGraph::PatternGraph(const std::vector<const pir::Operation*>& ops,
                           const policy::PolicyManager policy_manager)
    : policy_manager_(policy_manager) {
  std::unordered_map<const pir::Operation*, PatternNodePtr> op_to_node_map;

  for (const auto& op : ops) {
    PatternNodePtr node = std::make_shared<PatternNode>(op);
    op_to_node_map[op] = node;
    all_pattern_nodes_.emplace(node);
    node->sink_op_ = op;
  }

  for (const pir::Operation* op : ops) {
    PatternNodePtr cur_node = op_to_node_map[op];

    // add upstream nodes
    for (int i = 0; i < op->num_operands(); ++i) {
      ::pir::Operation* input_op = op->operand_source(i).defining_op();
      if (op_to_node_map.find(input_op) != op_to_node_map.end()) {
        PatternNodePtr upstream_node = op_to_node_map[input_op];
        cur_node->upstream_.push_back(upstream_node);
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

void PatternGraph::RemoveNode(const PatternNodePtr& node) {
  VLOG(4) << "Start Remove: " << node;
  if (all_pattern_nodes_.find(node) != all_pattern_nodes_.end()) {
    VLOG(4) << "Removed! ";
    all_pattern_nodes_.erase(node);
  }
  if (entrance_nodes_.find(node) != entrance_nodes_.end()) {
    entrance_nodes_.erase(node);
  }
  if (exit_nodes_.find(node) != exit_nodes_.end()) {
    exit_nodes_.erase(node);
  }
}

void PatternGraph::AppendNode(const PatternNodePtr& node) {
  all_pattern_nodes_.emplace(node);
  if (node->upstream_.empty()) {
    entrance_nodes_.emplace(node);
  }
  if (node->downstream_.empty()) {
    exit_nodes_.emplace(node);
  }
}

void PatternGraph::PrintGraph() {
  for (const auto& v : all_pattern_nodes_) {
    VLOG(4) << "Node: " << v;
    for (const auto& u : v->upstream_) {
      VLOG(4) << " -u>  " << u;
    }
    for (const auto& d : v->downstream_) {
      VLOG(4) << " <d- " << d;
    }
  }
}

}  // namespace cinn::frontend::group_cluster
