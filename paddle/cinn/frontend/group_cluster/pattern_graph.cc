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

std::vector<PatternNodePtr> PatternGraph::ClusterOps(
    bool with_horizontal_fusion) {
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

  // Horitical fusion.
  if (with_horizontal_fusion) {
    VLOG(4) << "Start Horitical Fusion.";
    HoriticalFusion();
  }

  return SortByTopoOrder();
}

std::vector<PatternNodePtr> PatternGraph::SortByTopoOrder() {
  // sort all_pattern_nodes_ by topo order.
  std::vector<PatternNodePtr> res;
  std::list<PatternNodePtr> topo_queue(entrance_nodes_.begin(),
                                       entrance_nodes_.end());
  std::map<PatternNodePtr, int> degree;
  for (const auto& node : all_pattern_nodes_) {
    degree[node] = node->upstream_.size();
  }
  while (!topo_queue.empty()) {
    PatternNodePtr node = topo_queue.front();
    topo_queue.pop_front();
    res.push_back(node);
    for (const auto& downstream_op : node->downstream_) {
      degree[downstream_op] = degree[downstream_op] - 1;
      if (degree[downstream_op] == 0) {
        topo_queue.push_back(downstream_op);
      }
    }
  }
  return res;
}

void PatternGraph::SinkTrivialPattern() {
  VLOG(4) << "SinkTrivialPattern";
  GraphTransformer<
      NodePattern,
      And<NonSinkNodeMatcher, StmtPatternGraphMatcher<TrivialPattern>>,
      TrivialPatternMerge>(this);
}

void PatternGraph::ReduceLiftReduceTree() {
  GraphTransformer<
      NodePattern,
      And<DownstreamSmallerThan<2>, StmtPatternGraphMatcher<ReducePattern>>,
      LiftReduceToReduceTree>(this);
}

void PatternGraph::HoriticalFusion() {
  VLOG(4) << "LiftToHorizontalFusionPattern";
  GraphTransformer<NodePattern,
                   StmtPatternGraphMatcher<TrivialPattern>,
                   LiftToHorizontalFusionPattern>(this);

  VLOG(4) << "HorizontalFusionOperation";
  GraphTransformer<NodePairPattern,
                   HorizontalFusionConstrain,
                   HorizontalFusionOperation>(this);

  VLOG(4) << "XK";
}

void PatternGraph::ReduceTreeGrown() {
  GraphTransformer<NodePattern,
                   CanFuseReduceTreeMatcher,
                   MergeReduceTreeOperation>(this);
}

void PatternGraph::ReduceTree_Trivial_Fusion() {
  PatternNodePtrSet visited;
  const auto FindReduceTree =
      [&](PatternNodePtrSet all_nodes) -> PatternNodePtr {
    for (PatternNodePtr node : all_nodes) {
      if (node->IsReduceTree() && !node->downstream_.empty() &&
          node->downstream_.at(0)->IsTrivial() &&
          visited.find(node) == visited.end()) {
        visited.emplace(node);
        return node;
      }
    }
    return nullptr;
  };
  PrintGraph();
  PatternNodePtr upstream;
  while ((upstream = FindReduceTree(all_pattern_nodes_)) != nullptr) {
    VLOG(4) << "Found A RT";
    CHECK_EQ(upstream->downstream_.size(), 1);
    auto downstream = upstream->downstream_.at(0);
    if (policy_manager_.CanFuse(upstream, downstream)) {
      VLOG(4) << "Start fuse";
      auto fake_reduce_iter_idx =
          policy_manager_.GetFakeReduceIterIdx(upstream, downstream);
      VLOG(4) << "fake_reduce_iter_idx ++: "
              << cinn::utils::Join(fake_reduce_iter_idx, ", ");
      PatternNodePtr merged_node = MergeNode(upstream, downstream);
      std::get<ReduceTreePlusTrivialPattern>(merged_node->stmt_pattern_)
          .fake_reduce_iter_idx = fake_reduce_iter_idx;
      VLOG(4) << "fake_reduce_iter_idx --: "
              << cinn::utils::Join(std::get<ReduceTreePlusTrivialPattern>(
                                       merged_node->stmt_pattern_)
                                       .fake_reduce_iter_idx,
                                   ", ");
      RemoveNode(downstream);
      RemoveNode(upstream);
    }
  }
}

PatternGraph::PatternGraph(const std::vector<pir::Operation*>& ops,
                           const policy::PolicyManager policy_manager,
                           const policy::PolicyManager topo_manager)
    : policy_manager_(policy_manager), topo_manager_(topo_manager) {
  std::unordered_map<pir::Operation*, PatternNodePtr> op_to_node_map;

  for (const auto& op : ops) {
    PatternNodePtr node = std::make_shared<PatternNode>(op);
    op_to_node_map[op] = node;
    all_pattern_nodes_.emplace(node);
    node->sink_op_ = op;
  }

  for (pir::Operation* op : ops) {
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

  for (PatternNodePtr& upstream : node->upstream_) {
    RemoveFromVector(&upstream->downstream_, node);
  }

  for (PatternNodePtr& downstream : node->downstream_) {
    RemoveFromVector(&downstream->upstream_, node);
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
    VLOG(4) << "Node: " << v << GetPatternName(v->stmt_pattern_);
    for (const auto& u : v->upstream_) {
      VLOG(4) << " -u>  " << u;
    }
    for (const auto& d : v->downstream_) {
      VLOG(4) << " <d- " << d;
    }
  }
}

PatternNodePtr PatternGraph::MergeNode(const PatternNodePtr& upstream,
                                       const PatternNodePtr& downstream) {
  PatternNodePtr merged_node =
      std::make_shared<PatternNode>(upstream, downstream);

  // deal with the reference.
  ExtendVector(&merged_node->upstream_, upstream->upstream_);
  ExtendVector(&merged_node->upstream_, downstream->upstream_);
  RemoveFromVector(&merged_node->upstream_, upstream);

  ExtendVector(&merged_node->downstream_, upstream->downstream_);
  ExtendVector(&merged_node->downstream_, downstream->downstream_);
  RemoveFromVector(&merged_node->downstream_, downstream);

  for (const auto& upstream_node : merged_node->upstream_) {
    upstream_node->downstream_.push_back(merged_node);
  }
  for (const auto& downstream_node : merged_node->downstream_) {
    downstream_node->upstream_.push_back(merged_node);
  }

  // deal with the graph storage.
  AppendNode(merged_node);
  return merged_node;
}

}  // namespace cinn::frontend::group_cluster
