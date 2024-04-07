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
  VLOG(4) << "[Group Cluster] Initial Condition: " << GraphInfo();

  VLOG(4) << "[Group Cluster] Start SinkTrivialPattern";
  SinkTrivialPattern();
  VLOG(4) << "[Group Cluster] After SinkTrivialPattern: " << GraphInfo();

  // ReducePattern -> ReduceTreePattern
  VLOG(4) << "[Group Cluster] Start ReduceLiftReduceTree";
  ReduceLiftReduceTree();
  VLOG(4) << "[Group Cluster] After ReduceLiftReduceTree: " << GraphInfo();

  // ReduceTreePattern + ReduceTreePattern fusion
  VLOG(4) << "[Group Cluster] Start ReduceTreeGrown";
  ReduceTreeGrown();
  VLOG(4) << "[Group Cluster] After ReduceTreeGrown: " << GraphInfo();

  // ReduceTreePattern + TrivialPattern fusion.
  VLOG(4) << "[Group Cluster] Start ReduceTree_Trivial_Fusion";
  ReduceTree_Trivial_Fusion();
  VLOG(4) << "[Group Cluster] After ReduceTree_Trivial_Fusion: " << GraphInfo();

  // Horizontal fusion.
  if (with_horizontal_fusion) {
    VLOG(4) << "[Group Cluster] Start HorizontalFusion";
    HorizontalFusion();
    VLOG(4) << "[Group Cluster] After HorizontalFusion: " << GraphInfo();
  }

  return SortByTopoOrder();
}

std::vector<PatternNodePtr> PatternGraph::SortByTopoOrder() {
  // sort all_pattern_nodes_ by topo order.
  std::vector<PatternNodePtr> res;
  std::list<PatternNodePtr> topo_queue;
  std::map<PatternNodePtr, int> degree;
  for (const auto& node : all_pattern_nodes_) {
    degree[node] = node->upstream_.size();
    if (degree[node] == 0) {
      topo_queue.push_back(node);
    }
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
  GraphTransformer<
      NodePattern,
      And<And<NonSinkNodeMatcher, StmtPatternGraphMatcher<TrivialPattern>>,
          IsNotOutputNodeMatcher>,
      MergeTrivialPatternOperation>(this);
}

void PatternGraph::ReduceLiftReduceTree() {
  GraphTransformer<
      NodePattern,
      And<DownstreamSmallerThan<2>, StmtPatternGraphMatcher<ReducePattern>>,
      LiftReduceToReduceTreeOperation>(this);
}

void PatternGraph::HorizontalFusion() {
  GraphTransformer<NodePattern,
                   StmtPatternGraphMatcher<TrivialPattern>,
                   LiftToHorizontalFusionPatternOperation>(this);

  GraphTransformer<NodePairPattern,
                   HorizontalFusionConstrain,
                   HorizontalFusionOperation>(this);
}

void PatternGraph::ReduceTreeGrown() {
  GraphTransformer<NodePattern,
                   And<CanFuseReduceTreeMatcher, IsNotOutputNodeMatcher>,
                   MergeReduceTreeOperation>(this);
}

void PatternGraph::ReduceTree_Trivial_Fusion() {
  GraphTransformer<
      NodePattern,
      And<CanFuseReduceTreeAndTrivialMatcher, IsNotOutputNodeMatcher>,
      MergeReduceTreeAndTrivialOperation>(this);
}

PatternGraph::PatternGraph(const std::vector<pir::Operation*>& ops,
                           const std::vector<pir::Value>& outputs,
                           const policy::PolicyManager policy_manager,
                           const policy::PolicyManager topo_manager)
    : policy_manager_(policy_manager),
      topo_manager_(topo_manager),
      outputs_(outputs) {
  std::unordered_map<pir::Operation*, PatternNodePtr> op_to_node_map;

  VLOG(4) << "len(outputs) = " << outputs_.size();
  for (const auto& v : outputs) {
    VLOG(4) << "output is" << OpsDebugStr({v.defining_op()});
  }

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

  for (PatternNodePtr& upstream : node->upstream_) {
    RemoveFromVector(&upstream->downstream_, node);
  }

  for (PatternNodePtr& downstream : node->downstream_) {
    RemoveFromVector(&downstream->upstream_, node);
  }
}

void PatternGraph::AppendNode(const PatternNodePtr& node) {
  all_pattern_nodes_.emplace(node);
}

std::string PatternGraph::GraphInfo() const {
  std::stringstream ss;
  ss << "\n========= GraphInfo ===========";
  for (const auto& v : all_pattern_nodes_) {
    ss << "\n" << v->DebugStr();
    ss << "\n    IsOutput: " << IsOutputNodeMatcher()(*this, v);
  }
  ss << "\n===============================";
  return ss.str();
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
    RemoveFromVector(&upstream_node->downstream_, upstream);
    RemoveFromVector(&upstream_node->downstream_, downstream);
  }
  for (const auto& downstream_node : merged_node->downstream_) {
    downstream_node->upstream_.push_back(merged_node);
    RemoveFromVector(&downstream_node->downstream_, upstream);
    RemoveFromVector(&downstream_node->downstream_, downstream);
  }

  const auto vec_unique = [](const std::vector<PatternNodePtr>& vec) {
    auto set = std::unordered_set(vec.begin(), vec.end());
    return set.size() == vec.size();
  };

  CHECK(vec_unique(merged_node->upstream_));
  CHECK(vec_unique(merged_node->downstream_));

  // deal with the graph storage.
  AppendNode(merged_node);
  return merged_node;
}
}  // namespace cinn::frontend::group_cluster
