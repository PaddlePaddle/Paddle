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

#include "paddle/cinn/operator_fusion/pattern_graph.h"
#include <functional>
#include "paddle/cinn/operator_fusion/backend/pattern.h"
#include "paddle/cinn/operator_fusion/backend/pattern_fuser.h"
#include "paddle/cinn/operator_fusion/frontend/pattern.h"
#include "paddle/cinn/operator_fusion/frontend/pattern_fuser.h"

namespace cinn::fusion {

template <typename T>
std::vector<PatternNodePtr<T>> PatternGraph<T>::ClusterOps() {
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
  VLOG(4) << "[Group Cluster] Start HorizontalFusion";
  HorizontalFusion();
  VLOG(4) << "[Group Cluster] After HorizontalFusion: " << GraphInfo();

  return SortByTopoOrder();
}

template <typename T>
std::vector<PatternNodePtr<T>> PatternGraph<T>::SortByTopoOrder() {
  // sort all_pattern_nodes_ by topo order.
  std::vector<PatternNodePtr<T>> res;
  std::list<PatternNodePtr<T>> topo_queue;
  std::map<PatternNodePtr<T>, int> degree;
  for (const auto& node : all_pattern_nodes_) {
    degree[node] = node->upstream().size();
    if (degree[node] == 0) {
      topo_queue.push_back(node);
    }
  }
  while (!topo_queue.empty()) {
    PatternNodePtr<T> node = topo_queue.front();
    topo_queue.pop_front();
    res.push_back(node);
    for (const auto& downstream_op : node->downstream()) {
      degree[downstream_op] = degree[downstream_op] - 1;
      if (degree[downstream_op] == 0) {
        topo_queue.push_back(downstream_op);
      }
    }
  }
  return res;
}

template <typename T>
void PatternGraph<T>::SinkTrivialPattern() {
  GraphTransformer<
      NodePattern,
      T,
      And<And<NonSinkNodeMatcher, StmtPatternGraphMatcher<TrivialPattern<T>>>,
          IsNotOutputNodeMatcher>,
      MergeTrivialPatternOperation>(this);
}

template <typename T>
void PatternGraph<T>::ReduceLiftReduceTree() {
  GraphTransformer<
      NodePattern,
      T,
      And<DownstreamSmallerThan<2>, StmtPatternGraphMatcher<ReducePattern<T>>>,
      LiftReduceToReduceTreeOperation>(this);
}

template <typename T>
void PatternGraph<T>::HorizontalFusion() {
  GraphTransformer<NodePattern,
                   T,
                   Or<StmtPatternGraphMatcher<TrivialPattern<T>>,
                      StmtPatternGraphMatcher<ReduceTreePlusTrivialPattern<T>>,
                      StmtPatternGraphMatcher<ReducePattern<T>>,
                      StmtPatternGraphMatcher<ReduceTreePattern<T>>>,
                   LiftToHorizontalFusionPatternOperation>(this);

  GraphTransformer<NodePairPattern,
                   T,
                   And<HorizontalFusionConstrain<T>,
                       InputOutputMaximumConstrain<T>,
                       HorizontalCheckMiddleOutputVar<T>>,  // Avoid two many
                                                            // inputs and
                                                            // outputs.
                   HorizontalFusionOperation>(this);
}

template <typename T>
void PatternGraph<T>::ReduceTreeGrown() {
  GraphTransformer<NodePattern,
                   T,
                   And<CanFuseReduceTreeMatcher, IsNotOutputNodeMatcher>,
                   MergeReduceTreeOperation>(this);
}

template <typename T>
void PatternGraph<T>::ReduceTree_Trivial_Fusion() {
  GraphTransformer<
      NodePattern,
      T,
      And<CanFuseReduceTreeAndTrivialMatcher, IsNotOutputNodeMatcher>,
      MergeReduceTreeAndTrivialOperation>(this);
}

template <typename T>
PatternGraph<T>::PatternGraph(const std::vector<PatternContent<T>>& contents,
                              const std::vector<pir::Value>& outputs,
                              const PolicyManager<T> policy_manager,
                              const PolicyManager<T> topo_manager)
    : policy_manager_(policy_manager),
      topo_manager_(topo_manager),
      outputs_(outputs) {
  std::unordered_map<pir::Operation*, PatternNodePtr<T>> op_to_node_map;

  VLOG(4) << "len(outputs) = " << outputs_.size();
  for (const auto& v : outputs) {
    VLOG(4) << "output is" << OpsDebugStr({v.defining_op()});
  }

  for (const auto& content : contents) {
    PatternNodePtr<T> node = std::make_shared<PatternNode<T>>(content);
    op_to_node_map[content.op] = node;
    all_pattern_nodes_.emplace(node);
  }

  for (const auto& content : contents) {
    PatternNodePtr<T> cur_node = op_to_node_map[content.op];

    // add upstream nodes
    for (int i = 0; i < content.op->num_operands(); ++i) {
      ::pir::Operation* input_op = content.op->operand_source(i).defining_op();
      if (op_to_node_map.find(input_op) != op_to_node_map.end()) {
        PatternNodePtr<T> upstream_node = op_to_node_map[input_op];
        cur_node->AddNodeToUpstream(upstream_node);
      }
    }

    // add downstream nodes
    for (int i = 0; i < content.op->num_results(); ++i) {
      pir::Value related_value = content.op->result(i);
      for (auto consumer_it = related_value.use_begin();
           consumer_it != related_value.use_end();
           ++consumer_it) {
        ::pir::Operation* output_op = consumer_it->owner();
        if (op_to_node_map.find(output_op) != op_to_node_map.end()) {
          PatternNodePtr<T> downstream_node = op_to_node_map[output_op];
          cur_node->AddNodeToDownstream(downstream_node);
        }
      }
    }

    // unique all upstream / downstream node.
    // c = a + a ; then add will have 2 same upstream.
    cur_node->UniqueUpstream();
    cur_node->UniqueDownstream();
  }

  VLOG(4) << "PatternGraph Created, pattern node size: "
          << all_pattern_nodes_.size();
}

template <typename T>
void PatternGraph<T>::RemoveNode(const PatternNodePtr<T>& node) {
  VLOG(4) << "Start Remove: " << node;
  if (all_pattern_nodes_.find(node) != all_pattern_nodes_.end()) {
    VLOG(4) << "Removed! ";
    all_pattern_nodes_.erase(node);
  }

  for (const PatternNodePtr<T>& upstream : node->upstream()) {
    upstream->RemoveNodeFromDownstream(node);
  }

  for (const PatternNodePtr<T>& downstream : node->downstream()) {
    downstream->RemoveNodeFromUpstream(node);
  }
}

template <typename T>
void PatternGraph<T>::AppendNode(const PatternNodePtr<T>& node) {
  all_pattern_nodes_.emplace(node);
}

template <typename T>
std::string PatternGraph<T>::GraphInfo() const {
  std::stringstream ss;
  ss << "\n========= GraphInfo ===========";
  for (const auto& v : all_pattern_nodes_) {
    ss << "\n" << v->DebugStr();
    ss << "\n    IsOutput: " << IsOutputNodeMatcher()(*this, v);
    ss << "\n    Loop Framework is: " << GetLoopFramework(v->stmt_pattern());
  }
  ss << "\n===============================";
  return ss.str();
}

template <typename T>
PatternNodePtr<T> PatternGraph<T>::MergeNode(
    const PatternNodePtr<T>& upstream,
    const PatternNodePtr<T>& downstream,
    MergePatternFn<T> merge_pattern_fn) {
  PatternNodePtr<T> merged_node =
      std::make_shared<PatternNode<T>>(upstream, downstream, merge_pattern_fn);

  // Update upstream and downstream nodes.
  for (const auto& upstream_node : merged_node->upstream()) {
    upstream_node->AddNodeToDownstream(merged_node);
    upstream_node->RemoveNodeFromDownstream(upstream);
    upstream_node->RemoveNodeFromDownstream(downstream);
  }
  for (const auto& downstream_node : merged_node->downstream()) {
    downstream_node->AddNodeToUpstream(merged_node);
    downstream_node->RemoveNodeFromDownstream(upstream);
    downstream_node->RemoveNodeFromDownstream(downstream);
  }

  const auto vec_unique = [](const std::vector<PatternNodePtr<T>>& vec) {
    auto set = std::unordered_set(vec.begin(), vec.end());
    return set.size() == vec.size();
  };

  PADDLE_ENFORCE_EQ(
      vec_unique(merged_node->upstream()),
      true,
      phi::errors::PreconditionNotMet(
          "The upstream nodes of the merged node are not unique."));
  PADDLE_ENFORCE_EQ(
      vec_unique(merged_node->downstream()),
      true,
      phi::errors::PreconditionNotMet(
          "The downstream nodes of the merged node are not unique."));

  // deal with the graph storage.
  AppendNode(merged_node);
  return merged_node;
}

template class PatternGraph<FrontendStage>;
template class PatternGraph<BackendStage>;

}  // namespace cinn::fusion
