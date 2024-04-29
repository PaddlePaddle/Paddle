// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#pragma once
/**
 * This file defines several graphs used by scheduler.
 */

#include <memory>
#include <string>
#include <vector>

#include "paddle/cinn/common/graph_utils.h"
#include "paddle/cinn/poly/stage.h"

namespace cinn {
namespace poly {

struct DataFlowGraphNode : public cinn::common::GraphNode {
  //! Used for union find to gather groups.
  DataFlowGraphNode* group_parent{};
  //! Each stage belongs to a node.
  Shared<Stage> stage;

  explicit DataFlowGraphNode(Stage* stage) : stage(stage) {}

  std::string id() const override;

  //! Get the ancestor.
  const DataFlowGraphNode* group_ancestor() const;
  DataFlowGraphNode* group_ancestor();

  //! Get the tree height for union find.
  int group_height() const;

  //! Tell whether this node is connected to another `node`, either inlink or
  //! outlink.
  bool IsLinkedTo(const DataFlowGraphNode* node) const;

  //! Merge two nodes into the same group.
  //! returns: the common ancestor.
  static DataFlowGraphNode* MergeGroup(DataFlowGraphNode* a,
                                       DataFlowGraphNode* b);

  //! Compare the the iteration_domain.apply(transform), return true if same.
  static bool TransformedDomainIsSame(const DataFlowGraphNode* a,
                                      const DataFlowGraphNode* b);
};

struct DataFlowGraphEdge : public cinn::common::GraphEdge {};

/**
 * DataFlowGraph help to record the data dependencies between the Stages.
 */
struct DataFlowGraph : public cinn::common::Graph {};

/**
 * Create a dependency graph given some stages.
 * NOTE The stages should sorted in topological order.
 *
 * @param stages The stages.
 * @param extra_links The extra links, each element is a pair of (a ->  b)
 */
std::unique_ptr<DataFlowGraph> CreateGraph(
    const std::vector<Stage*>& stages,
    const std::vector<std::pair<std::string, std::string>>& extra_links = {});

namespace detail {

struct Group {
  Group() = default;
  explicit Group(const std::vector<Shared<DataFlowGraphNode>>& nodes)
      : nodes(nodes) {}

  std::vector<Shared<DataFlowGraphNode>> nodes;
  std::vector<std::string> dimension_names;
};

/**
 * GraphPartitionBySpace partitions a data flow graph into several sub-graph
 * with consider of the dependency and space of the iteration domain. If two
 * Nodes has the stages has dependency relation and has the same iteration
 * domain, then they will be put in the same sub-graph.
 */
std::vector<Group> PartitionGraphByIterationDomain(cinn::common::Graph* graph);

}  // namespace detail

}  // namespace poly
}  // namespace cinn
