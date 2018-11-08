// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>
#include <tuple>
#include <utility>
#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"

#include <boost/optional.hpp>

namespace paddle {
namespace framework {
namespace ir {

using graph_ptr = std::unique_ptr<ir::Graph>;
using GraphWithStats = std::pair<ir::Graph*, int>;

void CorrectGraphEdges(Graph* graph, Node* from, Node* to);
bool IsReachable(ir::Graph* graph, Node* from, Node* to);
boost::optional<Node*> HasBias(const Node& op, const std::string& bias_name);

class ResidualConnectionMKLDNNFusePass : public FusePassBase {
 private:
  GraphWithStats FuseConvAsX(const std::string& name_scope,
                             const GraphWithStats& graph_with_stats) const;
  GraphWithStats FuseConvAsY(const std::string& name_scope,
                             const GraphWithStats& graph_with_stats) const;

  template <typename RetType>
  using GetNodeFunc =
      std::function<RetType(const GraphPatternDetector::subgraph_t& subgraph)>;
  using ConvFunc = GetNodeFunc<std::tuple<Node*, Node*, Node*, Node*>>;
  using ElementwiseAddFunc = GetNodeFunc<std::tuple<Node*, Node*, Node*>>;
  using CanFuseFunc = std::function<bool(Node*, Node*)>;

  std::tuple<Node*, Node*, Node*, Node*> GetNodesFromConv(
      const patterns::Conv& conv_pattern,
      const GraphPatternDetector::subgraph_t& subgraph) const;

  GraphWithStats ExecuteHandlerOnGraph(
      GraphPatternDetector* gpd, const GraphWithStats& graph_with_stats,
      const ConvFunc& get_node_from_conv,
      const ElementwiseAddFunc& get_node_from_elementwise_add) const;

  struct FuseHandler {
    FuseHandler(const ConvFunc& get_node_from_conv_op,
                const ElementwiseAddFunc& get_node_from_elementwise_add_op,
                const CanFuseFunc& can_fuse_func);

    void operator()(const GraphPatternDetector::subgraph_t& subgraph,
                    Graph* graph);
    int get_stats() const { return *fusion_stats; }

   private:
    std::shared_ptr<int> fusion_stats;
    ConvFunc get_node_from_conv_op;
    ElementwiseAddFunc get_node_from_elementwise_add_op;
    CanFuseFunc can_fuse_func;
  };

 public:
  virtual ~ResidualConnectionMKLDNNFusePass() {}

 protected:
  std::unique_ptr<ir::Graph> ApplyImpl(graph_ptr graph) const;

  const std::string name_scope_{"residual_connection_fuse_pass"};
};
}  // namespace ir
}  // namespace framework
}  // namespace paddle
