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

#include <memory>
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

class Graph;
class GraphPatternDetector;
class Node;
namespace patterns {
struct Conv;
}  // namespace patterns

using graph_ptr = ir::Graph*;
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
  GraphWithStats FuseProjectionConv(
      const std::string& name_scope,
      const GraphWithStats& graph_with_stats) const;

  template <typename RetType>
  using GetNodeFunc =
      std::function<RetType(const GraphPatternDetector::subgraph_t& subgraph)>;
  using IdentityConvFunc = GetNodeFunc<std::tuple<Node*, Node*, Node*, Node*>>;
  using IdentityElementwiseAddFunc =
      GetNodeFunc<std::tuple<Node*, Node*, Node*>>;

  using ProjectionConvFunc = IdentityConvFunc;
  using ProjectionElementwiseAddFunc = GetNodeFunc<std::tuple<Node*, Node*>>;

  using CanFuseFunc = std::function<bool(Node*, Node*)>;

  std::tuple<Node*, Node*, Node*, Node*> GetNodesFromConv(
      const patterns::Conv& conv_pattern,
      const GraphPatternDetector::subgraph_t& subgraph) const;

  std::tuple<Node*, Node*, Node*, Node*> GetNodesFromProjectionConv(
      const patterns::Conv& conv_pattern,
      const GraphPatternDetector::subgraph_t& subgraph) const;

  template <typename HandleType, typename... OpFuncs>
  GraphWithStats ExecuteHandleOnGraph(GraphPatternDetector* gpd,
                                      const GraphWithStats& graph_with_stats,
                                      OpFuncs&&... op_funcs) const {
    ir::Graph* graph;
    int stats;

    std::tie(graph, stats) = graph_with_stats;

    auto can_fuse = [this](Node* op1, Node* op2) -> bool {
      return this->FindFuseOption(*op1, *op2) == FUSE_MKLDNN;
    };

    auto fuse_handle = HandleType{can_fuse, std::forward<OpFuncs>(op_funcs)...};

    (*gpd)(graph, fuse_handle);

    return std::make_pair(graph, stats + fuse_handle.get_stats());
  }

  struct IdentityFuseHandle {
    IdentityFuseHandle(
        const CanFuseFunc& can_fuse_func,
        const IdentityConvFunc& get_node_from_conv_op,
        const IdentityElementwiseAddFunc& get_node_from_elementwise_add_op);

    void operator()(const GraphPatternDetector::subgraph_t& subgraph,
                    Graph* graph);
    int get_stats() const { return *fusion_stats; }

   private:
    std::shared_ptr<int> fusion_stats;
    CanFuseFunc can_fuse_func;
    IdentityConvFunc get_node_from_conv_op;
    IdentityElementwiseAddFunc get_node_from_elementwise_add_op;
  };

  struct ProjectionFuseHandle {
    ProjectionFuseHandle(
        const CanFuseFunc& can_fuse_func,
        const ProjectionConvFunc& get_node_from_conv_x_op,
        const ProjectionConvFunc& get_node_from_conv_y_op,
        const ProjectionElementwiseAddFunc& get_node_from_elementwise_add_op);

    void operator()(const GraphPatternDetector::subgraph_t& subgraph,
                    Graph* graph);
    int get_stats() const { return *fusion_stats; }

   private:
    std::shared_ptr<int> fusion_stats;
    CanFuseFunc can_fuse_func;
    ProjectionConvFunc get_node_from_conv_x_op;
    ProjectionConvFunc get_node_from_conv_y_op;
    ProjectionElementwiseAddFunc get_node_from_elementwise_add_op;
  };

 public:
  virtual ~ResidualConnectionMKLDNNFusePass() {}

 protected:
  void ApplyImpl(graph_ptr graph) const;
  static bool HasFusedActivation(Node* conv_node) {
    return !(conv_node->Op()
                 ->GetAttrIfExists<std::string>("fuse_activation")
                 .empty());
  }

  const std::string name_scope_{"residual_connection_fuse_pass"};
};
}  // namespace ir
}  // namespace framework
}  // namespace paddle
