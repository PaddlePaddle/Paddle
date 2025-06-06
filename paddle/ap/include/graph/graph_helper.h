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

#pragma once

#include <map>
#include "glog/logging.h"
#include "paddle/ap/include/graph/adt.h"
#include "paddle/ap/include/graph/graph_descriptor.h"
#include "paddle/ap/include/graph/node.h"
#include "paddle/ap/include/graph/node_arena.h"

namespace ap::graph {

template <typename NodeT, typename TopoKind>
struct GraphHelper {
  explicit GraphHelper(const GraphDescriptor<NodeT, TopoKind>& graph_descriptor)
      : graph_descriptor_(graph_descriptor) {}

  GraphHelper(const GraphHelper&) = delete;
  GraphHelper(GraphHelper&&) = delete;

  adt::Result<NodeT> FindAnchor(const NodeT& start) {
    const auto& True = [](const auto&) -> adt::Result<bool> { return true; };
    ADT_LET_CONST_REF(opt_anchor, FilterAnchor(start, True));
    ADT_CHECK(opt_anchor.has_value()) << adt::errors::MismatchError{};
    return opt_anchor.value();
  }

  template <typename FilterT>
  adt::Result<std::optional<NodeT>> FilterAnchor(const NodeT& start,
                                                 const FilterT& Filter) {
    const auto topo_walker = GetTopoWalker();
    const auto IsSource = [&](const NodeT& sg_node) -> adt::Result<bool> {
      bool has_source = false;
      auto SetHasSource = [&](const NodeT&) -> adt::Result<adt::Ok> {
        has_source = true;
        return adt::Ok{};
      };
      ADT_RETURN_IF_ERR(
          graph_descriptor_.VisitUpstreamNodes(sg_node, SetHasSource));
      return !has_source;
    };
    const auto IsSink = [&](const NodeT& sg_node) -> adt::Result<bool> {
      bool has_sink = false;
      auto SetHasSink = [&](const NodeT&) -> adt::Result<adt::Ok> {
        has_sink = true;
        return adt::Ok{};
      };
      ADT_RETURN_IF_ERR(
          graph_descriptor_.VisitDownstreamNodes(sg_node, SetHasSink));
      return !has_sink;
    };
    std::unordered_set<NodeT> source_or_sinks;
    auto CollectStarts = [&](const NodeT& sg_node) -> adt::Result<adt::Ok> {
      ADT_LET_CONST_REF(ignored, graph_descriptor_.IgnoredNode(sg_node));
      if (ignored) {
        return adt::Ok{};
      }
      ADT_LET_CONST_REF(is_source, IsSource(sg_node));
      ADT_LET_CONST_REF(is_sink, IsSink(sg_node));
      if (is_source || is_sink) {
        source_or_sinks.insert(sg_node);
      }
      return adt::Ok{};
    };
    const auto bfs_walker_without_ignore = GetBfsWalkerWithoutIgnore();
    ADT_RETURN_IF_ERR(bfs_walker_without_ignore(start, CollectStarts));
    ADT_CHECK(source_or_sinks.size() > 0);
    std::unordered_map<NodeT, size_t> node2depth;
    std::map<size_t, std::vector<NodeT>> depth2nodes;
    const auto bfs_walker = GetBfsWalker();
    auto UpdateNodeDepth = [&](const NodeT& sg_node) -> adt::Result<adt::Ok> {
      size_t max_depth = 0;
      ADT_RETURN_IF_ERR(bfs_walker.VisitNextNodes(
          sg_node, [&](const NodeT& prev) -> adt::Result<adt::Ok> {
            const auto& iter = node2depth.find(prev);
            if (iter != node2depth.end()) {
              max_depth = std::max(max_depth, iter->second);
            }
            return adt::Ok{};
          }));
      node2depth[sg_node] = max_depth;
      depth2nodes[max_depth].push_back(sg_node);
      return adt::Ok{};
    };
    ADT_RETURN_IF_ERR(bfs_walker(
        source_or_sinks.begin(), source_or_sinks.end(), UpdateNodeDepth));
    for (auto iter = depth2nodes.rbegin(); iter != depth2nodes.rend(); ++iter) {
      for (const auto& node : iter->second) {
        ADT_LET_CONST_REF(is_op_node, this->graph_descriptor_.IsOpNode(node));
        if (is_op_node) {
          ADT_LET_CONST_REF(filter_success, Filter(node));
          if (filter_success) {
            return node;
          }
        }
      }
    }
    return std::nullopt;
  }

  adt::BfsWalker<NodeT> GetBfsWalker() {
    auto graph = this->graph_descriptor_;
    const auto& ForEachNext =
        [graph](const NodeT& node,
                const auto& VisitNext) -> adt::Result<adt::Ok> {
      auto DoEach = [&](const NodeT& next) -> adt::Result<adt::Ok> {
        ADT_LET_CONST_REF(is_ignored, graph.IgnoredNode(next));
        if (is_ignored) {
          return adt::Ok{};
        }
        return VisitNext(next);
      };
      ADT_RETURN_IF_ERR(graph.VisitDownstreamNodes(node, DoEach));
      ADT_RETURN_IF_ERR(graph.VisitUpstreamNodes(node, DoEach));
      return adt::Ok{};
    };
    return adt::BfsWalker<NodeT>(ForEachNext);
  }

  adt::BfsWalker<NodeT> GetBfsWalkerWithoutIgnore() {
    auto graph = this->graph_descriptor_;
    const auto& ForEachNext =
        [graph](const NodeT& node,
                const auto& VisitNext) -> adt::Result<adt::Ok> {
      ADT_RETURN_IF_ERR(graph.VisitDownstreamNodes(node, VisitNext));
      ADT_RETURN_IF_ERR(graph.VisitUpstreamNodes(node, VisitNext));
      return adt::Ok{};
    };
    return adt::BfsWalker<NodeT>(ForEachNext);
  }

  adt::TopoWalker<NodeT> GetTopoWalker() {
    auto graph = this->graph_descriptor_;
    const auto& ForEachPrev =
        [graph](const NodeT& node,
                const auto& VisitPrev) -> adt::Result<adt::Ok> {
      auto DoEach = [&](const NodeT& prev) -> adt::Result<adt::Ok> {
        ADT_LET_CONST_REF(is_ignored, graph.IgnoredNode(prev));
        if (is_ignored) {
          return adt::Ok{};
        }
        return VisitPrev(prev);
      };
      return graph.VisitUpstreamNodes(node, DoEach);
    };
    const auto& ForEachNext =
        [graph](const NodeT& node,
                const auto& VisitNext) -> adt::Result<adt::Ok> {
      auto DoEach = [&](const NodeT& next) -> adt::Result<adt::Ok> {
        ADT_LET_CONST_REF(is_ignored, graph.IgnoredNode(next));
        if (is_ignored) {
          return adt::Ok{};
        }
        return VisitNext(next);
      };
      return graph.VisitDownstreamNodes(node, DoEach);
    };
    return adt::TopoWalker<NodeT>(ForEachPrev, ForEachNext);
  }

 private:
  GraphDescriptor<NodeT, TopoKind> graph_descriptor_;
};

}  // namespace ap::graph
