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
//! \file This file contains the utilities of graph.

#include <absl/container/flat_hash_map.h>
#include <glog/logging.h>

#include <algorithm>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "paddle/cinn/common/object.h"
#include "paddle/cinn/common/shared.h"
#include "paddle/cinn/common/type.h"
#include "paddle/common/enforce.h"
namespace cinn {
namespace common {

#ifdef As
#undef As
#endif

class GraphNode;

/**
 * Edge in the graph, which can hold some attributes.
 */
class GraphEdge : public Object {
 public:
  GraphEdge(GraphNode* source, GraphNode* sink, int index = -1)
      : source_(source), sink_(sink), index_(index) {}

  GraphNode* source() const { return source_; }
  GraphNode* sink() const { return sink_; }
  const char* type_info() const override { return __type_info__; }
  int index() const { return index_; }

 private:
  //! the index in sink node's inlinks_ or source node's outlinks_
  //! this is used to keep the input/output tensor's order of operator node
  int index_{-1};
  //! Source of this edge.
  GraphNode* source_{};
  //! End of this edge.
  GraphNode* sink_{};
  static constexpr char* __type_info__ = "graph_edge";
};

struct GraphEdgeCompare {
  bool operator()(const cinn::common::Shared<GraphEdge>& a,
                  const cinn::common::Shared<GraphEdge>& b) const;
};

/**
 * @brief The base class of all node of graph.
 * This is used to normalize and share the graph operations.
 */
class GraphNode : public Object {
 public:
  //! The unique identifier of the node.
  virtual std::string id() const = 0;
  inline int get_index() { return index; }
  inline void set_index(int index) { this->index = index; }

  //! Links from this to other.
  template <typename EdgeT = GraphEdge>
  std::tuple<EdgeT*, EdgeT*> LinkTo(GraphNode* other) {
    EdgeT *a, *b;
    PADDLE_ENFORCE_NOT_NULL(
        other, ::common::errors::InvalidArgument("The input node is null."));
    PADDLE_ENFORCE_NE(
        other,
        this,
        ::common::errors::InvalidArgument("Cannot link to itself"));
    auto outlink_edge = make_shared<GraphEdge>(this, other, index_outlinks);
    auto inlink_edge =
        make_shared<GraphEdge>(this, other, other->index_inlinks);
    index_outlinks++;
    other->index_inlinks++;
    outlinks_.insert(outlink_edge);
    other->inlinks_.insert(inlink_edge);

    for (auto& item : outlinks_) {
      if (item->index() == index_outlinks - 1) {
        a = static_cast<EdgeT*>(item.get());
        break;
      }
    }
    for (auto& item : other->inlinks_) {
      if (item->index() == other->index_inlinks - 1) {
        b = static_cast<EdgeT*>(item.get());
        break;
      }
    }
    PADDLE_ENFORCE_NOT_NULL(
        a, ::common::errors::InvalidArgument("Sorry,but outlinks is nullptr"));
    PADDLE_ENFORCE_NOT_NULL(b,
                            ::common::errors::InvalidArgument(
                                "Sorry, but other->inlinks_ is nullptr"));
    return std::make_tuple(a, b);
  }

  void Controls(GraphNode* other) {
    bool outlink_linked = false;
    bool inlink_linked = false;
    for (auto& item : outlinks_) {
      if (item->sink()->id() == other->id()) {
        outlink_linked = true;
        break;
      }
    }
    for (auto& item : other->inlinks_) {
      if (item->source()->id() == this->id()) {
        inlink_linked = true;
        break;
      }
    }
    PADDLE_ENFORCE_EQ(outlink_linked,
                      inlink_linked,
                      ::common::errors::InvalidArgument(
                          "The outlink_linked should same as inlink_linked."));
    if (outlink_linked)
      return;
    else
      this->LinkTo(other);
  }

  void UnLinkAllTo(GraphNode* other) {
    if (other == this) return;
    // remove all this node's outlink
    {
      auto it = std::find_if(
          outlinks_.begin(), outlinks_.end(), [&](const Shared<GraphEdge>& x) {
            return x->source() == this && x->sink() == other;
          });
      while (it != outlinks_.end()) {
        outlinks_.erase(it);
        it = std::find_if(outlinks_.begin(),
                          outlinks_.end(),
                          [&](const Shared<GraphEdge>& x) {
                            return x->source() == this && x->sink() == other;
                          });
      }
    }
    // remove all other node's inlink
    {
      auto it = std::find_if(other->inlinks_.begin(),
                             other->inlinks_.end(),
                             [&](const Shared<GraphEdge>& x) {
                               return x->source() == this && x->sink() == other;
                             });
      while (it != other->inlinks_.end()) {
        other->inlinks_.erase(it);
        it = std::find_if(other->inlinks_.begin(),
                          other->inlinks_.end(),
                          [&](const Shared<GraphEdge>& x) {
                            return x->source() == this && x->sink() == other;
                          });
      }
    }
  }

  void UnLinkSingleTo(GraphNode* other) {
    if (other == this) return;
    // remove single outlink
    {
      auto it = std::find_if(
          outlinks_.begin(), outlinks_.end(), [&](const Shared<GraphEdge>& x) {
            return x->source() == this && x->sink() == other;
          });
      if (it != outlinks_.end()) outlinks_.erase(it);
    }
    // remove single inlink
    {
      auto it = std::find_if(other->inlinks_.begin(),
                             other->inlinks_.end(),
                             [&](const Shared<GraphEdge>& x) {
                               return x->source() == this && x->sink() == other;
                             });
      if (it != other->inlinks_.end()) other->inlinks_.erase(it);
    }
  }

  bool IsLinkedTo(GraphNode* other) const {
    for (auto& e : outlinks_) {
      if (e->sink()->id() == other->id()) return true;
    }
    return false;
  }

  //! Get the input links of the node.
  virtual const std::set<Shared<GraphEdge>, GraphEdgeCompare>& inlinks() const {
    return inlinks_;
  }
  //! Get the output links of the node.
  virtual const std::set<Shared<GraphEdge>, GraphEdgeCompare>& outlinks()
      const {
    return outlinks_;
  }

  //! Reset graph traversal meta info.
  void ResetVisitMeta() { visited_time_ = 0; }
  void VisitOnce() const { visited_time_++; }
  bool visited() const {
    return inlinks_.empty() || visited_time_ == inlinks_.size();
  }

  const char* type_info() const override { return __type_info__; }

  GraphNode() = default;

  static const char* __type_info__;

 protected:
  //! The input links of the node.
  //! \note We record the raw pointer rather than the shared pointer to avoid
  //! cycle reference.
  std::set<cinn::common::Shared<GraphEdge>, GraphEdgeCompare> inlinks_;
  //! The output links of the node.
  //! \note We record the raw pointer rather than the shared pointer to avoid
  //! cycle reference.
  std::set<cinn::common::Shared<GraphEdge>, GraphEdgeCompare> outlinks_;

  mutable int visited_time_{};
  //! used to mark the index of node's input/output tensors
  int index_inlinks{0};
  int index_outlinks{0};
  int index{0};
};

/**
 * @brief The base class of all the graph.
 */
class Graph {
 public:
  using node_order_t = std::vector<GraphNode*>;
  using edge_order_t = std::vector<GraphEdge*>;

  //! Add a node to the graph.
  //! @{
  GraphNode* RegisterNode(size_t key, GraphNode* node);
  GraphNode* RegisterNode(const std::string& key, GraphNode* node);
  //! @}

  //! Retrieve a node.
  //! @{
  GraphNode* RetrieveNode(size_t key) const;
  GraphNode* RetrieveNode(const std::string& key) const;
  //! @}

  //! Get the start point of the graph (the nodes those has no inlinks).
  std::vector<const GraphNode*> start_points() const;
  std::vector<GraphNode*> start_points();

  //! Return the graph's nodes and edges(visited) in topological order.
  std::tuple<std::vector<GraphNode*>, std::vector<GraphEdge*>>
  topological_order() const;

  //! Return the graph's DFS order.
  std::vector<GraphNode*> dfs_order();

  //! Return the dependency nodes of a set of nodes.
  std::set<GraphNode*> dependencies(const std::vector<GraphNode*>& nodes);

  std::vector<const GraphNode*> nodes() const;
  std::vector<GraphNode*> nodes();

  //! Collect the nodes match the condition defined by \p teller in the graph.
  std::set<GraphNode*> CollectNodes(
      std::function<bool(const cinn::common::GraphNode*)>&& teller);

  void DropNode(GraphNode* n) {
    auto it = std::find_if(
        nodes_.begin(), nodes_.end(), [&](auto& x) { return x.get() == n; });
    if (it != nodes_.end()) {
      nodes_.erase(it);
    }
  }

  //! Get a string representation to visualize a graph.
  std::string Visualize() const;

  void ClearUnlinkedNodes(
      absl::flat_hash_map<std::string, std::vector<int>>* shape_dict,
      absl::flat_hash_map<std::string, cinn::common::Type>* type_dict,
      absl::flat_hash_map<std::string, std::string>* layout_dict);

  size_t num_nodes() const { return nodes_.size(); }

 protected:
  //! A lookup table that map from hash key to graph node, note that it doesn't
  //! own the graph node.
  std::map<size_t, GraphNode*> registry_;
  //! A list owns the graph nodes.
  std::vector<Shared<GraphNode>> nodes_;
};

}  // namespace common
}  // namespace cinn

namespace std {
template <>
struct hash<cinn::common::GraphNode> {
  size_t operator()(const cinn::common::GraphNode& x) {
    return reinterpret_cast<size_t>(hash<std::string>()(x.id()));
  }
};

}  // namespace std
