// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"

namespace paddle {
namespace framework {
namespace ir {

/*
 * PDNode2 is a light weight helper for PDNode. It can be copied or moved.
 * It is designed to make PDNode operation easier.
 */
class PDNode2 {
 public:
  PDNode2(PDNode2&& other)
      : node_(other.node_),
        pattern_(other.pattern_),
        op_type_(std::move(other.op_type_)) {}
  PDNode2(const PDNode2& other)
      : node_(other.node_),
        pattern_(other.pattern_),
        op_type_(other.op_type_) {}

  PDNode2(PDPattern* pattern, const std::string& key) : pattern_(pattern) {
    node_ = pattern_->NewNode(patterns::UniqueKey(key));
  }

  PDNode2(PDPattern* pattern, const std::string& key, PDNode::teller_t&& teller)
      : pattern_(pattern) {
    node_ = pattern_->NewNode(std::move(teller), patterns::UniqueKey(key));
  }

  void SetOpType(const std::string& op_type) { op_type_ = op_type; }

  // Link this to another node.
  const PDNode2& operator>>(const PDNode2& other) const;

  // Link many nodes to this node.
  friend const PDNode2& operator>>(const std::vector<PDNode2>& others,
                                   const PDNode2& me);

  // Link this to many other nodes.
  const PDNode2& operator>>(const std::vector<PDNode2>& nodes) const;

  PDNode& pd_node() const { return *node_; }

 private:
  PDNode* node_{nullptr};
  mutable PDPattern* pattern_{nullptr};
  std::string op_type_;
};

class FuseBase {
 public:
  using key2nodes_t = std::map<std::string, ir::Node*>;

  virtual ~FuseBase() = default;

  void operator()(Graph* graph) {
    BuildPattern();
    PerformPatternDetector(graph);

    for (const auto& matched : key2nodes_) {
      InsertNewNode(graph, matched);
    }

    DeleteInterNodes(graph);
  }

  // Build a PDPattern using PDNode2.
  virtual void BuildPattern() = 0;

  // Generate an operator desc with a matched subgraph.
  virtual OpDesc GenOpDesc(const key2nodes_t& matched) = 0;

  PDNode2& OpNode(const std::string& key, const std::string& op_type);

  PDNode2& VarNode(const std::string& key);

 protected:
  virtual void InsertNewNode(ir::Graph* graph, const key2nodes_t& matched) = 0;

 private:
  void PerformPatternDetector(Graph* graph);

  // Delete nodes that are marked as Intermediate
  void DeleteInterNodes(ir::Graph* graph);

 private:
  PDNode2& Node(const std::string& key);

 protected:
  GraphPatternDetector detector_;
  std::map<std::string, PDNode2> nodes_;
  std::vector<key2nodes_t> key2nodes_;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
