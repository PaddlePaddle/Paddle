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

#ifdef PADDLE_WITH_TESTING
#include <gtest/gtest_prod.h>
#endif

#include <numeric>
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/inference/analysis/dot.h"

namespace paddle {
namespace framework {
namespace ir {
class PDPattern;

// Some basic terminologies:
//   - PDPattern: a pattern defined as a data flow graph.
//   - PDNode: the node in the pattern, each PDNode represents an `ir::Node`
//     that meets some conditions defined in `PDNode.teller`.
//   - A pattern is defined with PDNodes with edges.

// Pattern detector node. This node helps to build a pattern.
struct PDNode {
  // tell whether an ir::Node* is a candidation for a PDNode.
  using teller_t = std::function<bool(Node*)>;
  enum class Type { kOp, kVar };
  enum class Role {
    kUnknown,      // No role,
    kInput,        // an input and will be retained,
    kOutput,       // an output and will be retained,
    kIntermediate  // will be removed after handler.
  };

  // this link to others
  PDNode& LinksTo(const std::vector<PDNode*>& others);
  PDNode& LinksFrom(const std::vector<PDNode*>& others);

  bool Tell(Node* node) const {
    PADDLE_ENFORCE(teller_ != nullptr, "teller should be set for a PDNode");
    return teller_(node);
  }

  bool IsOp() const { return type_ == Type::kOp; }
  bool IsVar() const { return type_ == Type::kVar; }

  const std::string& name() const { return name_; }

  PDNode(const PDNode&) = delete;
  PDNode& operator=(const PDNode&) = delete;

  // Mark this node is an Input of a subgraph and will be retained.
  PDNode& AsInput() {
    role_ = Role::kInput;
    return *this;
  }
  // Mark this node is an Output of a subgraph and will be retained.
  PDNode& AsOutput() {
    role_ = Role::kOutput;
    return *this;
  }
  // Mark this node will be removed, so all the links should be inside a matched
  // sub-graph.
  PDNode& AsIntermediate() {
    role_ = Role::kIntermediate;
    return *this;
  }

  bool IsIntermediate() const { return role_ == Role::kIntermediate; }
  bool IsInput() const { return role_ == Role::kInput; }
  bool IsOutput() const { return role_ == Role::kOutput; }

 private:
  PDNode(teller_t&& teller, PDPattern* pattern, const std::string& name = "",
         Type type = Type::kVar)
      : teller_(std::move(teller)),
        pattern_(pattern),
        name_(name),
        type_(type) {
    PADDLE_ENFORCE(teller_ != nullptr, "invalid teller functer is set.");
  }

  PDNode(PDNode&& other) = default;

  friend class PDPattern;

  teller_t teller_;
  PDPattern* pattern_;
  std::string name_;
  Type type_;
  Role role_{Role::kUnknown};
};

/*
 * A pattern in a graph, which defined with PDNode and edges. Most graph
 * patterns can be divided into PDNodes and link relations between them.
 *
 * For example, the FC fusion need to filter the MUL and ELEMENTWISE_ADD
 * operators from the computation graph, the MUL's output should have only one
 * consumer which is the ELEMENTWISE_ADD.
 * This pattern can be defined as with the following pseudo codes
 *
 *     // Create two operator PDNodes.
 *     MUL = PDPattern.NewNode()
 *     ELE = PDPattern.NewNode()
 *     // Create the variable PDNodes.
 *     MUL_out = PDPattern.NewNode()
 *     // Add teller to define some rules that help to filter the target Nodes.
 *     MUL.teller = lambda(node): node->IsOp() && node->Op()->Type == "mul";
 *     ELE.teller = lambda(node): \
 *                        node->IsOp() && node->Op()->Type == "elementwise_add";
 *     MUL_out.teller = lambda(node): node->IsVar() && (MUL in node->inputs)
 *                                                  && (ELE in node->outputs)
 *
 * One can add more specific tellers for PDNodes or edges, both the Operator
 * and Variable Nodes can be ruled in PDNode.teller.
 *
 * PDPattern can record the general patterns, such as the pattern represents
 *   - Op in CPU -> Op in GPU -> Op in CPU, to findout the IO abnormal place.
 *   - Ops whose inputs and outputs share the same variables
 */
class PDPattern {
 public:
  using edge_t = std::pair<PDNode*, PDNode*>;

  void AddEdge(PDNode* a, PDNode* b);

  PDNode* NewNode(PDNode::teller_t&& teller, const std::string& name = NewID());
  PDNode* RetrieveNode(const std::string& id) const;

  const std::vector<std::unique_ptr<PDNode>>& nodes() const { return nodes_; }
  const std::vector<edge_t>& edges() const { return edges_; }

  std::string DotString() const;

 private:
#ifdef PADDLE_WITH_TESTING
  FRIEND_TEST(PDPattern, AddEdge);
  FRIEND_TEST(PDPattern, NewNode);
#endif

  static std::string NewID() { return "pdnode-" + std::to_string(id_++); }

  std::vector<std::unique_ptr<PDNode>> nodes_;
  std::vector<edge_t> edges_;
  std::unordered_map<std::string, PDNode*> node_map_;
  static size_t id_;
};

/*
 * GraphPatternDetector helps to detect the specific patterns in the graph.
 * Input a pattern, output a list of the matched subgraphs/nodes.
 * This helper can be used to support fuse(conv+batchnorm => batchnorm e.g.).
 *
 * The algorithm has three phases:
 *   1. Mark the nodes that match the defined PDNodes in a PDPattern,
 *   2. Extend a PDNode to subgraphs by deducing the connection relation defined
 *      in PAPattern(the edges),
 *   3. Get the filtered subgraphs and treat them with a pre-defined handler.
 *
 * Usage:
 *    // Create a detector
 *    GraphPatternDetector detector;
 *    // Define the detector's pattern, by adding PDNode and define the edges.
 *    auto* node0 = detector.mutable_pattern().AddNode(...)
 *    auto* node1 = detector.mutable_pattern().AddNode(...)
 *    node0->teller = some lambda.
 *    node1->teller = some lambda.
 *    detector.mutable_pattern().AddEdge(node0, node1);
 *    // Create an handler, to define the behavior of treating the filtered
 *    // subgraphs that comply with the patterns.
 *    GraphPatternDetector::handle_t handler = some labmda
 *    // Execute the detector.
 *    detector(&graph, handler);
 */
class GraphPatternDetector {
 public:
  using subgraph_t = std::unordered_map<PDNode*, Node*>;

  // Operate on the detected pattern.
  using handle_t =
      std::function<void(const subgraph_t& /*hitted pattern*/, Graph*)>;

  void operator()(Graph* graph, handle_t handler);

  const PDPattern& pattern() const { return pattern_; }
  PDPattern* mutable_pattern() { return &pattern_; }

 private:
  // Mark the nodes that fits the pattern.
  bool MarkPDNodesInGraph(const ir::Graph& graph);

  // Detect all the pattern and output the hit records.
  std::vector<subgraph_t> DetectPatterns();

  // Remove duplicate patterns.
  void UniquePatterns(std::vector<subgraph_t>* subgraphs);

  // Remove overlapped match subgraphs, when overlapped, keep the previous one.
  void RemoveOverlappedMatch(std::vector<subgraph_t>* subgraphs);

  // Validate whether the intermediate nodes are linked by external nodes.
  void ValidateByNodeRole(std::vector<subgraph_t>* subgraphs);

#ifdef PADDLE_WITH_TESTING
  FRIEND_TEST(GraphPatternDetecter, MarkPDNodesInGraph);
  FRIEND_TEST(GraphPatternDetecter, DetectPatterns);
#endif

 private:
  using hit_rcd_t =
      std::pair<Node* /*node in graph*/, PDNode* /*node in pattern*/>;
  PDPattern pattern_;
  std::unordered_map<const PDNode*, std::unordered_set<Node*>> pdnodes2nodes_;
};

// some helper methods.

// Op's input.
static bool VarLinksToOp(Node* node, const std::string& op_type) {
  for (auto* out : node->outputs) {
    if (out->IsOp() && out->Op()->Type() == op_type) {
      return true;
    }
  }
  return false;
}

// Op's output.
static bool VarLinksFromOp(Node* node, const std::string& op_type) {
  for (auto* out : node->inputs) {
    if (out->IsOp() && out->Op()->Type() == op_type) {
      return true;
    }
  }
  return false;
}

// Check whether a var node is a op node's nth input.
static bool IsNthInput(Node* var, Node* op, const std::string& argument,
                       size_t nth) {
  PADDLE_ENFORCE(var->IsVar());
  PADDLE_ENFORCE(op->IsOp());
  if (op->inputs.size() <= nth) return false;
  return var->Name() == op->Op()->Input(argument)[nth];
}

static void GraphSafeRemoveNodes(Graph* graph,
                                 const std::unordered_set<const Node*>& nodes) {
  for (auto* node : nodes) {
    graph->RemoveNode(const_cast<Node*>(node));
  }

  for (auto* node : graph->Nodes()) {
    for (auto it = node->inputs.begin(); it != node->inputs.end();) {
      if (nodes.count(*it)) {
        it = const_cast<Node*>(node)->inputs.erase(it);
      } else
        it++;
    }
    for (auto it = node->outputs.begin(); it != node->outputs.end();) {
      if (nodes.count(*it)) {
        it = const_cast<Node*>(node)->outputs.erase(it);
      } else
        it++;
    }
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle
