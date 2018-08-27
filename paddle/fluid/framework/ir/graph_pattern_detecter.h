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

namespace paddle {
namespace framework {
namespace ir {

// Some basic torminolygies:
//   - PDPattern: a pattern defined as a data flow graph.
//   - PDNode: the node in the pattern, each PDNode represents an `ir::Node`
//     that meets some conditions defined in `PDNode.teller`.
//   - A pattern is defined with PDNodes with edges.

// Pattern detector node. This node helps to build a pattern.
struct PDNode {
  // tell whether an ir::Node* is a candidation for a PDNode.
  using teller_t = std::function<bool(Node*)>;

  PDNode(teller_t&& teller, const std::string& name = "")
      : teller_(teller), name_(name) {
    PADDLE_ENFORCE(teller_ != nullptr, "invalid teller functer is set.");
  }

  PDNode(PDNode&& other) = default;

  std::vector<PDNode*> inlinks;
  std::vector<PDNode*> outlinks;

  bool Tell(Node* node) const {
    PADDLE_ENFORCE(teller_ != nullptr, "teller should be set for a PDNode");
    return teller_(node);
  }

  const std::string& name() const { return name_; }

  PDNode(const PDNode&) = delete;
  PDNode& operator=(const PDNode&) = delete;

 private:
  teller_t teller_;
  std::string name_;
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
  PDNode* RetriveNode(const std::string& id) const;

  const std::vector<std::unique_ptr<PDNode>>& nodes() const { return nodes_; }
  const std::vector<edge_t>& edges() const { return edges_; }

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
 * GraphPatternDetecter helps to detect the specific patterns in the graph.
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
 *    GraphPatternDetecter detector;
 *    // Define the detector's pattern, by adding PDNode and define the edges.
 *    auto* node0 = detector.mutable_pattern().AddNode(...)
 *    auto* node1 = detector.mutable_pattern().AddNode(...)
 *    node0->teller = some lambda.
 *    node1->teller = some lambda.
 *    detector.mutable_pattern().AddEdge(node0, node1);
 *    // Create an handler, to define the behavior of treating the filtered
 *    // subgraphs that comply with the patterns.
 *    GraphPatternDetecter::handle_t handler = some labmda
 *    // Execute the detector.
 *    detector(&graph, handler);
 */
class GraphPatternDetecter {
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

#ifdef PADDLE_WITH_TESTING
  FRIEND_TEST(GraphPatternDetecter, MarkPDNodesInGraph);
  FRIEND_TEST(GraphPatternDetecter, DetectPatterns);
#endif

 private:
  using hit_rcd_t =
      std::pair<Node* /*node in graph*/, PDNode* /*node in pattern*/>;
  PDPattern pattern_;
  std::vector<hit_rcd_t> marked_records_;
  std::unordered_map<const PDNode*, std::unordered_set<Node*>> pdnodes2nodes_;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
