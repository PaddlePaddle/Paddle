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
    if (teller_) return teller_(node);

    for (auto& asrt : asserts_) {
      if (!asrt(node)) return false;
    }
    return true;
  }

  bool IsOp() const { return type_ == Type::kOp; }
  bool IsVar() const { return type_ == Type::kVar; }

  const std::string& name() const { return name_; }

  PDNode& operator=(const PDNode&) = delete;
  PDNode(const PDNode&) = delete;

  // Mark this node is an Input of a subgraph and will be retained.
  PDNode* AsInput() {
    role_ = Role::kInput;
    return this;
  }
  // Mark this node is an Output of a subgraph and will be retained.
  PDNode* AsOutput() {
    role_ = Role::kOutput;
    return this;
  }
  // Mark this node will be removed, so all the links should be inside a matched
  // sub-graph.
  PDNode* AsIntermediate() {
    role_ = Role::kIntermediate;
    return this;
  }

  bool IsIntermediate() const { return role_ == Role::kIntermediate; }
  bool IsInput() const { return role_ == Role::kInput; }
  bool IsOutput() const { return role_ == Role::kOutput; }

  // Assertions, helper functions to simplify the pattern definition.
  PDNode* assert_is_op();
  PDNode* assert_is_op(const std::string& op_type);
  PDNode* assert_is_var();
  PDNode* assert_var_not_persistable();
  PDNode* assert_is_persistable_var();
  PDNode* assert_is_op_output(const std::string& op_type);
  PDNode* assert_is_op_output(const std::string& op_type,
                              const std::string& argument);
  PDNode* assert_is_op_input(const std::string& op_type);
  PDNode* assert_is_op_input(const std::string& op_type,
                             const std::string& argument);
  PDNode* assert_is_op_nth_input(const std::string& op_type,
                                 const std::string& argument, int nth);
  PDNode* assert_is_op_nth_output(const std::string& op_type,
                                  const std::string& argument, int nth);
  PDNode* assert_is_only_input_of_op(const std::string& op_type);
  PDNode* assert_is_only_output_of_op(const std::string& op_type);
  PDNode* assert_op_has_n_inputs(const std::string& op_type, size_t n);
  PDNode* assert_op_has_n_outputs(const std::string& op_type, size_t n);
  PDNode* assert_more(teller_t&& teller);

 private:
  PDNode(PDPattern* pattern, const std::string& name = "",
         Type type = Type::kVar)
      : pattern_(pattern), name_(name), type_(type) {}
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

  // Will removed latter.
  teller_t teller_;
  std::vector<teller_t> asserts_;
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
 *     MUL = PDPattern.NewNode().assert_is_op("mul");
 *     ELE = PDPattern.NewNode().assert_is_op("elementwise_add");
 *     // Create the variable PDNodes.
 *     MUL_out = PDPattern.NewNode().assert_is_op_output("mul") \
 *                                  .assert_is_op_input("elementwise_add") \
 *                                  .AsIntermediate();
 *     // Add relations.
 *     MUL->LinksTo({MUL_out});
 *     MUL_out->LinksTo({ELE});
 *
 * One can add more specific asserts for PDNodes or edges, both the Operator
 * and Variable Nodes can be ruled in PDNode.assert_more(...).
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
  PDNode* NewNode(const std::string& name = NewID());
  PDNode* NewNode(const std::string& prefix, const std::string& name) {
    return NewNode(prefix + "/" + name);
  }
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
  // The intermediate PDNodes will be removed, so can't shared by multiple
  // patterns.
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

// Tell if a var links to an Op
bool VarLinksToOp(Node* node, const std::string& op_type);

// Tell if an op links to a var
bool VarLinksFromOp(Node* node, const std::string& op_type);

// Check whether a var node is a op node's nth input.
bool IsNthInput(Node* var, Node* op, const std::string& argument, size_t nth);

// Tell whether a var node is a op node's nth output.
bool IsNthOutput(Node* var, Node* op, const std::string& argument, size_t nth);

// Graph safely remove some nodes, will automatically clean up the edges.
void GraphSafeRemoveNodes(Graph* graph,
                          const std::unordered_set<const Node*>& nodes);

// Some pre-defined patterns those can be reused in multiple passes.
namespace patterns {

// FC with bias
// op: mul + elementwise_add
// named nodes:
// mul, elementwise_add
// w, mul_out, bias, fc_out
PDNode* FC(PDPattern* pattern, const std::string& name_scope, PDNode* x,
           bool with_bias);

PDNode* LSTM(PDPattern* pattern, const std::string& name_scope, PDNode* x);

}  // namespace patterns

#define IR_NODE_LINK_TO(a, b) \
  a->outputs.push_back(b);    \
  b->inputs.push_back(a);

}  // namespace ir
}  // namespace framework
}  // namespace paddle
