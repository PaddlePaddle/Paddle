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

#include <glog/logging.h>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "paddle/fluid/lite/core/mir/node.h"
#include "paddle/fluid/lite/core/mir/ssa_graph.h"
#include "paddle/fluid/lite/model_parser/pb/op_desc.h"
#include "paddle/fluid/lite/utils/string.h"

namespace paddle {
namespace lite {
namespace mir {
class PMPattern;

// Some basic terminologies:
//   - PMPattern: a pattern defined as a data flow graph.
//   - PMNode: the node in the pattern, each PMNode represents an `mir::Node`
//     that meets some conditions defined in `PMNode.teller`.
//   - A pattern is defined with PMNodes with edges.

// Pattern matcher node. This node helps to build a pattern.
struct PMNode {
  // tell whether an mir::Node* is a candidation for a PMNode.
  using teller_t = std::function<bool(const Node*)>;
  enum class Type { kOp, kVar };
  enum class Role {
    kUnknown,      // No role,
    kInput,        // an input and will be retained,
    kOutput,       // an output and will be retained,
    kIntermediate  // will be removed after handler.
  };

  // this link to others
  PMNode& LinksTo(const std::vector<PMNode*>& others);
  PMNode& LinksFrom(const std::vector<PMNode*>& others);

  // Link this to another node.
  PMNode& operator>>(PMNode& right);

  // Link many nodes to this node.
  friend PMNode& operator>>(std::vector<PMNode*>& others, PMNode& me);

  // Link this to many other nodes.
  PMNode& operator>>(std::vector<PMNode*>& nodes);

  bool Tell(const Node* node) const {
    if (teller_) return teller_(node);

    for (auto& asrt : asserts_) {
      if (!asrt(node)) return false;
    }
    return true;
  }

  bool IsOp() const { return type_ == Type::kOp; }
  bool IsVar() const { return type_ == Type::kVar; }

  const std::string& name() const { return name_; }

  PMNode& operator=(const PMNode&) = delete;
  PMNode(const PMNode&) = delete;

  // Mark this node is an Input of a subgraph and will be retained.
  PMNode* AsInput() {
    role_ = Role::kInput;
    return this;
  }
  // Mark this node is an Output of a subgraph and will be retained.
  PMNode* AsOutput() {
    role_ = Role::kOutput;
    return this;
  }
  // Mark this node will be removed, so all the links should be inside a matched
  // sub-graph.
  PMNode* AsIntermediate() {
    role_ = Role::kIntermediate;
    return this;
  }

  PMNode* AsVar() {
    type_ = Type::kVar;
    assert_is_var();
    return this;
  }

  PMNode* AsOp(const std::string& op_type) {
    type_ = Type::kOp;
    assert_is_op(op_type);
    return this;
  }

  void set_op_type(const std::string& op_type) { op_type_ = op_type; }

  bool IsIntermediate() const { return role_ == Role::kIntermediate; }
  bool IsInput() const { return role_ == Role::kInput; }
  bool IsOutput() const { return role_ == Role::kOutput; }

  // Assertions, helper functions to simplify the pattern definition.
  PMNode* assert_is_op();
  PMNode* assert_is_op(const std::string& op_type);
  PMNode* assert_is_var();
  PMNode* assert_var_not_persistable();
  PMNode* assert_is_persistable_var();
  PMNode* assert_is_op_output(const std::string& op_type);
  PMNode* assert_is_op_input(const std::string& op_type);
  PMNode* assert_is_op_input(const std::string& op_type,
                             const std::string& argument);
  PMNode* assert_is_op_output(const std::string& op_type,
                              const std::string& argument);

  PMNode* assert_is_op_nth_input(const std::string& op_type,
                                 const std::string& argument, int nth);
  PMNode* assert_is_op_nth_output(const std::string& op_type,
                                  const std::string& argument, int nth);

  template <typename T>
  PMNode* assert_op_attr(const std::string& attr_name, const T& attr) {
    asserts_.push_back([=](const Node* x) {
      if (x && x->IsStmt()) {
        auto* op_info = x->stmt()->op_info();
        return op_info->HasAttr(attr_name) &&
               op_info->GetAttr<T>(attr_name) == attr;
      }
      return false;
    });
    return this;
  }

 private:
  PMNode(PMPattern* pattern, const std::string& name = "",
         Type type = Type::kVar)
      : pattern_(pattern), name_(name), type_(type) {}
  PMNode(teller_t&& teller, PMPattern* pattern, const std::string& name = "",
         Type type = Type::kVar)
      : teller_(std::move(teller)),
        pattern_(pattern),
        name_(name),
        type_(type) {
    CHECK(teller_ != nullptr) << "invalid teller functer is set.";
  }

  PMNode(PMNode&& other) = default;

  friend class PMPattern;

  // Will removed latter.
  teller_t teller_;
  std::vector<teller_t> asserts_;
  PMPattern* pattern_;
  std::string name_;
  std::string op_type_;
  Type type_;
  Role role_{Role::kUnknown};
};

/*
 * A pattern in a graph, which defined with PMNode and edges. Most graph
 * patterns can be divided into PMNodes and link relations between them.
 *
 * For example, the FC fusion need to filter the MUL and ELEMENTWISE_ADD
 * operators from the computation graph, the MUL's output should have only one
 * consumer which is the ELEMENTWISE_ADD.
 * This pattern can be defined as with the following pseudo codes
 *
 *     // Create two operator PMNodes.
 *     MUL = PMPattern.NewNode().assert_is_op("mul");
 *     ELE = PMPattern.NewNode().assert_is_op("elementwise_add");
 *     // Create the variable PMNodes.
 *     MUL_out = PMPattern.NewNode().assert_is_op_output("mul") \
 *                                  .assert_is_op_input("elementwise_add") \
 *                                  .AsIntermediate();
 *     // Add relations.
 *     MUL->LinksTo({MUL_out});
 *     MUL_out->LinksTo({ELE});
 *
 * One can add more specific asserts for PMNodes or edges, both the Operator
 * and Variable Nodes can be ruled in PMNode.assert_more(...).
 *
 * PMPattern can record the general patterns, such as the pattern represents
 *   - Op in CPU -> Op in GPU -> Op in CPU, to findout the IO abnormal place.
 *   - Ops whose inputs and outputs share the same variables
 */
class PMPattern {
 public:
  using edge_t = std::pair<PMNode*, PMNode*>;

  void AddEdge(PMNode* a, PMNode* b);

  PMNode* NewNode(PMNode::teller_t&& teller, const std::string& name = NewID());
  PMNode* NewNode(const std::string& name = NewID());
  PMNode* NewNode(const std::string& prefix, const std::string& name) {
    return NewNode(prefix + "/" + name);
  }
  PMNode* RetrieveNode(const std::string& id) const;

  const std::vector<std::unique_ptr<PMNode>>& nodes() const { return nodes_; }
  const std::vector<edge_t>& edges() const { return edges_; }

  std::string DotString() const;

 private:
#ifdef PADDLE_WITH_TESTING
  FRIEND_TEST(PMPattern, AddEdge);
  FRIEND_TEST(PMPattern, NewNode);
#endif

  static std::string NewID() { return string_format("pmnode-%d", id_++); }

  std::vector<std::unique_ptr<PMNode>> nodes_;
  std::vector<edge_t> edges_;
  std::unordered_map<std::string, PMNode*> node_map_;
  static size_t id_;
};

/*
 * PatternMatcher helps to detect the specific patterns in the graph.
 * Input a pattern, output a list of the matched subgraphs/nodes.
 * This helper can be used to support fuse(conv+batchnorm => batchnorm e.g.).
 *
 * The algorithm has three phases:
 *   1. Mark the nodes that match the defined PMNodes in a PMPattern,
 *   2. Extend a PMNode to subgraphs by deducing the connection relation defined
 *      in PAPattern(the edges),
 *   3. Get the filtered subgraphs and treat them with a pre-defined handler.
 *
 * Usage:
 *    // Create a matcher
 *    PatternMatcher matcher;
 *    // Define the matcher's pattern, by adding PMNode and define the edges.
 *    auto* node0 = matcher.mutable_pattern().AddNode(...)
 *    auto* node1 = matcher.mutable_pattern().AddNode(...)
 *    node0->teller = some lambda.
 *    node1->teller = some lambda.
 *    matcher.mutable_pattern().AddEdge(node0, node1);
 *    // Create an handler, to define the behavior of treating the filtered
 *    // subgraphs that comply with the patterns.
 *    PatternMatcher::handle_t handler = some labmda
 *    // Execute the matcher.
 *    matcher(&graph, handler);
 */
class PatternMatcher {
 public:
  using subgraph_t = std::unordered_map<PMNode*, Node*>;

  // Operate on the detected pattern.
  using handle_t =
      std::function<void(const subgraph_t& /*hitted pattern*/, SSAGraph*)>;

  void operator()(SSAGraph* graph, handle_t handler);

  const PMPattern& pattern() const { return pattern_; }
  PMPattern* mutable_pattern() { return &pattern_; }

 private:
  // Mark the nodes that fits the pattern.
  bool MarkPMNodesInGraph(SSAGraph* graph);

  // Detect all the pattern and output the hit records.
  std::vector<subgraph_t> DetectPatterns();

  // Remove duplicate patterns.
  void UniquePatterns(std::vector<subgraph_t>* subgraphs);

  // Remove overlapped match subgraphs, when overlapped, keep the previous one.
  // The intermediate PMNodes will be removed, so can't shared by multiple
  // patterns.
  void RemoveOverlappedMatch(std::vector<subgraph_t>* subgraphs);

  // Validate whether the intermediate nodes are linked by external nodes.
  void ValidateByNodeRole(std::vector<subgraph_t>* subgraphs);

#ifdef PADDLE_WITH_TESTING
  FRIEND_TEST(PatternMatcher, MarkPMNodesInGraph);
  FRIEND_TEST(PatternMatcher, DetectPatterns);
#endif

 private:
  using hit_rcd_t =
      std::pair<Node* /*node in graph*/, PMNode* /*node in pattern*/>;
  PMPattern pattern_;
  std::unordered_map<const PMNode*, std::unordered_set<Node*>> pmnodes2nodes_;
};

// Check whether a var node is a op node's nth input.
bool IsNthInput(const Node& var, const Node& op, const std::string& argument,
                int nth);

// Check whether the op node has input of given name.
bool HasInput(const Node& op, const std::string& argument);

// Graph safely remove some nodes, will automatically clean up the edges.
void GraphSafeRemoveNodes(SSAGraph* graph,
                          const std::unordered_set<const Node*>& nodes);

// Some pre-defined patterns those can be reused in multiple passes.
// The related Fluid Layer or Op should be one pattern here for better re-usage
// across different fusion.
namespace patterns {

struct KeyCounter {
  static KeyCounter& Instance() {
    static KeyCounter x;
    return x;
  }

  int IncCounter(const std::string& key) { return dic_[key]++; }

 private:
  std::unordered_map<std::string, size_t> dic_;
};

// Generate a unique PMNode's name with name_scope and id.
// The format is {name_scope}/{repr}/{id}/{name}
static std::string PMNodeName(const std::string& name_scope,
                              const std::string& repr, size_t id,
                              const std::string& name) {
  std::stringstream ss;
  ss << name_scope << "/" << repr << "/" << id << "/" << name;
  return ss.str();
}
// Generate a unique PMNode's name.
// The format is {name_scope}/{repr}/{id}
static std::string PMNodeName(const std::string& name_scope,
                              const std::string& repr) {
  std::stringstream ss;
  ss << name_scope << "/" << repr << "/"
     << KeyCounter::Instance().IncCounter(repr);
  return ss.str();
}
// Generate a unique key. It can be used for a universally unique temporary
// name.
// The format is {repr}/{id}
static std::string UniqueKey(const std::string& repr) {
  std::stringstream ss;
  ss << repr << "/" << KeyCounter::Instance().IncCounter(repr);
  return ss.str();
}

// Declare a PMNode in a pattern, will create two methods:
// std::string xxx_repr(); return this PMNode's string id.
// PMNode* xxx_n(); return the corresponding PMNode.
#define PATTERN_DECL_NODE(name__)                        \
  std::string name__##_repr() const {                    \
    return PMNodeName(name_scope_, repr_, id_, #name__); \
  }                                                      \
  PMNode* name__##_n() const { return pattern->RetrieveNode(name__##_repr()); }

// Get an mir::Node* from the matched subgraph.
// var: variable.
// arg: the argument declared by PATTERN_DECL_NODE in a pattern definition.
// pat: the pattern object.
#define GET_IR_NODE_FROM_SUBGRAPH(var, arg, pat)        \
  CHECK(subgraph.count(pat.arg##_n()))                  \
      << "Node not found for PMNode " pat.arg##_repr(); \
  Node* var = subgraph.at(pat.arg##_n());               \
  CHECK(var) << "node " << #arg << "not exists in the sub-graph"

// The base class of all the patterns.
struct PatternBase {
  PatternBase(PMPattern* pattern, const std::string& name_scope,
              const std::string& repr)
      : pattern(pattern),
        name_scope_(name_scope),
        repr_(repr),
        id_(KeyCounter::Instance().IncCounter(repr)) {}

  PMPattern* pattern;

 protected:
  std::string name_scope_;
  std::string repr_;
  size_t id_;
};

}  // namespace patterns

// Link two mir::Nodes from each other.
#define IR_NODE_LINK_TO(a, b) \
  a->outlinks.push_back(b);   \
  b->inlinks.push_back(a);

// Set the out_var as the output of the op
#define IR_OP_VAR_LINK(op, out_var) \
  op->outlinks.push_back(out_var);  \
  out_var->inlinks.clear();         \
  out_var->inlinks.push_back(op);

}  // namespace mir
}  // namespace lite
}  // namespace paddle
