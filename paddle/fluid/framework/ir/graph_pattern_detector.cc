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

#include <array>
#include <string>
#include <vector>

#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/graph_traits.h"
#include "paddle/fluid/framework/ir/graph_viz_pass.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/string/pretty_log.h"
#include "paddle/fluid/string/printf.h"
namespace paddle {
namespace framework {
namespace ir {

using string::PrettyLogEndl;
using string::PrettyLog;
using string::Style;

size_t PDPattern::id_ = 0UL;

PDNode *PDPattern::NewNode(const std::string &name) {
  if (!name.empty()) {
    PADDLE_ENFORCE_EQ(node_map_.count(name), 0,
                      "PDNode's name should be unique, get duplicate [%s]",
                      name);
  }

  nodes_.emplace_back(new PDNode(this, name));
  auto *cur = nodes_.back().get();
  node_map_[name] = cur;
  return cur;
}

PDNode *PDPattern::NewNode(PDNode::teller_t &&teller, const std::string &name) {
  if (!name.empty()) {
    PADDLE_ENFORCE_EQ(node_map_.count(name), 0,
                      "PDNode's name should be unique, get duplicate [%s]",
                      name);
  }

  nodes_.emplace_back(new PDNode(std::move(teller), this, name));
  auto *cur = nodes_.back().get();
  node_map_[name] = cur;
  return cur;
}

PDNode *PDPattern::RetrieveNode(const std::string &id) const {
  auto it = node_map_.find(id);
  if (it == node_map_.end()) {
    return nullptr;
  }

  return it->second;
}

void PDPattern::AddEdge(PDNode *a, PDNode *b) {
  PADDLE_ENFORCE(a);
  PADDLE_ENFORCE(b);
  PADDLE_ENFORCE(a != b, "can't connect to the same nodes.");
  edges_.emplace_back(a, b);
}

void GraphPatternDetector::operator()(Graph *graph,
                                      GraphPatternDetector::handle_t handler) {
  if (!MarkPDNodesInGraph(*graph)) {
    return;
  }

  auto subgraphs = DetectPatterns();
  UniquePatterns(&subgraphs);
  RemoveOverlappedMatch(&subgraphs);
  ValidateByNodeRole(&subgraphs);

  if (subgraphs.empty()) return;
  PrettyLogEndl(Style::detail(), "---  detect %d subgraphs", subgraphs.size());
  int id = 0;
  for (auto &g : subgraphs) {
    VLOG(3) << "optimizing #" << id++ << " subgraph";
    handler(g, graph);
  }
}

bool GraphPatternDetector::MarkPDNodesInGraph(const ir::Graph &graph) {
  VLOG(3) << "mark pdnodes in graph";
  if (graph.Nodes().empty()) return false;

  for (auto &node : GraphTraits::DFS(graph)) {
    for (const auto &pdnode : pattern_.nodes()) {
      if (pdnode->Tell(&node)) {
        VLOG(4) << "pdnode " << pdnode->name() << " marked";
        pdnodes2nodes_[pdnode.get()].insert(&node);
      }
    }
  }
  // Check to early stop if some PDNode can't find matched Node.
  for (auto &pdnode : pattern_.nodes()) {
    if (!pdnodes2nodes_.count(pdnode.get())) {
      VLOG(4) << pdnode->name() << " can't find matched Node, early stop";
      // return false;
    }
  }
  for (auto &item : pdnodes2nodes_) {
    for (auto &n : item.second) {
      GetMarkedNodes(const_cast<Graph *>(&graph)).insert(n);
    }
  }
  VLOG(3) << pdnodes2nodes_.size() << " nodes marked";

  return !pdnodes2nodes_.empty();
}

// The intermediate Nodes can only link to the nodes inside the pattern, or this
// subgraph will be droped.
void GraphPatternDetector::ValidateByNodeRole(
    std::vector<GraphPatternDetector::subgraph_t> *subgraphs) {
  std::vector<GraphPatternDetector::subgraph_t> result;

  subgraphs->erase(
      std::remove_if(
          subgraphs->begin(), subgraphs->end(),
          [](const GraphPatternDetector::subgraph_t &subgraph) -> bool {
            // Collect the inputs and outputs.
            std::unordered_set<Node *> ios;
            for (auto &item : subgraph) {
              if (!item.first->IsIntermediate()) {
                ios.insert(item.second);
              }
            }
            for (auto &item : subgraph) {
              if (item.first->IsIntermediate()) {
                for (auto *x : item.second->inputs) {
                  if (!ios.count(x)) {
                    return true;
                  }
                }
                for (auto *x : item.second->outputs) {
                  if (!ios.count(x)) {
                    return true;
                  }
                }
              }
            }
            return false;
          }),
      subgraphs->end());
}

struct HitGroup {
  std::unordered_map<PDNode *, Node *> roles;

  bool Match(Node *node, PDNode *pat) {
    if (nodes_.count(node)) {
      if (!roles.count(pat)) return false;
      return roles[pat] == node;
    }
    return !roles.count(pat) || roles.at(pat) == node;
  }

  void Register(Node *node, PDNode *pat) {
    roles[pat] = node;
    nodes_.insert(node);
  }

 private:
  std::unordered_set<Node *> nodes_;
};

// Tell whether Node a links to b.
bool IsNodesLink(Node *a, Node *b) {
  for (auto *node : a->outputs) {
    if (b == node) {
      return true;
    }
  }
  return false;
}

std::vector<GraphPatternDetector::subgraph_t>
GraphPatternDetector::DetectPatterns() {
  // Init empty subgraphs.
  std::vector<GraphPatternDetector::subgraph_t> result;
  std::vector<HitGroup> init_groups;
  std::array<std::vector<HitGroup>, 2> bi_records;
  // PADDLE_ENFORCE(!pattern_.edges().empty(), "At least one edge is needed");
  auto *first_pnode = pattern_.edges().empty() ? pattern().nodes().front().get()
                                               : pattern_.edges().front().first;
  if (!pdnodes2nodes_.count(first_pnode)) return result;
  for (auto *node : pdnodes2nodes_[first_pnode]) {
    HitGroup group;
    group.roles[first_pnode] = node;
    init_groups.emplace_back(group);
  }

  int step = 0;
  bi_records[0] = std::move(init_groups);

  // Extend a PDNode to subgraphs by deducing the connection relations defined
  // in edges of PDNodes.
  for (const auto &edge : pattern_.edges()) {
    VLOG(4) << "check " << edge.first->name() << " -> " << edge.second->name();
    // TODO(Superjomn) Fix bug here, the groups might be duplicate here.
    // Each role has two PDNodes, which indicates two roles.
    // Detect two Nodes that can match these two roles and they are connected.
    auto &pre_groups = bi_records[step % 2];
    auto &cur_groups = bi_records[1 - (step++ % 2)];
    cur_groups.clear();
    if (pre_groups.empty()) break;
    // source -> target
    for (Node *source : pdnodes2nodes_[edge.first]) {
      for (Node *target : pdnodes2nodes_[edge.second]) {
        VLOG(8) << "check " << source->id() << " -- " << target->id();
        // TODO(Superjomn) add some prune strategies.
        for (const auto &group : pre_groups) {
          HitGroup new_group = group;
          if (IsNodesLink(source, target) &&
              new_group.Match(source, edge.first)) {
            new_group.Register(source, edge.first);
            if (new_group.Match(target, edge.second)) {
              new_group.Register(target, edge.second);
              cur_groups.push_back(new_group);
              // TODO(Superjomn) need to unique
            }
          }
        }
      }
    }
    VLOG(3) << "step " << step << " get records: " << cur_groups.size();
    for (auto &group : cur_groups) {
      for (auto &item : group.roles) {
        VLOG(4) << "node " << item.second->id() << " as " << item.first->name();
      }
      VLOG(4) << "=========================================================";
    }
  }

  for (auto &group : bi_records[step % 2]) {
    GraphPatternDetector::subgraph_t subgraph;
    for (auto &role : group.roles) {
      subgraph.emplace(role.first, role.second);
    }
    result.emplace_back(subgraph);
  }
  return result;
}

void GraphPatternDetector::UniquePatterns(
    std::vector<GraphPatternDetector::subgraph_t> *subgraphs) {
  if (subgraphs->empty()) return;
  std::vector<GraphPatternDetector::subgraph_t> result;

  std::unordered_set<size_t> set;
  for (auto &g : *subgraphs) {
    size_t key = 0;
    for (auto &item : g) {
      key ^= std::hash<void *>{}(item.first);
      key ^= std::hash<void *>{}(item.second);
    }
    if (!set.count(key)) {
      result.emplace_back(g);
      set.insert(key);
    }
  }
  *subgraphs = result;
}

void GraphPatternDetector::RemoveOverlappedMatch(
    std::vector<subgraph_t> *subgraphs) {
  std::vector<subgraph_t> result;
  std::unordered_set<Node *> node_set;

  for (const auto &subgraph : *subgraphs) {
    bool valid = true;
    for (auto &item : subgraph) {
      if (item.first->IsIntermediate() && node_set.count(item.second)) {
        valid = false;
        break;
      }
    }
    if (valid) {
      for (auto &item : subgraph) {
        node_set.insert(item.second);
      }
      result.push_back(subgraph);
    }
  }
  *subgraphs = result;
}

std::string PDPattern::DotString() const {
  using inference::analysis::Dot;
  Dot dot;
  int id = 0;
  // Create Nodes
  std::unordered_map<PDNode *, std::string> node2dot;
  for (const auto &node : nodes()) {
    std::string node_id = "Node" + std::to_string(id++);
    dot.AddNode(node_id, {}, node->name());
    node2dot[node.get()] = node_id;
  }
  // Create Edges
  for (const auto &edge : edges()) {
    if (!node2dot.count(edge.first) || !node2dot.count(edge.second)) {
      LOG(ERROR) << "no node " << edge.first << " " << edge.second;
      continue;
    }
    auto &src = node2dot.at(edge.first);
    auto &trg = node2dot.at(edge.second);
    dot.AddEdge(src, trg, {});
  }
  return dot.Build();
}

PDNode &PDNode::LinksTo(const std::vector<PDNode *> &others) {
  // extend outlinks.
  for (PDNode *x : others) {
    pattern_->AddEdge(this, x);
  }
  return *this;
}

PDNode &PDNode::LinksFrom(const std::vector<PDNode *> &others) {
  // extend outlinks.
  for (PDNode *x : others) {
    pattern_->AddEdge(x, this);
  }
  return *this;
}

PDNode *PDNode::assert_is_op() {
  asserts_.emplace_back([](Node *x) { return x && x->IsOp(); });
  return this;
}

PDNode *PDNode::assert_is_op(const std::string &op_type) {
  asserts_.emplace_back([op_type](Node *x) {
    return x && x->IsOp() && x->Op()->Type() == op_type;
  });
  return this;
}

PDNode *PDNode::assert_is_var() {
  asserts_.emplace_back([](Node *x) { return x && x->IsVar(); });
  return this;
}

PDNode *PDNode::assert_is_not_ctrl_var() {
  asserts_.emplace_back([](Node *x) { return x && !x->IsCtrlVar(); });
  return this;
}

PDNode *PDNode::assert_var_not_persistable() {
  assert_is_var();
  asserts_.emplace_back([](Node *x) { return !x->Var()->Persistable(); });
  return this;
}

PDNode *PDNode::assert_is_persistable_var() {
  assert_is_var();
  asserts_.emplace_back([=](Node *x) { return x->Var()->Persistable(); });
  return this;
}

PDNode *PDNode::assert_is_op_nth_input(const std::string &op_type,
                                       const std::string &argument, int nth) {
  assert_is_var();
  assert_is_op_input(op_type);
  asserts_.emplace_back([=](Node *x) {
    for (auto *op : x->outputs) {
      if (op->IsOp() && op->Op()->Type() == op_type &&
          IsNthInput(x, op, argument, nth))
        return true;
    }
    return false;
  });
  return this;
}

PDNode *PDNode::assert_is_op_nth_output(const std::string &op_type,
                                        const std::string &argument, int nth) {
  assert_is_var();
  asserts_.emplace_back([=](Node *x) {
    for (auto *op : x->inputs) {
      if (op->IsOp() && op->Op()->Type() == op_type &&
          IsNthOutput(x, op, argument, nth))
        return true;
    }
    return false;
  });
  return this;
}

PDNode *PDNode::assert_is_only_input_of_op(const std::string &op_type) {
  assert_is_var();
  asserts_.emplace_back([=](Node *x) {
    for (auto *op : x->outputs) {
      if (op && op->IsOp() && op->Op() && op->Op()->Type() == op_type &&
          op->inputs.size() == 1) {
        return true;
      }
    }
    return false;
  });
  return this;
}

PDNode *PDNode::assert_is_only_output_of_op(const std::string &op_type) {
  assert_is_var();
  asserts_.emplace_back([=](Node *x) {
    for (auto *op : x->inputs) {
      if (op && op->IsOp() && op->Op() && op->Op()->Type() == op_type &&
          op->outputs.size() == 1) {
        return true;
      }
    }
    return false;
  });
  return this;
}

PDNode *PDNode::assert_is_op_output(const std::string &op_type) {
  assert_is_var();
  asserts_.emplace_back([=](Node *x) {
    for (auto *op : x->inputs) {
      if (op && op->IsOp() && op->Op() && op->Op()->Type() == op_type) {
        return true;
      }
    }
    return false;
  });
  return this;
}

PDNode *PDNode::assert_is_op_output(const std::string &op_type,
                                    const std::string &argument) {
  assert_is_var();
  assert_is_op_nth_output(op_type, argument, 0);
  return this;
}
PDNode *PDNode::assert_is_op_input(const std::string &op_type) {
  assert_is_var();
  asserts_.emplace_back([=](Node *x) {
    for (auto *op : x->outputs) {
      if (op && op->IsOp() && op->Op() && op->Op()->Type() == op_type) {
        return true;
      }
    }
    return false;
  });
  return this;
}

PDNode *PDNode::assert_is_op_input(const std::string &op_type,
                                   const std::string &argument) {
  assert_is_var();
  assert_is_op_nth_input(op_type, argument, 0);
  return this;
}

PDNode *PDNode::assert_op_has_n_inputs(const std::string &op_type, size_t n) {
  assert_is_op(op_type);
  asserts_.emplace_back([=](Node *x) { return x->inputs.size() == n; });
  return this;
}

PDNode *PDNode::assert_op_has_n_outputs(const std::string &op_type, size_t n) {
  assert_is_op(op_type);
  asserts_.emplace_back([=](Node *x) { return x->outputs.size() == n; });
  return this;
}

PDNode *PDNode::assert_more(PDNode::teller_t &&teller) {
  asserts_.emplace_back(std::move(teller));
  return this;
}

PDNode *PDNode::assert_is_ops(const std::unordered_set<std::string> &op_types) {
  asserts_.emplace_back([op_types](Node *x) {
    return x && x->IsOp() && op_types.count(x->Op()->Type());
  });
  return this;
}

PDNode *PDNode::assert_is_ops_nth_input(
    const std::unordered_set<std::string> &op_types,
    const std::string &argument, int nth) {
  assert_is_var();
  assert_is_ops_input(op_types);
  asserts_.emplace_back([=](Node *x) {
    for (auto *op : x->outputs) {
      if (op->IsOp() && op_types.count(op->Op()->Type()) &&
          IsNthInput(x, op, argument, nth))
        return true;
    }
    return false;
  });
  return this;
}

PDNode *PDNode::assert_is_ops_nth_output(
    const std::unordered_set<std::string> &op_types,
    const std::string &argument, int nth) {
  assert_is_var();
  asserts_.emplace_back([=](Node *x) {
    for (auto *op : x->inputs) {
      if (op->IsOp() && op_types.count(op->Op()->Type()) &&
          IsNthOutput(x, op, argument, nth))
        return true;
    }
    return false;
  });
  return this;
}
PDNode *PDNode::assert_is_ops_output(
    const std::unordered_set<std::string> &op_types) {
  assert_is_var();
  asserts_.emplace_back([=](Node *x) {
    for (auto *op : x->inputs) {
      if (op && op->IsOp() && op->Op() && op_types.count(op->Op()->Type())) {
        return true;
      }
    }
    return false;
  });
  return this;
}

PDNode *PDNode::assert_is_ops_output(
    const std::unordered_set<std::string> &op_types,
    const std::string &argument) {
  assert_is_var();
  assert_is_ops_nth_output(op_types, argument, 0);
  return this;
}

PDNode *PDNode::assert_is_ops_input(
    const std::unordered_set<std::string> &op_types) {
  assert_is_var();
  asserts_.emplace_back([=](Node *x) {
    for (auto *op : x->outputs) {
      if (op && op->IsOp() && op->Op() && op_types.count(op->Op()->Type())) {
        return true;
      }
    }
    return false;
  });
  return this;
}

PDNode *PDNode::assert_is_ops_input(
    const std::unordered_set<std::string> &op_types,
    const std::string &argument) {
  assert_is_var();
  assert_is_ops_nth_input(op_types, argument, 0);
  return this;
}

bool VarLinksToOp(Node *node, const std::string &op_type) {
  for (auto *out : node->outputs) {
    if (out->IsOp() && out->Op()->Type() == op_type) {
      return true;
    }
  }
  return false;
}

bool IsNthInput(Node *var, Node *op, const std::string &argument, size_t nth) {
  PADDLE_ENFORCE(var->IsVar());
  PADDLE_ENFORCE(op->IsOp());
  if (op->Op()->Input(argument).size() <= nth) return false;
  return var->Name() == op->Op()->Input(argument)[nth];
}

bool IsNthOutput(Node *var, Node *op, const std::string &argument, size_t nth) {
  PADDLE_ENFORCE(var->IsVar());
  PADDLE_ENFORCE(op->IsOp());
  if (op->Op()->Output(argument).size() <= nth) return false;
  return var->Name() == op->Op()->Output(argument)[nth];
}

void GraphSafeRemoveNodes(Graph *graph,
                          const std::unordered_set<const Node *> &nodes) {
  for (auto *node : nodes) {
    graph->RemoveNode(const_cast<Node *>(node));
  }

  for (auto *node : graph->Nodes()) {
    for (auto it = node->inputs.begin(); it != node->inputs.end();) {
      if (nodes.count(*it)) {
        it = const_cast<Node *>(node)->inputs.erase(it);
      } else {
        it++;
      }
    }
    for (auto it = node->outputs.begin(); it != node->outputs.end();) {
      if (nodes.count(*it)) {
        it = const_cast<Node *>(node)->outputs.erase(it);
      } else {
        it++;
      }
    }
  }
}

bool VarLinksFromOp(Node *node, const std::string &op_type) {
  for (auto *out : node->inputs) {
    if (out->IsOp() && out->Op()->Type() == op_type) {
      return true;
    }
  }
  return false;
}

PDNode *patterns::ConvReLU::operator()(
    paddle::framework::ir::PDNode *conv_input) {
  // Create Operators
  conv_input->assert_is_op_input("conv2d", "Input");
  auto *conv_op = pattern->NewNode(conv_repr())->assert_is_op("conv2d");
  auto *relu_op = pattern->NewNode(relu_repr())->assert_is_op("relu");
  // Create variables
  // Filter
  auto *conv_weight_var = pattern->NewNode(conv_weight_repr())
                              ->AsInput()
                              ->assert_is_persistable_var()
                              ->assert_is_op_input("conv2d", "Filter");
  // Bias
  auto *conv_bias_var = pattern->NewNode(conv_bias_repr())
                            ->AsInput()
                            ->assert_is_persistable_var()
                            ->assert_is_op_input("conv2d", "Bias");
  // intermediate variable, will be removed in the IR after fuse.
  auto *conv_out_var = pattern->NewNode(conv_out_repr())
                           ->AsIntermediate()
                           ->assert_is_only_output_of_op("conv2d")
                           ->assert_is_op_input("relu");
  // output
  auto *relu_out_var = pattern->NewNode(relu_out_repr())
                           ->AsOutput()
                           ->assert_is_op_output("relu");

  conv_op->LinksFrom({conv_input, conv_weight_var, conv_bias_var})
      .LinksTo({conv_out_var});
  relu_op->LinksFrom({conv_out_var}).LinksTo({relu_out_var});
  return relu_out_var;
}

PDNode *patterns::FC::operator()(paddle::framework::ir::PDNode *x,
                                 bool with_bias) {
  // Create shared nodes.
  x->assert_is_op_input("mul", "X");
  auto *mul = pattern->NewNode(mul_repr())->assert_is_op("mul");

  auto *mul_w_var = pattern->NewNode(w_repr())
                        ->AsInput()
                        ->assert_is_persistable_var()
                        ->assert_is_op_input("mul", "Y");

  auto *mul_out_var =
      pattern->NewNode(mul_out_repr())->assert_is_op_output("mul");

  if (!with_bias) {  // not with bias
    // Add links.
    mul->LinksFrom({x, mul_w_var}).LinksTo({mul_out_var});
    return mul_out_var;

  } else {  // with bias
    mul_out_var->AsIntermediate()->assert_is_op_input("elementwise_add");
    // Create operators.
    auto *elementwise_add = pattern->NewNode(elementwise_add_repr())
                                ->assert_is_op("elementwise_add");
    // Create variables.
    auto *bias = pattern->NewNode(bias_repr())
                     ->assert_is_op_input("elementwise_add")
                     ->AsInput();

    auto *fc_out = pattern->NewNode(Out_repr())
                       ->AsOutput()
                       ->assert_is_op_output("elementwise_add");

    mul->LinksFrom({mul_w_var, x}).LinksTo({mul_out_var});
    elementwise_add->LinksFrom({mul_out_var, bias}).LinksTo({fc_out});
    return fc_out;
  }
}

PDNode *patterns::LSTM::operator()(PDNode *x) {
  x->assert_is_op_input("lstm", "Input");
  auto *lstm_op = pattern->NewNode(lstm_repr())->assert_is_op("lstm");
#define NEW_NODE(arg__, io__) \
  auto *arg__ =               \
      pattern->NewNode(arg__##_repr())->assert_is_op_##io__("lstm", #arg__);

  // Currently, the H0 and C0 are optional
  // TODO(Superjomn) upgrade the fuse framework to support optional.
  // NEW_NODE(H0, input);
  // NEW_NODE(C0, input);
  NEW_NODE(Weight, input);
  NEW_NODE(Bias, input);

  NEW_NODE(Hidden, output);
  NEW_NODE(Cell, output);
  NEW_NODE(BatchGate, output);
  NEW_NODE(BatchCellPreAct, output);
#undef NEW_NODE

  lstm_op->LinksFrom({x, Weight, Bias});
  lstm_op->LinksTo({Hidden, Cell, BatchGate, BatchCellPreAct});
  return Hidden;
}

PDNode *patterns::GRU::operator()(PDNode *x) {
  x->assert_is_op_input("gru", "Input");
  auto *gru_op = pattern->NewNode(gru_repr())->assert_is_op("gru");
#define NEW_NODE(arg__, io__) \
  auto *arg__ =               \
      pattern->NewNode(arg__##_repr())->assert_is_op_##io__("gru", #arg__);

  NEW_NODE(Weight, input);
  // TODO(Superjomn): upgrade the fuse framework to support optional.
  // H0 and bias are optional
  NEW_NODE(Bias, input);  // also optional
  // NEW_NODE(H0, input);

  NEW_NODE(Hidden, output);
  // below are intermediate
  NEW_NODE(BatchGate, output);
  NEW_NODE(BatchResetHiddenPrev, output);
  NEW_NODE(BatchHidden, output);
#undef NEW_NODE

  BatchGate->AsIntermediate();
  BatchResetHiddenPrev->AsIntermediate();
  BatchHidden->AsIntermediate();

  gru_op->LinksFrom({x, Weight, Bias});
  gru_op->LinksTo({Hidden, BatchGate, BatchResetHiddenPrev, BatchHidden});
  return Hidden;
}

PDNode *patterns::ActElewiseAdd::operator()(
    paddle::framework::ir::PDNode *in_var,
    std::unordered_set<std::string> act_types) {
  in_var->assert_is_ops_input(act_types, "X");

  auto *act = pattern->NewNode(act_repr())->assert_is_ops(act_types);
  auto *act_out_var = pattern->NewNode(act_out_repr())
                          ->assert_is_not_ctrl_var()
                          ->assert_is_ops_output(act_types);
  act_out_var->AsIntermediate()->assert_is_op_input("elementwise_add");

  auto *ele_x_var = pattern->NewNode(ele_x_repr())
                        ->assert_is_not_ctrl_var()
                        ->assert_is_op_input("elementwise_add")
                        ->AsInput();
  auto *elementwise_add =
      pattern->NewNode(ele_add_repr())->assert_is_op("elementwise_add");

  auto *elewise_add_out = pattern->NewNode(elewise_add_out_repr())
                              ->AsOutput()
                              ->assert_is_op_output("elementwise_add", "Out");

  act->LinksFrom({in_var}).LinksTo({act_out_var});
  elementwise_add->LinksFrom({act_out_var, ele_x_var})
      .LinksTo({elewise_add_out});

  return elewise_add_out;
}

PDNode *patterns::ElewiseAddAct::operator()(
    paddle::framework::ir::PDNode *ele_x_var,
    std::unordered_set<std::string> act_types) {
  auto *ele_y_var = pattern->NewNode(ele_y_repr())
                        ->assert_is_op_input("elementwise_add", "Y");

  auto *ele_add =
      pattern->NewNode(ele_add_repr())->assert_is_op("elementwise_add");

  auto *ele_out_var = pattern->NewNode(elewise_add_out_repr())
                          ->assert_is_op_output("elementwise_add", "Out");

  ele_out_var->AsIntermediate()->assert_is_ops_input(act_types);

  auto *act = pattern->NewNode(act_repr())->assert_is_ops(act_types);

  auto *act_out_var =
      pattern->NewNode(act_out_repr())->assert_is_ops_output(act_types, "Out");

  ele_add->LinksFrom({ele_x_var, ele_y_var}).LinksTo({ele_out_var});
  act->LinksFrom({ele_out_var}).LinksTo({act_out_var});

  return act_out_var;
}

PDNode *patterns::ElewiseAddActInplaceGrad::operator()(
    paddle::framework::ir::PDNode *d_act_out_var,
    std::unordered_set<std::string> act_types) {
  // act_grad: in["Out", "Out@GRAD"], out["X@GRAD"]
  // ele_add_grad: in["Y", "Out@GRAD"], out["X@GRAD", "Y@GRAD"]
  auto *act_grad = pattern->NewNode(act_grad_repr())->assert_is_ops(act_types);

  auto *act_out_var =
      pattern->NewNode(act_out_repr())->assert_is_ops_input(act_types, "Out");

  auto *d_intermediate_var =
      pattern->NewNode(d_itermediate_out_repr())
          ->assert_is_ops_output(act_types, GradVarName("X"));

  act_grad->LinksFrom({d_act_out_var, act_out_var})
      .LinksTo({d_intermediate_var});

  auto *ele_y_var = pattern->NewNode(ele_y_repr())
                        ->assert_is_not_ctrl_var()
                        ->assert_is_op_input("elementwise_add_grad", "Y");

  auto *ele_add_grad = pattern->NewNode(ele_add_grad_repr())
                           ->assert_is_op("elementwise_add_grad");

  auto *d_ele_x_var =
      pattern->NewNode(d_ele_x_repr())
          ->assert_is_not_ctrl_var()
          ->assert_is_op_output("elementwise_add_grad", GradVarName("X"));

  auto *d_ele_y_var =
      pattern->NewNode(d_ele_y_repr())
          ->assert_is_not_ctrl_var()
          ->assert_is_op_output("elementwise_add_grad", GradVarName("Y"));

  ele_add_grad->LinksFrom({d_intermediate_var, ele_y_var})
      .LinksTo({d_ele_x_var, d_ele_y_var});

  return ele_add_grad;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle
