/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/inference/analysis/data_flow_graph.h"
#include "paddle/fluid/inference/analysis/dot.h"
#include "paddle/fluid/inference/analysis/node.h"

namespace paddle {
namespace inference {
namespace analysis {
using ir_node_t = framework::ir::Node;
using ir_graph_t = framework::ir::Graph;

// It is a better idea that the inputs and outputs of this graph is set manually
// before, but there must be a Pass that helps to prune the unnecessary ops that
// do not contribute to the given targets, so in this pass, analysis and get the
// inputs and outputs is OK.
void DataFlowGraph::Build() {
  inputs_.clear();
  outputs_.clear();
  std::unordered_set<Node *> ins;
  std::unordered_set<Node *> outs;
  for (auto &node : nodes.nodes()) {
    for (auto *in : node->inlinks) {
      ins.insert(in);
    }
    for (auto *out : node->outlinks) {
      outs.insert(out);
    }
  }

  // The nodes that in ins but not in outs is the graph's inputs
  // similarly, the nodes that in outs but not in ins is the graphs' outputs
  for (auto *in : ins) {
    if (!outs.count(in)) {
      inputs_.push_back(in);
    }
  }
  for (auto *out : outs) {
    if (!ins.count(out)) {
      outputs_.push_back(out);
    }
  }

  Clean();
}

void DataFlowGraph::Build(const framework::proto::ProgramDesc &prog) {
  // insert vars
  // The `var2id` keeps a map from a variable's name to its Node-id, the Node-id
  // will keep updating to its latest alias during the graph-building.
  std::unordered_map<std::string, size_t> var2id;
  auto &main_block = prog.blocks(framework::kRootBlockIndex);
  for (int i = 0; i < main_block.vars_size(); i++) {
    const auto &var = main_block.vars(i);
    auto *v = nodes.Create(Node::Type::kValue);
    v->SetName(var.name());
    v->SetPbDesc(const_cast<void *>(static_cast<const void *>(&var)));
    v->SetPbMsg(var.SerializeAsString());
    var2id[var.name()] = v->id();
  }

  // The variables in a SSA can only write once, so if a variable is written
  // multiple times(quite common in our ProgramDesc design), multiple alias
  // Nodes of this variable will be created, and each will just write once.

  // An set that keep all the names of the variables(the original, not alias)
  // that have been written(as outputs). Once an Op's output variable hit the
  // set, it should create a new alias and update the global alias for this
  // variable. And that make a Data Flow Graph a SSA.
  std::unordered_set<Node *> unique_written_vars;
  for (int i = 0; i < main_block.ops_size(); i++) {
    const auto &op = main_block.ops(i);
    auto *o = nodes.Create(Node::Type::kFunction);
    o->SetName(op.type());
    static_cast<Function *>(o)->SetFuncType(op.type());
    // Link to the original protobuf message's memory, make it easier to
    // generate from a data flow graph to fluid ProgramDesc.
    o->SetPbDesc(const_cast<void *>(static_cast<const void *>(&op)));
    o->SetPbMsg(op.SerializeAsString());

    // set inputs and outputs
    for (int j = 0; j < op.inputs_size(); j++) {
      auto &in_var = op.inputs(j);
      for (int k = 0; k < in_var.arguments_size(); k++) {
        auto *in = nodes.GetMutable(var2id.at(in_var.arguments(k)));
        in->outlinks.push_back(o);
        o->inlinks.push_back(in);
        unique_written_vars.insert(in);
      }
    }
    for (int j = 0; j < op.outputs_size(); j++) {
      auto &out_var = op.outputs(j);
      for (int k = 0; k < out_var.arguments_size(); k++) {
        auto *out = nodes.GetMutable(var2id[out_var.arguments(k)]);
        if (unique_written_vars.count(out)) {
          // Loop found, for example, a = op(a), use SSA, change to a1 = op(a).
          auto *out_alias = nodes.Create(Node::Type::kValue);
          out_alias->SetName(out->name());
          out_alias->SetPbDesc(out->pb_desc());
          out_alias->SetPbMsg(out->pb_msg());
          var2id[out_alias->name()] =
              out_alias->id();  // update variable's alias Node
          VLOG(40) << "loop found in graph, create SSA alias node ["
                   << out_alias->repr() << "] for [" << out->repr() << "]";
          out = out_alias;
        }
        out->inlinks.push_back(o);
        o->outlinks.push_back(out);
      }
    }
  }
  // Analysis and extract the inputs and outputs of this graph.
  Build();
}

void DataFlowGraph::Build(const framework::ir::Graph &graph) {
  // Create nodes
  std::unordered_map<ir_node_t *, Node *> ir_node_map;
  for (auto *ir_node : graph.Nodes()) {
    Node *x{nullptr};
    if (ir_node->IsOp()) {
      PADDLE_ENFORCE(ir_node->Op());
      VLOG(40) << "get op " << ir_node << " " << ir_node->Name();
      x = nodes.Create(Node::Type::kFunction);
      x->attr("ir_node").Pointer() = ir_node;
      PADDLE_ENFORCE(ir_node->Op()->Proto());
      x->SetName(ir_node->Op()->Proto()->type());
      x->SetPbMsg(ir_node->Op()->Proto()->SerializeAsString());
    } else if (ir_node->IsVar()) {
      // Not create a Node for IR ControlDepVar, considering Inference currently
      // just used in single thread scenerio.
      VLOG(40) << "get var " << ir_node->Name();
      x = nodes.Create(Node::Type::kValue);
      x->attr("ir_node").Pointer() = ir_node;
      x->SetName(ir_node->Name());
      // x->SetPbMsg(ir_node->Var()->Proto()->SerializeAsString());
    } else {
      PADDLE_THROW("Failed to create an Node from IR, unknown type");
    }
    ir_node_map.emplace(ir_node, x);
  }
  VLOG(40) << "finish creating Nodes";

  VLOG(40) << "to create edge";
  // Create links
  for (auto *ir_node : graph.Nodes()) {
    auto it = ir_node_map.find(ir_node);
    // Skip ControlDepVar.
    if (it == ir_node_map.end()) continue;
    auto *node = it->second;
    for (auto *x : ir_node->inputs) {
      if (!ir_node_map.count(x)) continue;
      node->inlinks.push_back(ir_node_map.at(x));
    }
    for (auto *x : ir_node->outputs) {
      if (!ir_node_map.count(x)) continue;
      node->outlinks.push_back(ir_node_map.at(x));
    }
  }

  Build();
  PADDLE_ENFORCE(!inputs_.empty(),
                 "Can't deduce any inputs from the graph, Is the graph empty?");

  ir_graph = &graph;
  VLOG(30) << "finished build from IR";
}

void DataFlowGraph::Clean() {
  for (auto &node : nodes.nodes()) {
    std::unordered_set<Node *> inlinks_set(node->inlinks.begin(),
                                           node->inlinks.end());
    std::unordered_set<Node *> outlinks_set(node->outlinks.begin(),
                                            node->outlinks.end());
    if (inlinks_set.size() < node->inlinks.size()) {
      node->inlinks.assign(inlinks_set.begin(), inlinks_set.end());
    }
    if (outlinks_set.size() < node->outlinks.size()) {
      node->outlinks.assign(outlinks_set.begin(), outlinks_set.end());
    }
  }
}

std::string DataFlowGraph::DotString() const {
  Dot dot;

  // Add nodes
  for (size_t i = 0; i < nodes.size(); i++) {
    const Node &node = nodes.Get(i);
    dot.AddNode(node.repr(), node.dot_attrs());
  }

  // Add edges
  for (size_t i = 0; i < nodes.size(); i++) {
    const Node &node = nodes.Get(i);
    for (auto &in : node.inlinks) {
      dot.AddEdge(in->repr(), node.repr(), {});
    }
  }
  return dot.Build();
}

std::string DataFlowGraph::HumanReadableInfo(bool show_values,
                                             bool show_functions) const {
  std::stringstream values, functions;
  for (auto &n : nodes.nodes()) {
    if (show_values && n->IsValue()) {
      values << n->repr() << "\n";
    }
    if (show_functions && n->IsFunction()) {
      functions << n->repr() << "\n";
    }
  }
  return "Values:\n" + values.str() + "\n\n" + "Functions:\n" + functions.str();
}

//
// NodesBFSIterator
//

GraphTraits<DataFlowGraph>::NodesBFSIterator::NodesBFSIterator(
    const std::vector<Node *> &source)
    : queue_(source.begin(), source.end()) {}

GraphTraits<DataFlowGraph>::NodesBFSIterator::NodesBFSIterator(
    GraphTraits<DataFlowGraph>::NodesBFSIterator &&other) noexcept
    : queue_(std::move(other.queue_)),
      visited_(std::move(other.visited_)) {}

GraphTraits<DataFlowGraph>::NodesBFSIterator::NodesBFSIterator(
    const GraphTraits<DataFlowGraph>::NodesBFSIterator &other)
    : queue_(other.queue_), visited_(other.visited_) {}

Node &GraphTraits<DataFlowGraph>::NodesBFSIterator::operator*() {
  PADDLE_ENFORCE(!queue_.empty());
  return *queue_.front();
}

Node *GraphTraits<DataFlowGraph>::NodesBFSIterator::operator->() {
  PADDLE_ENFORCE(!queue_.empty());
  return queue_.front();
}

GraphTraits<DataFlowGraph>::NodesBFSIterator &
GraphTraits<DataFlowGraph>::NodesBFSIterator::operator=(
    const GraphTraits<DataFlowGraph>::NodesBFSIterator &other) {
  queue_ = other.queue_;
  visited_ = other.visited_;
  return *this;
}

GraphTraits<DataFlowGraph>::NodesBFSIterator
    &GraphTraits<DataFlowGraph>::NodesBFSIterator::operator++() {
  PADDLE_ENFORCE(!queue_.empty());
  auto *cur = queue_.front();
  visited_.insert(cur);
  queue_.pop_front();
  for (auto *output : cur->outlinks) {
    if (!visited_.count(output)) {
      queue_.push_back(output);
      visited_.insert(output);
    }
  }
  return *this;
}

bool GraphTraits<DataFlowGraph>::NodesBFSIterator::operator==(
    const GraphTraits<DataFlowGraph>::NodesBFSIterator &other) {
  if (queue_.empty()) return other.queue_.empty();
  if ((!queue_.empty()) && (!other.queue_.empty())) {
    return queue_.front() == other.queue_.front() &&
           visited_.size() == other.visited_.size();
    // equality of queue and
    // visited. Just a light but week implementation.
  }
  return false;
}

//
// NodesDFSIterator
//
GraphTraits<DataFlowGraph>::NodesDFSIterator::NodesDFSIterator(
    const std::vector<Node *> &source) {
  for (auto *x : source) stack_.push(x);
}

GraphTraits<DataFlowGraph>::NodesDFSIterator::NodesDFSIterator(
    GraphTraits<DataFlowGraph>::NodesDFSIterator &&other) noexcept
    : stack_(std::move(other.stack_)),
      visited_(std::move(other.visited_)) {}

GraphTraits<DataFlowGraph>::NodesDFSIterator::NodesDFSIterator(
    const GraphTraits<DataFlowGraph>::NodesDFSIterator &other)
    : stack_(other.stack_), visited_(other.visited_) {}

Node &GraphTraits<DataFlowGraph>::NodesDFSIterator::operator*() {
  PADDLE_ENFORCE(!stack_.empty());
  return *stack_.top();
}

GraphTraits<DataFlowGraph>::NodesDFSIterator
    &GraphTraits<DataFlowGraph>::NodesDFSIterator::operator++() {
  if (stack_.empty()) return *this;
  visited_.insert(stack_.top());
  auto *cur = stack_.top();
  stack_.pop();
  for (auto *x : cur->outlinks) {
    if (!visited_.count(x)) {
      stack_.push(x);
      visited_.insert(x);
    }
  }
  return *this;
}
bool GraphTraits<DataFlowGraph>::NodesDFSIterator::operator==(
    const GraphTraits<DataFlowGraph>::NodesDFSIterator &other) {
  if (stack_.empty()) return other.stack_.empty();
  if ((!stack_.empty()) && (!other.stack_.empty())) {
    return stack_.top() == other.stack_.top();
  }
  return false;
}

GraphTraits<DataFlowGraph>::NodesDFSIterator &
GraphTraits<DataFlowGraph>::NodesDFSIterator::operator=(
    const GraphTraits<DataFlowGraph>::NodesDFSIterator &other) {
  stack_ = other.stack_;
  visited_ = other.visited_;
  return *this;
}
Node *GraphTraits<DataFlowGraph>::NodesDFSIterator::operator->() {
  return stack_.top();
}

inline bool CheckNodeIndegreeEquals(const Node &node, size_t n) {
  return node.inlinks.size() == n;
}

GraphTraits<DataFlowGraph>::NodesTSIterator::NodesTSIterator(
    const std::vector<Node *> &source) {
  PADDLE_ENFORCE(!source.empty(),
                 "Start points of topological sorting should not be empty!");
  // CHECK all the inputs' in-degree is 0
  for (auto *node : source) {
    PADDLE_ENFORCE(CheckNodeIndegreeEquals(*node, 0));
  }

  std::unordered_set<Node *> visited;
  std::unordered_set<Node *> to_visit{source.begin(), source.end()};

  std::vector<Node *> inlink_visited;
  while (!to_visit.empty()) {
    std::vector<Node *> queue(to_visit.begin(), to_visit.end());
    for (auto *p : queue) {
      if (p->deleted()) {
        visited.insert(p);
        to_visit.erase(p);
        continue;
      }
      inlink_visited.clear();

      std::copy_if(p->inlinks.begin(), p->inlinks.end(),
                   std::back_inserter(inlink_visited),
                   [&](Node *x) { return visited.count(x); });

      if (inlink_visited.size() == p->inlinks.size()) {
        sorted_.push_back(p);
        for (auto *_ : p->outlinks) {
          if (!visited.count(_)) {
            to_visit.insert(_);
          }
        }

        to_visit.erase(p);
        visited.insert(p);
      }
    }
  }
}

GraphTraits<DataFlowGraph>::NodesTSIterator::NodesTSIterator(
    const paddle::inference::analysis::GraphTraits<
        DataFlowGraph>::NodesTSIterator &other)
    : sorted_(other.sorted_), cursor_(other.cursor_) {}

Node &GraphTraits<DataFlowGraph>::NodesTSIterator::operator*() {
  PADDLE_ENFORCE_LT(cursor_, sorted_.size());
  return *sorted_[cursor_];
}

paddle::inference::analysis::GraphTraits<DataFlowGraph>::NodesTSIterator
    &GraphTraits<DataFlowGraph>::NodesTSIterator::operator++() {
  if (++cursor_ >= sorted_.size()) {
    sorted_.clear();
    cursor_ = 0;
  }
  return *this;
}
paddle::inference::analysis::GraphTraits<DataFlowGraph>::NodesTSIterator &
GraphTraits<DataFlowGraph>::NodesTSIterator::operator=(
    const paddle::inference::analysis::GraphTraits<
        DataFlowGraph>::NodesTSIterator &other) {
  cursor_ = other.cursor_;
  sorted_ = other.sorted_;
  return *this;
}

bool GraphTraits<DataFlowGraph>::NodesTSIterator::operator==(
    const paddle::inference::analysis::GraphTraits<
        DataFlowGraph>::NodesTSIterator &other) {
  return sorted_ == other.sorted_ && cursor_ == other.cursor_;
}

Node *GraphTraits<DataFlowGraph>::NodesTSIterator::operator->() {
  PADDLE_ENFORCE_LT(cursor_, sorted_.size());
  return sorted_[cursor_];
}

std::pair<std::vector<Node *>, std::vector<Node *>>
ExtractInputAndOutputOfSubGraph(std::vector<Node *> &graph) {  // NOLINT
  std::unordered_set<Node *> nodes(graph.begin(), graph.end());
  std::unordered_set<Node *> inputs;
  std::unordered_set<Node *> outputs;
  // Input a Value, check whether its inlink is in the subgraph.
  auto inlink_in_subgraph = [&](Node *n) {
    for (auto *in : n->inlinks) {
      if (nodes.count(in)) return true;
    }
    return false;
  };

  for (auto &node : graph) {
    for (auto *in : node->inlinks) {
      // The Value that is written by nodes inside a sub-graph shouldn't be the
      // input of the sub-graph.
      if (!nodes.count(in) && in->type() == Node::Type::kValue &&
          !inlink_in_subgraph(in)) {
        inputs.insert(in);
      }
    }
    for (auto *out : node->outlinks) {
      if (!nodes.count(out) && out->type() == Node::Type::kValue) {
        outputs.insert(out);
      }
    }
  }
  return std::make_pair(std::vector<Node *>(inputs.begin(), inputs.end()),
                        std::vector<Node *>(outputs.begin(), outputs.end()));
}

// Filter the Intermediate results of the subgraph node.
void FilterRedundantOutputOfSubGraph(DataFlowGraph *graph) {
  std::vector<Node *> op_nodes;
  for (auto &node : GraphTraits<DataFlowGraph>(*graph).nodes_in_TS()) {
    if (node.type() == Node::Type::kValue || node.deleted()) {
      continue;
    }
    op_nodes.push_back(&node);
  }
  size_t op_num = op_nodes.size();
  for (size_t i = 0; i < op_num; i++) {
    if (op_nodes[i]->type() == Node::Type::kFunction) continue;
    std::unordered_set<std::string> follow_up_input_names;
    for (size_t j = i + 1; j < op_num; j++) {
      for (auto *in : op_nodes[j]->inlinks) {
        follow_up_input_names.insert(in->name());
      }
    }
    std::vector<Node *> filtered_subgraph_outlinks;
    for (auto *out : op_nodes[i]->outlinks) {
      if (follow_up_input_names.count(out->name())) {
        filtered_subgraph_outlinks.push_back(out);
      } else {
        out->SetDeleted();
      }
    }
    // The filtered_subgraph_outlinks may be empty.
    op_nodes[i]->outlinks = filtered_subgraph_outlinks;
  }
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
