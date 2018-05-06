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

namespace paddle {
namespace inference {
namespace analysis {

std::string DataFlowGraph::DotString() const {
  Dot dot;

  // Add nodes
  for (size_t i = 0; i < nodes.size(); i++) {
    const Node &node = nodes.Get(i);
    switch (node.type()) {
      case Node::Type::kValue:
        dot.AddNode(node.repr(), node.dot_attrs());
        break;
      case Node::Type::kFunction:
        dot.AddNode(node.repr(), node.dot_attrs());
        break;
      case Node::Type::kFunctionBlock:
        dot.AddNode(node.repr(), node.dot_attrs());
        break;
      default:
        PADDLE_THROW("unsupported Node type %d", static_cast<int>(node.type()));
    }
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
  for (auto *input : cur->inlinks) {
    if (!visited_.count(input)) {
      queue_.push_back(input);
    }
  }
  return *this;
}

bool GraphTraits<DataFlowGraph>::NodesBFSIterator::operator==(
    const GraphTraits<DataFlowGraph>::NodesBFSIterator &other) {
  if (queue_.empty()) return other.queue_.empty();
  if ((!queue_.empty()) && (!other.queue_.empty())) {
    return queue_.front() == other.queue_.front();
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
  for (auto *x : stack_.top()->outlinks) {
    if (!visited_.count(x)) stack_.push(x);
  }
  stack_.pop();
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

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
