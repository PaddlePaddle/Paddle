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

#include <list>
#include <map>

#include "paddle/fluid/framework/ir/graph_traits.h"

namespace paddle {
namespace framework {
namespace ir {

//
// NodesDFSIterator
//
class Node;

bool IsReachable(ir::Graph *graph, Node *from, Node *to) {
  if (from == to) {
    return true;
  }

  std::map<Node *, bool> visited;

  for (auto &node : GraphTraits::DFS(*graph)) {
    visited[&node] = false;
  }

  visited[from] = true;

  std::list<Node *> queue;
  queue.push_back(from);

  while (!queue.empty()) {
    auto cur = FindNode(graph, queue.front());
    queue.pop_front();

    if (!cur) return false;

    for (const auto &n : cur->outputs) {
      if (n == to) {
        return true;
      }

      if (!visited[n]) {
        visited[n] = true;
        queue.push_back(n);
      }
    }
  }
  return false;
}

Node *FindNode(ir::Graph *graph, const Node *node) {
  for (const auto &n : graph->Nodes()) {
    if (n == node) {
      return n;
    }
  }
  return nullptr;
}

NodesDFSIterator::NodesDFSIterator(const std::vector<Node *> &source) {
  for (auto *x : source) stack_.push(x);
}

NodesDFSIterator::NodesDFSIterator(NodesDFSIterator &&other) noexcept
    : stack_(std::move(other.stack_)),
      visited_(std::move(other.visited_)) {}

NodesDFSIterator::NodesDFSIterator(const NodesDFSIterator &other)
    : stack_(other.stack_), visited_(other.visited_) {}

Node &NodesDFSIterator::operator*() {
  PADDLE_ENFORCE_EQ(stack_.empty(), false, platform::errors::OutOfRange(
                                               "The iterator exceeds range."));
  return *stack_.top();
}

NodesDFSIterator &NodesDFSIterator::operator++() {
  PADDLE_ENFORCE_EQ(stack_.empty(), false, platform::errors::OutOfRange(
                                               "The iterator exceeds range."));
  visited_.insert(stack_.top());
  auto *cur = stack_.top();
  stack_.pop();
  for (auto *x : cur->outputs) {
    if (!visited_.count(x)) {
      stack_.push(x);
    }
  }
  return *this;
}
bool NodesDFSIterator::operator==(const NodesDFSIterator &other) {
  if (stack_.empty()) return other.stack_.empty();
  if ((!stack_.empty()) && (!other.stack_.empty())) {
    return stack_.top() == other.stack_.top();
  }
  return false;
}

NodesDFSIterator &NodesDFSIterator::operator=(const NodesDFSIterator &other) {
  stack_ = other.stack_;
  visited_ = other.visited_;
  return *this;
}
Node *NodesDFSIterator::operator->() { return stack_.top(); }

inline bool CheckNodeIndegreeEquals(const Node &node, size_t n) {
  return node.inputs.size() == n;
}

NodesTSIterator::NodesTSIterator(const std::vector<Node *> &source) {
  PADDLE_ENFORCE_EQ(
      source.empty(), false,
      platform::errors::InvalidArgument(
          "Start points of topological sorting should not be empty!"));
  // CHECK all the inputs' in-degree is 0
  for (auto *node : source) {
    PADDLE_ENFORCE_EQ(
        CheckNodeIndegreeEquals(*node, 0), true,
        platform::errors::InvalidArgument(
            "In start points of topological sorting, the indegree of each "
            "point should be 0. Node(%s)'s indegree is not 0.",
            node->Name()));
  }

  std::set<Node *> to_visit{source.begin(), source.end()};
  std::vector<Node *> inlink_sorted;
  while (!to_visit.empty()) {
    std::vector<Node *> queue(to_visit.begin(), to_visit.end());
    for (auto *p : queue) {
      to_visit.erase(p);
      sorted_.push_back(p);
      for (auto *out : p->outputs) {
        inlink_sorted.clear();
        std::copy_if(out->inputs.begin(), out->inputs.end(),
                     std::back_inserter(inlink_sorted), [&](Node *x) -> bool {
                       return std::find(sorted_.begin(), sorted_.end(), x) !=
                              sorted_.end();
                     });
        if (inlink_sorted.size() == out->inputs.size()) {
          to_visit.insert(out);
        }
      }
    }
  }
}

NodesTSIterator::NodesTSIterator(const NodesTSIterator &other)
    : sorted_(other.sorted_), cursor_(other.cursor_) {}

Node &NodesTSIterator::operator*() {
  PADDLE_ENFORCE_LT(
      cursor_, sorted_.size(),
      platform::errors::OutOfRange(
          "The iterator exceeds range. Container size is %d, but index is %d.",
          sorted_.size(), cursor_));
  return *sorted_[cursor_];
}

NodesTSIterator &NodesTSIterator::operator++() {
  if (++cursor_ >= sorted_.size()) {
    sorted_.clear();
    cursor_ = 0;
  }
  return *this;
}
NodesTSIterator &NodesTSIterator::operator=(const NodesTSIterator &other) {
  cursor_ = other.cursor_;
  sorted_ = other.sorted_;
  return *this;
}

bool NodesTSIterator::operator==(const NodesTSIterator &other) {
  return sorted_ == other.sorted_ && cursor_ == other.cursor_;
}

Node *NodesTSIterator::operator->() {
  PADDLE_ENFORCE_LT(
      cursor_, sorted_.size(),
      platform::errors::OutOfRange(
          "The iterator exceeds range. Container size is %d, but index is %d.",
          sorted_.size(), cursor_));
  return sorted_[cursor_];
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle
