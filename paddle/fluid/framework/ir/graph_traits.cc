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

#include "paddle/fluid/framework/ir/graph_traits.h"

#include <set>
#include <utility>
#include <vector>

namespace paddle {
namespace framework {
namespace ir {

//
// NodesDFSIterator
//
NodesDFSIterator::NodesDFSIterator(const std::vector<Node *> &source) {
  for (auto *x : source) stack_.push(x);
}

NodesDFSIterator::NodesDFSIterator(NodesDFSIterator &&other) noexcept
    : stack_(std::move(other.stack_)),
      visited_(std::move(other.visited_)) {}

NodesDFSIterator::NodesDFSIterator(const NodesDFSIterator &other)
    : stack_(other.stack_), visited_(other.visited_) {}

Node &NodesDFSIterator::operator*() {
  PADDLE_ENFORCE(!stack_.empty());
  return *stack_.top();
}

NodesDFSIterator &NodesDFSIterator::operator++() {
  PADDLE_ENFORCE(!stack_.empty(), "the iterator exceeds range");
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
  PADDLE_ENFORCE(!source.empty(),
                 "Start points of topological sorting should not be empty!");
  // CHECK all the inputs' in-degree is 0
  for (auto *node : source) {
    PADDLE_ENFORCE(CheckNodeIndegreeEquals(*node, 0));
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
  PADDLE_ENFORCE_LT(cursor_, sorted_.size());
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
  PADDLE_ENFORCE_LT(cursor_, sorted_.size());
  return sorted_[cursor_];
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle
