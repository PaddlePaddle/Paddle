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

#include "paddle/fluid/framework/details/op_handle_graph.h"
#include <queue>
#include <utility>

namespace paddle {
namespace framework {
namespace details {

OpHandleGraph::OpHandleGraph(
    const std::vector<std::unique_ptr<OpHandleBase>> &ops) {
  BuildGraph(ops);
}

void OpHandleGraph::BuildGraph(
    const std::vector<std::unique_ptr<OpHandleBase>> &ops) {
  for (auto &op : ops) {
    preceding_ops_[op.get()];
    pending_ops_[op.get()];
    for (auto &var : op->Outputs()) {
      for (auto &pending_op : var->PendingOps()) {
        preceding_ops_[pending_op].insert(op.get());
        pending_ops_[op.get()].insert(pending_op);
      }
    }
  }
  PADDLE_ENFORCE(
      preceding_ops_.size() == ops.size() && pending_ops_.size() == ops.size(),
      "There are duplicate ops in graph.");
}

size_t OpHandleGraph::OpNumber() const { return preceding_ops_.size(); }

std::unordered_set<OpHandleBase *> OpHandleGraph::AllOps() const {
  std::unordered_set<OpHandleBase *> ret;
  for (auto &pair : preceding_ops_) {
    ret.insert(pair.first);
  }
  return ret;
}

bool OpHandleGraph::HasOp(OpHandleBase *op) const {
  return preceding_ops_.count(op) != 0;
}

void OpHandleGraph::EnforceHasOp(OpHandleBase *op) const {
  PADDLE_ENFORCE(HasOp(op), "Cannot found op %s in OpHandleGraph",
                 op == nullptr ? "nullptr" : op->DebugString());
}

const std::unordered_set<OpHandleBase *> &OpHandleGraph::PrecedingOps(
    OpHandleBase *op) const {
  EnforceHasOp(op);
  return preceding_ops_.at(op);
}

const std::unordered_set<OpHandleBase *> &OpHandleGraph::PendingOps(
    OpHandleBase *op) const {
  EnforceHasOp(op);
  return pending_ops_.at(op);
}

std::vector<std::unordered_set<OpHandleBase *>> OpHandleGraph::AllPrecedingOps(
    OpHandleBase *op) const {
  EnforceHasOp(op);
  std::queue<OpHandleBase *> queue[2];
  int cur = 0;
  std::unordered_set<OpHandleBase *> visited_ops;
  std::vector<std::unordered_set<OpHandleBase *>> ret;
  for (auto &tmp : preceding_ops_.at(op)) {
    queue[cur].push(tmp);
    visited_ops.insert(tmp);
  }

  while (!queue[cur].empty()) {
    std::unordered_set<OpHandleBase *> cur_level_ops;
    auto *tmp = queue[cur].front();
    queue[cur].pop();
    for (auto &preceding_op : preceding_ops_.at(tmp)) {
      if (visited_ops.count(preceding_op)) {
        continue;
      } else {
        queue[1 - cur].push(preceding_op);
        cur_level_ops.insert(preceding_op);
        visited_ops.insert(preceding_op);
      }
    }
    if (!cur_level_ops.empty()) {
      ret.emplace_back(std::move(cur_level_ops));
    }
    cur = 1 - cur;
  }
  return ret;
}

std::vector<std::unordered_set<OpHandleBase *>> OpHandleGraph::AllPendingOps(
    OpHandleBase *op) const {
  EnforceHasOp(op);
  std::queue<OpHandleBase *> queue[2];
  int cur = 0;
  std::unordered_set<OpHandleBase *> visited_ops;
  std::vector<std::unordered_set<OpHandleBase *>> ret;
  for (auto &tmp : preceding_ops_.at(op)) {
    queue[cur].push(tmp);
    visited_ops.insert(tmp);
  }

  while (!queue[cur].empty()) {
    std::unordered_set<OpHandleBase *> cur_level_ops;
    auto *tmp = queue[cur].front();
    queue[cur].pop();
    for (auto &next_op : pending_ops_.at(tmp)) {
      if (visited_ops.count(next_op)) {
        continue;
      } else {
        queue[1 - cur].push(next_op);
        cur_level_ops.insert(next_op);
        visited_ops.insert(next_op);
      }
    }
    if (!cur_level_ops.empty()) {
      ret.emplace_back(std::move(cur_level_ops));
    }
    cur = 1 - cur;
  }
  return ret;
}

OpHandleGraph::Relation OpHandleGraph::RelationBetween(
    OpHandleBase *op1, OpHandleBase *op2) const {
  EnforceHasOp(op1);
  EnforceHasOp(op2);
  if (op1 == op2) {
    return kSame;
  } else if (IsBeforeOrSameImpl(op1, op2)) {
    return kBefore;
  } else if (IsBeforeOrSameImpl(op2, op1)) {
    return kAfter;
  } else {
    return kNoDeps;
  }
}

bool OpHandleGraph::IsSame(OpHandleBase *op1, OpHandleBase *op2) const {
  EnforceHasOp(op1);
  EnforceHasOp(op2);
  return op1 == op2;
}

bool OpHandleGraph::IsBeforeOrSame(OpHandleBase *op1, OpHandleBase *op2) const {
  EnforceHasOp(op1);
  EnforceHasOp(op2);
  return IsBeforeOrSameImpl(op1, op2);
}

bool OpHandleGraph::IsBefore(OpHandleBase *op1, OpHandleBase *op2) const {
  EnforceHasOp(op1);
  EnforceHasOp(op2);
  return op1 != op2 && IsBeforeOrSameImpl(op1, op2);
}

bool OpHandleGraph::IsBeforeOrSameImpl(OpHandleBase *op1,
                                       OpHandleBase *op2) const {
  std::queue<OpHandleBase *> queue;
  // BFS
  queue.push(op1);
  do {
    auto *op = queue.front();
    queue.pop();
    if (op == op2) return true;
    for (auto &pending_op : pending_ops_.at(op)) {
      queue.push(pending_op);
    }
  } while (!queue.empty());
  return false;
}

bool OpHandleGraph::IsAfterOrSame(OpHandleBase *op1, OpHandleBase *op2) const {
  EnforceHasOp(op1);
  EnforceHasOp(op2);
  return IsBeforeOrSameImpl(op2, op1);
}

bool OpHandleGraph::IsAfter(OpHandleBase *op1, OpHandleBase *op2) const {
  return IsBefore(op2, op1);
}

bool OpHandleGraph::IsNoDeps(OpHandleBase *op1, OpHandleBase *op2) const {
  return RelationBetween(op1, op2) == kNoDeps;
}

std::unordered_set<OpHandleBase *> OpHandleGraph::NoPendingOpSet() const {
  std::unordered_set<OpHandleBase *> ret;
  for (auto &pair : pending_ops_) {
    if (pair.second.empty()) ret.insert(pair.first);
  }
  return ret;
}

std::unordered_set<OpHandleBase *> OpHandleGraph::NoPrecedingOpSet() const {
  std::unordered_set<OpHandleBase *> ret;
  for (auto &pair : preceding_ops_) {
    if (pair.second.empty()) ret.insert(pair.first);
  }
  return ret;
}

OpHandleBase *OpHandleGraph::NearestCommonParent(OpHandleBase *op1,
                                                 OpHandleBase *op2) const {
  EnforceHasOp(op1);
  EnforceHasOp(op2);
  // FIXME(zjl): A brute-force O(2*n) algorithm here
  // First, BFS all preceding_ops of op1 and record them in set S
  // Second, BFS all preceding_ops of op2 and found whether it is in set S
  std::unordered_set<OpHandleBase *> all_preceding_ops;
  std::queue<OpHandleBase *> queue;
  queue.push(op1);
  do {
    auto *op = queue.front();
    queue.pop();
    all_preceding_ops.insert(op);
    for (auto &preceding_op : preceding_ops_.at(op)) {
      queue.push(preceding_op);
    }
  } while (!queue.empty());

  queue.push(op2);
  do {
    auto *op = queue.front();
    queue.pop();
    if (all_preceding_ops.count(op)) return op;
    for (auto &preceding_op : preceding_ops_.at(op)) {
      queue.push(preceding_op);
    }
  } while (!queue.empty());
  return nullptr;
}

OpHandleBase *OpHandleGraph::NearestCommonParentAfter(OpHandleBase *op,
                                                      OpHandleBase *op1,
                                                      OpHandleBase *op2) const {
  EnforceHasOp(op);
  EnforceHasOp(op1);
  EnforceHasOp(op2);
  std::unordered_map<OpHandleBase *, int> all_preceding_ops;
  int max_depth = -1;
  std::queue<std::pair<OpHandleBase *, int>> queue;
  queue.push(std::make_pair(op1, 0));
  do {
    auto tmp = queue.front();
    queue.pop();
    all_preceding_ops.insert(tmp);
    if (tmp.first == op1) {
      max_depth = tmp.second;
      break;
    }
    for (auto &preceding_op : preceding_ops_.at(tmp.first)) {
      queue.push(std::make_pair(preceding_op, tmp.second + 1));
    }
  } while (!queue.empty());

  if (max_depth == -1) {
    return nullptr;
  }

  std::queue<OpHandleBase *> queue2;
  queue2.push(op2);
  do {
    auto *tmp = queue2.front();
    queue2.pop();
    if (all_preceding_ops.count(tmp) &&
        (tmp == op || all_preceding_ops[tmp] < max_depth)) {
      return tmp;
    }
  } while (!queue2.empty());
  return nullptr;
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
