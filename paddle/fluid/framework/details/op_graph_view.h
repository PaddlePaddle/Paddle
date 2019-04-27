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

#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "paddle/fluid/framework/details/op_handle_base.h"

namespace paddle {
namespace framework {
namespace details {

class OpGraphView {
 public:
  explicit OpGraphView(const std::vector<OpHandleBase *> &ops);

  std::unordered_set<OpHandleBase *> AllOps() const;

  const std::unordered_set<OpHandleBase *> &PendingOps(OpHandleBase *op) const;

  bool HasOp(OpHandleBase *op) const;

  // Use a visitor to visit all pending ops of op
  // Stop when callback returns false
  template <typename Callback>
  bool VisitAllPendingOps(OpHandleBase *op, Callback &&callback) const;

 private:
  void Build(const std::vector<OpHandleBase *> &ops);
  void EnforceHasOp(OpHandleBase *op) const;

  std::unordered_map<OpHandleBase *, std::unordered_set<OpHandleBase *>>
      preceding_ops_;
  std::unordered_map<OpHandleBase *, std::unordered_set<OpHandleBase *>>
      pending_ops_;
};

template <typename Callback>
bool OpGraphView::VisitAllPendingOps(OpHandleBase *op,
                                     Callback &&callback) const {
  EnforceHasOp(op);
  std::unordered_set<OpHandleBase *> visited;
  std::queue<OpHandleBase *> q;
  q.push(op);
  do {
    op = q.front();
    q.pop();
    for (auto &pending_op : pending_ops_.at(op)) {
      if (visited.count(pending_op) == 0) {
        visited.insert(pending_op);
        if (!callback(pending_op)) {
          return false;
        }
        q.push(pending_op);
      }
    }
  } while (!q.empty());
  return true;
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
