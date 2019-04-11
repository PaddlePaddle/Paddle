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

#include "paddle/fluid/framework/details/op_graph_view.h"
#include <queue>
#include <utility>

namespace paddle {
namespace framework {
namespace details {

OpGraphView::OpGraphView(const std::vector<OpHandleBase *> &ops) { Build(ops); }

void OpGraphView::Build(const std::vector<OpHandleBase *> &ops) {
  preceding_ops_.clear();
  pending_ops_.clear();
  for (auto &op : ops) {
    preceding_ops_[op];
    pending_ops_[op];
    for (auto &var : op->Outputs()) {
      for (auto &pending_op : var->PendingOps()) {
        preceding_ops_[pending_op].insert(op);
        pending_ops_[op].insert(pending_op);
      }
    }
  }
  PADDLE_ENFORCE(
      preceding_ops_.size() == ops.size() && pending_ops_.size() == ops.size(),
      "There are duplicate ops in graph.");
}

std::unordered_set<OpHandleBase *> OpGraphView::AllOps() const {
  std::unordered_set<OpHandleBase *> ret;
  ret.reserve(preceding_ops_.size());
  for (auto &pair : preceding_ops_) {
    ret.insert(pair.first);
  }
  return ret;
}

bool OpGraphView::HasOp(OpHandleBase *op) const {
  return preceding_ops_.count(op) != 0;
}

void OpGraphView::EnforceHasOp(OpHandleBase *op) const {
  PADDLE_ENFORCE(HasOp(op), "Cannot find op %s in OpGraphView",
                 op == nullptr ? "nullptr" : op->DebugString());
}

const std::unordered_set<OpHandleBase *> &OpGraphView::PendingOps(
    OpHandleBase *op) const {
  EnforceHasOp(op);
  return pending_ops_.at(op);
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
