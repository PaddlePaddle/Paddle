//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/details/ssa_graph.h"
#include <deque>

namespace paddle {
namespace framework {
namespace details {

size_t SSAGraph::GraphNumber() const {
  // Collect ready vars
  std::unordered_set<VarHandleBase *> ready_vars;
  auto collect_ready_var = [&](VarHandleBase *var) {
    if (var->generated_op_ == nullptr) {
      ready_vars.emplace(var);
    }
  };

  for (auto &var_map : this->vars_) {
    for (auto &name_pair : var_map) {
      for (auto &version_pair : name_pair.second) {
        collect_ready_var(version_pair.get());
      }
    }
  }

  for (auto &var : this->dep_vars_) {
    collect_ready_var(var.get());
  }

  // Collect need visit Op
  std::deque<OpHandleBase *> need_visite_ops;
  std::unordered_set<OpHandleBase *> has_visited_ops;

  auto collect_unvisit_op = [&has_visited_ops,
                             &need_visite_ops](OpHandleBase *op) {
    for (auto &out_var : op->Outputs()) {
      for (auto pend_op : out_var->pending_ops_) {
        if (has_visited_ops.count(pend_op) == 0) {
          need_visite_ops.push_back(pend_op);
        }
      }
    }
    for (auto &in_var : op->Inputs()) {
      auto gen_op = in_var->generated_op_;
      if (gen_op && has_visited_ops.count(gen_op) == 0) {
        need_visite_ops.push_back(gen_op);
      }
    }
  };

  std::vector<std::unordered_set<OpHandleBase *>> vec_thiss;
  std::unordered_set<VarHandleBase *> visited_vars;
  auto ready_var_iter = ready_vars.begin();

  while (has_visited_ops.size() != this->ops_.size()) {
    VarHandleBase *unvisited_var = nullptr;
    for (; ready_var_iter != ready_vars.end(); ++ready_var_iter) {
      if (visited_vars.count(*ready_var_iter) == 0) {
        visited_vars.emplace(*ready_var_iter);
        unvisited_var = *ready_var_iter;
        break;
      }
    }

    if (unvisited_var == nullptr || unvisited_var->pending_ops_.empty()) {
      PADDLE_THROW("This Graph has some error.");
    }

    need_visite_ops.clear();

    for (auto op : unvisited_var->pending_ops_) {
      if (has_visited_ops.count(op) == 0) {
        need_visite_ops.push_back(op);
        collect_unvisit_op(op);
      }
    }

    std::unordered_set<OpHandleBase *> cur_this;
    while (!need_visite_ops.empty()) {
      auto op = need_visite_ops.front();
      need_visite_ops.pop_front();

      has_visited_ops.emplace(op);
      cur_this.emplace(op);
      collect_unvisit_op(op);
    }

    if (cur_this.size() > 0) {
      vec_thiss.emplace_back(cur_this);
    }
  }
  return vec_thiss.size();
}
}  // namespace details
}  // namespace framework
}  // namespace paddle
