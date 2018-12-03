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

#include <queue>
#include <string>
#include <type_traits>
#include <vector>

#include "paddle/fluid/framework/details/computation_op_handle.h"
#include "paddle/fluid/framework/details/eager_deletion_op_handle.h"
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/details/op_graph_view.h"
#include "paddle/fluid/framework/details/reference_count_pass.h"
#include "paddle/fluid/framework/details/reference_count_pass_helper.h"
#include "paddle/fluid/framework/ir/graph_helper.h"

namespace paddle {
namespace framework {
namespace details {

struct OpConnectionDetector {
 public:
  enum RelationShip { kSame = 0, kNoDeps = 1, kBefore = 2, kAfter = 3 };

  explicit OpConnectionDetector(const std::vector<OpHandleBase *> &all_ops)
      : graph_(all_ops) {}

  template <typename OpSet>
  std::unordered_set<typename OpSet::key_type> MaxNoDepOps(
      const OpSet &op_set) {
    using KeyType = typename OpSet::key_type;
    static_assert(
        std::is_base_of<OpHandleBase,
                        typename std::remove_pointer<KeyType>::type>::value,
        "Key type of OpSet must be or derived of OpHandleBase");

    std::vector<OpHandleBase *> ops(op_set.begin(), op_set.end());
    std::unordered_set<KeyType> ret;
    auto rels = GetRelations(ops);
    auto not_before = [](RelationShip r) { return r != kBefore; };
    for (size_t i = 0; i < rels.size(); ++i) {
      if (std::all_of(rels[i].begin(), rels[i].end(), not_before)) {
        ret.insert(static_cast<KeyType>(ops[i]));
      }
    }
    return ret;
  }

 private:
  std::vector<std::vector<RelationShip>> GetRelations(
      const std::vector<OpHandleBase *> ops) {
    std::unordered_map<OpHandleBase *, size_t> op_to_idx;
    for (size_t i = 0; i < ops.size(); ++i) {
      PADDLE_ENFORCE(graph_.HasOp(ops[i]), "Op does not exist in graph");
      op_to_idx[ops[i]] = i;
    }

    PADDLE_ENFORCE(op_to_idx.size() == ops.size(), "Duplicate ops");

    std::vector<std::vector<RelationShip>> ret(ops.size());
    for (auto &e : ret) {
      e.assign(ops.size(), kSame);
    }

    size_t found_num = ops.size();
    size_t total_num = ops.size() * ops.size();
    auto visitor = [&](OpHandleBase *op, size_t i) {
      auto it = op_to_idx.find(op);
      if (it != op_to_idx.end()) {
        size_t j = it->second;
        if (ret[i][j] != kSame) {
          ret[i][j] = kBefore;
          ret[j][i] = kAfter;
          found_num += 2;
          if (found_num == total_num) {
            return false;
          }
        }
      }
      return true;
    };

    for (size_t i = 0; i < ops.size(); ++i) {
      auto sub_visitor = [&, i](OpHandleBase *op) { return visitor(op, i); };
      if (!graph_.VisitAllPendingOps(ops[i], sub_visitor)) {
        break;
      }
    }

    for (size_t i = 0; i < ops.size(); ++i) {
      for (size_t j = i + 1; j < ops.size(); ++j) {
        if (ret[i][j] != kSame) continue;
        ret[i][j] = kNoDeps;
        ret[j][i] = kNoDeps;
      }
    }

    return ret;
  }

  const OpGraphView graph_;
};

static ComputationOpHandle *FindNextComputationOpHandleOrReturnItself(
    OpHandleBase *op, size_t scope_idx) {
  std::queue<OpHandleBase *> q;
  std::unordered_set<OpHandleBase *> visited;
  q.push(op);
  do {
    auto *op = q.front();
    q.pop();
    auto *compute_op = dynamic_cast<ComputationOpHandle *>(op);
    if (compute_op != nullptr && compute_op->GetScopeIdx() == scope_idx) {
      return compute_op;
    }
    for (auto *out_var : op->Outputs()) {
      for (auto *pending_op : out_var->PendingOps()) {
        if (visited.count(pending_op)) continue;
        visited.insert(pending_op);
      }
    }
  } while (!q.empty());
  return nullptr;
}

std::unique_ptr<ir::Graph> ReferenceCountPass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  auto &vars = graph->Get<GraphVars>(kGraphVars);
  auto &ref_cnts = Get<std::vector<ReferenceCountMap>>(kGlobalReferenceCount);
  auto &last_live_ops_of_vars =
      Get<std::vector<LastLiveOpsOfVars>>(kLastLiveOpsOfVars);

  last_live_ops_of_vars = std::vector<LastLiveOpsOfVars>(vars.size());
  ref_cnts = std::vector<ReferenceCountMap>(vars.size());

  OpConnectionDetector detector(ir::FilterByNodeWrapper<OpHandleBase>(*graph));

  for (size_t i = 0; i < vars.size(); ++i) {
    for (auto &name_var_pair : vars[i]) {
      if (name_var_pair.second.empty()) {
        continue;
      }

      const std::string &var_name = name_var_pair.first;
      auto *last_ver_var = name_var_pair.second.back();

      VarDesc *var_desc = nullptr;
      std::find_if(name_var_pair.second.rbegin(), name_var_pair.second.rend(),
                   [&](VarHandle *var_handle) -> bool {
                     var_desc = var_handle->Node()->Var();
                     return var_desc != nullptr;
                   });

      if (var_desc == nullptr || var_desc->Persistable()) {
        continue;
      }

      auto var_type = var_desc->Proto()->type().type();
      if (var_type != proto::VarType::LOD_TENSOR &&
          var_type != proto::VarType::SELECTED_ROWS &&
          var_type != proto::VarType::LOD_TENSOR_ARRAY) {
        continue;
      }

      std::unordered_set<ComputationOpHandle *> last_live_op;
      auto add_last_live_op = [&](OpHandleBase *op) -> bool {
        auto *compute_op = FindNextComputationOpHandleOrReturnItself(op, i);
        if (compute_op) {
          last_live_op.insert(compute_op);
          return true;
        } else {
          return false;
        }
      };

      bool can_delete = false;
      auto &pending_ops = last_ver_var->PendingOps();
      if (pending_ops.empty()) {
        auto *generated_op = last_ver_var->GeneratedOp();
        if (generated_op && add_last_live_op(generated_op)) {
          can_delete = true;
        }
      } else {
        can_delete = true;
        for (auto *pending_op : pending_ops) {
          if (!add_last_live_op(pending_op)) {
            can_delete = false;
            break;
          }
        }
      }

      if (can_delete) {
        size_t original_size = last_live_op.size();
        last_live_op = detector.MaxNoDepOps(last_live_op);
        if (last_live_op.size() != original_size) {
          VLOG(10) << "Shrink last living op number of " << var_name << " from "
                   << original_size << " to " << last_live_op.size();
        }
        ref_cnts[i].emplace(var_name, last_live_op.size());
        last_live_ops_of_vars[i].emplace(var_name, std::move(last_live_op));
      }
    }
  }

  return graph;
}

}  // namespace details
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(reference_count_pass,
              paddle::framework::details::ReferenceCountPass)
    .RequirePassAttr(paddle::framework::details::kGlobalReferenceCount)
    .RequirePassAttr(paddle::framework::details::kLastLiveOpsOfVars);
