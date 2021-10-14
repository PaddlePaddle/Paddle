// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <algorithm>
#include "paddle/fluid/framework/details/eager_deletion_op_handle.h"
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/details/op_handle_base.h"
#include "paddle/fluid/framework/details/var_handle.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/memory_optimize_pass/op_graph_view.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/string/string_helper.h"

namespace paddle {
namespace framework {
namespace ir {

static std::string kSep(1, static_cast<char>(1));  // NOLINT

// NOTE: VariableNameMap is sorted!
static std::string VarNameMapToString(const VariableNameMap &var_map) {
  std::vector<std::string> tmp_strs;
  tmp_strs.reserve(var_map.size());
  for (auto &pair : var_map) {
    auto str = pair.first + kSep + string::join_strings(pair.second, kSep);
    tmp_strs.emplace_back(std::move(str));
  }
  return string::join_strings(tmp_strs, kSep);
}

static std::string OpDescToString(const OpDesc &op) {
  return "OpDesc" + kSep + op.Type() + kSep + VarNameMapToString(op.Inputs()) +
         kSep + VarNameMapToString(op.Outputs());
}

static std::string VarHandleListToString(
    const std::vector<details::VarHandleBase *> &vars) {
  std::vector<std::string> valid_vars;
  valid_vars.reserve(vars.size());
  for (auto *v : vars) {
    auto *valid_var = dynamic_cast<details::VarHandle *>(v);
    if (valid_var != nullptr) {
      valid_vars.emplace_back(valid_var->Name());
    }
  }
  std::sort(valid_vars.begin(), valid_vars.end());
  return string::join_strings(valid_vars, kSep);
}

static std::string EagerDeletionOpHandleToString(
    const details::EagerDeletionOpHandle &op);
static std::string OpHandleToString(const details::OpHandleBase &op);

static std::string EagerDeletionOpHandleToString(
    const details::EagerDeletionOpHandle &op) {
  auto vars_to_delete = op.VarsToDelete();
  std::unordered_set<details::OpHandleBase *> prev_ops;
  std::vector<std::string> prev_op_strs;
  prev_op_strs.reserve(op.Inputs().size());
  for (auto *var : op.Inputs()) {
    auto *prev_op = var->GeneratedOp();
    if (prev_op == nullptr) continue;
    prev_op_strs.push_back(OpHandleToString(*prev_op));
  }
  std::sort(prev_op_strs.begin(), prev_op_strs.end());
  // NOTE: gc op does not have any valid input/output vars
  return "OpHandleBase" + kSep + op.Name() + kSep +
         string::join_strings(vars_to_delete, kSep) + kSep +
         string::join_strings(prev_op_strs, kSep);
}

static std::string OpHandleToString(const details::OpHandleBase &op) {
  // NOTE: gc op does not have any valid input/output vars
  auto gc_op = dynamic_cast<const details::EagerDeletionOpHandle *>(&op);
  if (gc_op) {
    return EagerDeletionOpHandleToString(*gc_op);
  }
  return "OpHandleBase" + kSep + op.Name() + kSep +
         VarHandleListToString(op.Inputs()) + kSep +
         VarHandleListToString(op.Outputs());
}

static void AddSequentialDepsForSortedOps(
    Graph *graph, const std::vector<details::OpHandleBase *> &sorted_ops) {
  size_t n = sorted_ops.size();
  for (size_t i = 1; i < n; ++i) {
    auto *prev_op = sorted_ops[i - 1];
    auto *cur_op = sorted_ops[i];
    auto *dep_var = new details::DummyVarHandle(graph->CreateControlDepVar());
    graph->Get<details::GraphDepVars>(details::kGraphDepVars).emplace(dep_var);
    prev_op->AddOutput(dep_var);
    cur_op->AddInput(dep_var);
  }
}

class FixOpRunOrderPass : public Pass {
 protected:
  void ApplyImpl(Graph *graph) const override {
    const auto &program = graph->OriginProgram();
    std::unordered_map<std::string, size_t> op_to_idx;
    size_t i = 0;
    for (auto *op_desc : program.Block(0).AllOps()) {
      auto op_desc_str = OpDescToString(*op_desc);
      PADDLE_ENFORCE_EQ(
          op_to_idx.emplace(op_desc_str, i).second, true,
          platform::errors::PermissionDenied(
              "FixOpRunOrderPass cannot handle OpDesc with same "
              "type, inputs and outputs yet, error string repr: %s",
              op_desc_str));
      ++i;
    }

    // a map to record: "Node" -> "Node Index"
    std::unordered_map<Node *, size_t> node_to_idx;
    // a map to record found "Node Index"
    std::unordered_set<size_t> found_node_indices;
    // a map to record the new OpDesc created by other Passes. These ops does
    // not exist in the origin program
    std::map<std::string, Node *> new_op_desc_nodes;
    // a map to record the new OpHandle created by other Passes. These ops does
    // not have OpDesc and does not exist in the origin program
    std::map<std::string, Node *> new_op_handle_nodes;

    // Step 1: handle the unchanged OpDesc, and record new OpDesc/OpHandle
    auto op_handles = FilterByNodeWrapper<details::OpHandleBase>(*graph);
    for (auto *op_handle : op_handles) {
      auto *node = op_handle->Node();
      if (node->Op() == nullptr) {
        auto node_str = OpHandleToString(*op_handle);
        PADDLE_ENFORCE_EQ(new_op_handle_nodes.emplace(node_str, node).second,
                          true,
                          platform::errors::PermissionDenied(
                              "FixOpRunOrderPass cannot OpHandle with same "
                              "inputs and outputs yet, error repr: %s",
                              node_str));
        continue;
      }

      auto node_str = OpDescToString(*(node->Op()));
      auto iter = op_to_idx.find(node_str);
      if (iter != op_to_idx.end()) {
        size_t idx = iter->second;
        PADDLE_ENFORCE_EQ(
            found_node_indices.count(idx), 0,
            platform::errors::PermissionDenied(
                "FixOpRunOrderPass cannot handle OpDesc with same "
                "type, inputs and outputs yet, error repr: %s",
                node_str));
        found_node_indices.insert(idx);
        node_to_idx[node] = idx;
      } else {
        PADDLE_ENFORCE_EQ(
            new_op_desc_nodes.emplace(node_str, node).second, true,
            platform::errors::PermissionDenied(
                "FixOpRunOrderPass cannot handle OpDesc with same "
                "type, inputs and outputs yet, error repr: %s",
                node_str));
      }
    }

    VLOG(10) << "Found unchanged OpDesc " << node_to_idx.size()
             << ", new OpDesc " << new_op_desc_nodes.size() << ", new OpHandle "
             << new_op_handle_nodes.size();

    // Step 2: assign node index to new OpDesc
    size_t node_id_offset = op_to_idx.size();
    for (auto &pair : new_op_desc_nodes) {
      node_to_idx[pair.second] = node_id_offset;
      ++node_id_offset;
    }

    // Step 3: assign node index to new OpHandle
    for (auto &pair : new_op_handle_nodes) {
      node_to_idx[pair.second] = node_id_offset;
      ++node_id_offset;
    }

    // Step 4: sort unchanged OpDesc/new OpDesc/new OpHandle by topological
    // order and node index
    OpGraphView graph_view(op_handles);
    auto comp = [&node_to_idx](details::OpHandleBase *op1,
                               details::OpHandleBase *op2) {
      auto priority1 = static_cast<int>(op1->GetPriority());
      auto priority2 = static_cast<int>(op2->GetPriority());
      if (priority1 != priority2) {
        return priority1 < priority2;
      }
      return node_to_idx.at(op1->Node()) < node_to_idx.at(op2->Node());
    };

    std::vector<details::OpHandleBase *> sorted_ops;
    sorted_ops.reserve(op_handles.size());
    std::queue<details::OpHandleBase *> q;
    std::vector<details::OpHandleBase *> tmp_ops;
    auto op_deps = graph_view.GetPrecedingDepNum();
    // Get ready ops first
    for (auto iter = op_deps.begin(); iter != op_deps.end();) {
      if (iter->second != 0) {
        ++iter;
        continue;
      }
      tmp_ops.push_back(iter->first);
      op_deps.erase(iter++);
    }
    // Sort ready ops by node index
    std::sort(tmp_ops.begin(), tmp_ops.end(), comp);
    for (auto *op : tmp_ops) {
      q.push(op);
    }
    while (!q.empty()) {
      auto *cur_op = q.front();
      q.pop();
      sorted_ops.push_back(cur_op);

      auto &pending_ops = graph_view.PendingOps(cur_op);
      tmp_ops.clear();
      for (auto *pending_op : pending_ops) {
        if (--op_deps.at(pending_op) == 0) {
          op_deps.erase(pending_op);
          tmp_ops.push_back(pending_op);
        }
      }
      // sort next ready ops by node index
      std::sort(tmp_ops.begin(), tmp_ops.end(), comp);
      for (auto *op : tmp_ops) {
        q.push(op);
      }
    }

    PADDLE_ENFORCE_EQ(
        sorted_ops.size(), op_handles.size(),
        platform::errors::PermissionDenied("There are unvisited ops"));
    if (VLOG_IS_ON(10)) {
      // print op order to debug
      std::vector<size_t> sorted_ops_indices;
      sorted_ops_indices.reserve(sorted_ops.size());
      for (auto *op : sorted_ops) {
        sorted_ops_indices.push_back(node_to_idx.at(op->Node()));
      }
      VLOG(10) << "Fix op order: "
               << string::join_strings(sorted_ops_indices, ',');
    }

    // Step 5: add sequential deps for ops to guarantee there is only one
    // toposort order
    AddSequentialDepsForSortedOps(graph, sorted_ops);
    PADDLE_ENFORCE_EQ(IsTopologySortOperationsUnique(*graph), true,
                      platform::errors::PermissionDenied(
                          "The topological order must be unique "
                          "after FixOpRunOrderPass is applied"));
  }
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fix_op_run_order_pass, paddle::framework::ir::FixOpRunOrderPass);
