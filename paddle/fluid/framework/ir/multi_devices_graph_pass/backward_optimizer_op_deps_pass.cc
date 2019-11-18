// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/details/container_cast.h"
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/details/op_handle_base.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/scope.h"

namespace paddle {
namespace framework {
namespace ir {

class BackWardOpDepsPass : public ir::Pass {
 protected:
  void AddDep(ir::Graph* graph, details::OpHandleBase* l,
              details::OpHandleBase* r) const {
    auto* dep_var = new details::DummyVarHandle(graph->CreateControlDepVar());
    graph->Get<details::GraphDepVars>(details::kGraphDepVars).emplace(dep_var);
    l->AddOutput(dep_var);
    r->AddInput(dep_var);
    VLOG(10) << "add deps:" << l->DebugString() << " and " << r->DebugString();
  }

  void ApplyImpl(ir::Graph* graph) const override {
    // NOTE: The operator nodes should be in topology order.
    std::vector<details::OpHandleBase*> backward_op_handles;
    std::vector<details::OpHandleBase*> all_opt_handles;
    details::ParamsAndGrads params_grads;
    std::vector<ir::Node*> topo_nodes = ir::TopologySortOperations(*graph);
    for (auto& node : topo_nodes) {
      if (!node->Op()) continue;

      GetBackWardOpHandles(node, &backward_op_handles, &params_grads);
      GetOptimizerOpHandles(node, &all_opt_handles);
    }

    VLOG(10) << "backward_op_handles size:" << backward_op_handles.size()
             << ", opt_handles size:" << all_opt_handles.size();

    if (backward_op_handles.size() <= 1 || all_opt_handles.size() <= 1) {
      VLOG(10) << "need not backward_op_deps_pass";
      return;
    }

    std::vector<details::OpHandleBase*> opt_handles;
    GetOptimizerHandlesRoot(all_opt_handles, &opt_handles, params_grads);

    if (opt_handles.size() <= 1) {
      VLOG(10) << "need not backward_op_deps_pass";
      return;
    }

    VLOG(10) << "add optimize deps";
    for (size_t i = 1; i < opt_handles.size(); ++i) {
      AddDep(graph, opt_handles[i - 1], opt_handles[i]);
    }

    VLOG(10) << "add deps between backward and optimze:";
    AddDep(graph, backward_op_handles[backward_op_handles.size() - 1],
           opt_handles[0]);
  }

  /*
   * When the backward ophandles complete, the optimizer ophandle's inputs var
   * are ready.Since the optimizer ophandles can be seen as graphs which each of
   * them doesn't connect to each other, they can run parallelly or by a
   * specified order, such as by the grads generated order. This function will
   * get these graphs' root.
   */
  void GetOptimizerHandlesRoot(
      const std::vector<details::OpHandleBase*>& ops,
      std::vector<details::OpHandleBase*>* result,
      const details::ParamsAndGrads& params_grads) const {
    std::unordered_set<details::OpHandleBase*> visit;
    for (auto op : ops) {
      if (visit.find(op) != visit.end()) {
        continue;
      }

      VLOG(10) << "visiting all_opt_handles:" << op->DebugString();

      result->emplace_back(op);
      visit.insert(op);
      VisitChildrens(op, &visit);
    }

    for (size_t i = 0; i < result->size(); i++) {
      VLOG(10) << "get potential head op:" << (*result)[i]->DebugString();
    }

    // sort by param_grad order
    std::unordered_map<std::string, int> pg_order;
    int order = 0;
    for (auto& p_g : params_grads) {
      pg_order[p_g.second] = order++;
    }

    std::vector<std::pair<details::OpHandleBase*, int>> op_handles;
    for (auto op : *result) {
      int order = 0;
      for (auto input : op->Inputs()) {
        if (dynamic_cast<details::VarHandle*>(input) == nullptr) continue;

        if (pg_order.find(input->Name()) == pg_order.end()) {
          VLOG(10) << "not find input " << input->Name() << " in grad";
          continue;
        }

        if (order < pg_order.at(input->Name())) {
          order = pg_order.at(input->Name());
        }
      }
      op_handles.emplace_back(std::make_pair(op, order));
    }

    sort(op_handles.begin(), op_handles.end(),
         [](const std::pair<details::OpHandleBase*, int>& left,
            const std::pair<details::OpHandleBase*, int>& right) -> bool {
           return left.second < right.second;
         });

    result->clear();
    for (auto p : op_handles) {
      result->emplace_back(p.first);
    }

    for (size_t i = 0; i < result->size(); i++) {
      VLOG(10) << "get head op:" << (*result)[i]->DebugString();
    }
  }

  void VisitChildrens(details::OpHandleBase* op,
                      std::unordered_set<details::OpHandleBase*>* visit) const {
    for (auto out : op->Outputs()) {
      for (auto* pending_op : out->PendingOps()) {
        if (visit->find(pending_op) != visit->end()) {
          continue;
        }

        VLOG(10) << "visiting:" << pending_op->DebugString();

        visit->insert(pending_op);
        VisitChildrens(pending_op, visit);
      }
    }
  }

  void GetBackWardOpHandles(
      ir::Node* node, std::vector<details::OpHandleBase*>* backward_op_handles,
      details::ParamsAndGrads* params_grads) const {
    auto& op_desc = *(node->Op());
    bool is_bk_op = details::IsOpRole(op_desc, OpRole::kBackward);
    if (!is_bk_op) return;

    // Currently, we assume that once gradient is generated, it can be
    // broadcast, and each gradient is only broadcast once.
    auto backward_vars = details::GetOpRoleVarsOrEmpty(op_desc);
    PADDLE_ENFORCE_EQ(node->IsWrappedBy<details::OpHandleBase>(), true,
                      platform::errors::InvalidArgument(
                          "Node must be wrapped by OpHandleBase"));

    backward_op_handles->emplace_back(&node->Wrapper<details::OpHandleBase>());

    for (size_t i = 0; i < backward_vars.size(); i += 2) {
      VLOG(10) << "Trainable parameter: " << backward_vars[i]
               << ", gradient: " << backward_vars[i + 1];

      params_grads->emplace_back(std::make_pair(backward_vars[i] /*param*/,
                                                backward_vars[i + 1] /*grad*/));
    }
  }

  void GetOptimizerOpHandles(
      ir::Node* node, std::vector<details::OpHandleBase*>* opt_handles) const {
    if (details::IsOpRole(*(node->Op()), OpRole::kOptimize)) {
      opt_handles->emplace_back(&node->Wrapper<details::OpHandleBase>());
    }
  }
};
}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(backward_optimizer_op_deps_pass,
              paddle::framework::ir::BackWardOpDepsPass);
