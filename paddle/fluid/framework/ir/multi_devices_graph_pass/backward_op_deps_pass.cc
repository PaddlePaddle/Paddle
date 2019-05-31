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
    std::vector<ir::Node*> opt_nodes;
    std::vector<ir::Node*> topo_nodes = ir::TopologySortOperations(*graph);
    for (auto& node : topo_nodes) {
      // node->Wrapper<details::OpHandleBase>().DebugString();
      if (!node->Op()) continue;

      GetBackWardOpHandles(node, &backward_op_handles);
      GetOptHandles(node, &opt_nodes);
    }

    VLOG(10) << "backward_op_handles size:" << backward_op_handles.size()
             << ", opt_handles size:" << opt_nodes.size();
    if (backward_op_handles.size() <= 1 || opt_nodes.size() <= 1) {
      VLOG(10) << "need not backward_op_deps_pass";
      return;
    }

    VLOG(10) << "add backward deps";
    for (size_t i = 1; i < backward_op_handles.size(); ++i) {
      AddDep(graph, backward_op_handles[i - 1], backward_op_handles[i]);
    }

    std::vector<details::OpHandleBase*> opt_handles;
    TravelOptGraph(opt_nodes, &opt_handles);
    PADDLE_ENFORCE_EQ(opt_nodes.size(), opt_handles.size());

    VLOG(10) << "add optimize deps";
    for (size_t i = 1; i < opt_handles.size(); ++i) {
      AddDep(graph, opt_handles[i - 1], opt_handles[i]);
    }

    VLOG(10) << "add deps between backward and optimze:";
    AddDep(graph, backward_op_handles[backward_op_handles.size() - 1],
           opt_handles[0]);
  }

  void TravelOptGraph(const std::vector<ir::Node*>& nodes,
                      std::vector<details::OpHandleBase*>* handles) const {
    std::unordered_set<ir::Node*> v_set;
    std::vector<std::vector<ir::Node*>> result;
    for (auto n : nodes) {
      int order = 0;
      // auto& arr = result.emplace();
      std::vector<ir::Node*> arr;
      VisitNode(n, &v_set, &arr, &order);
      result.emplace_back(arr);
    }

    int idx = 0;
    for (auto& a : result) {
      idx++;
      if (a.size() == 0) continue;

      VLOG(10) << "depth first no:" << idx << ", size:" << a.size();

      for (auto& n : a) {
        handles->emplace_back(&n->Wrapper<details::OpHandleBase>());
        VLOG(10) << n->Wrapper<details::OpHandleBase>().DebugString();
      }
    }
  }
  void VisitNode(ir::Node* node, std::unordered_set<ir::Node*>* visit,
                 std::vector<ir::Node*>* v_arr, int* order) const {
    if (visit->find(node) != visit->end()) return;

    visit->insert(node);
    v_arr->emplace_back(node);

    if (order != 0 && node->inputs.size() > 1) {
      for (auto input : node->inputs) {
        if (visit->find(input) == visit->end()) {
          return;
        }
      }
    }

    (*order) += 1;
    for (auto o : node->outputs) {
      VisitNode(o, visit, v_arr, order);
    }
  }

  void GetBackWardOpHandles(
      ir::Node* node,
      std::vector<details::OpHandleBase*>* backward_op_handles) const {
    try {
      bool is_bk_op =
          static_cast<bool>(boost::get<int>(node->Op()->GetAttr(
                                OpProtoAndCheckerMaker::OpRoleAttrName())) &
                            static_cast<int>(OpRole::kBackward));
      if (!is_bk_op) return;

      // Currently, we assume that once gradient is generated, it can be
      // broadcast, and each gradient is only broadcast once.
      auto backward_vars =
          boost::get<std::vector<std::string>>(node->Op()->GetNullableAttr(
              OpProtoAndCheckerMaker::OpRoleVarAttrName()));
      PADDLE_ENFORCE_EQ(backward_vars.size() % 2, static_cast<size_t>(0));
      PADDLE_ENFORCE(node->IsWrappedBy<details::OpHandleBase>());

      backward_op_handles->emplace_back(
          &node->Wrapper<details::OpHandleBase>());
    } catch (boost::bad_get e) {
    }
  }

  void GetOptHandles(ir::Node* node,
                     std::vector<ir::Node*>* opt_handles) const {
    try {
      bool is_opt_op =
          static_cast<bool>(boost::get<int>(node->Op()->GetAttr(
                                OpProtoAndCheckerMaker::OpRoleAttrName())) &
                            static_cast<int>(OpRole::kOptimize));
      if (!is_opt_op) return;

      opt_handles->emplace_back(node);
    } catch (boost::bad_get e) {
    }
  }
};
}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(backward_op_deps_pass, paddle::framework::ir::BackWardOpDepsPass);
