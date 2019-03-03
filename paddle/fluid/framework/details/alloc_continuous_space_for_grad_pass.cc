//   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include <string>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/details/build_strategy.h"
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace framework {
namespace details {

namespace {
framework::proto::VarType::Type kDefaultDtype =
    framework::proto::VarType::Type::VarType_Type_BOOL;
}

class AllocContinuousSpaceForGradPass : public ir::Pass {
 protected:
  std::unique_ptr<ir::Graph> ApplyImpl(
      std::unique_ptr<ir::Graph> graph) const override {
    ir::Graph &result = *graph;

    auto &places = Get<const std::vector<platform::Place>>(kPlaces);
    auto &local_scopes = Get<const std::vector<Scope *>>(kLocalScopes);
    ResetAttribute<ParamsAndGrads>(kParamsAndGrads, &result);

    // NOTE: The operator nodes should be in topology order.
    std::vector<ir::Node *> topo_nodes = ir::TopologySortOperations(result);
    auto &params_grads = result.Get<ParamsAndGrads>(kParamsAndGrads);
    for (auto &node : topo_nodes) {
      RecordParamsAndGrads(node, &params_grads);
    }

    if (params_grads.size() == 0) {
      VLOG(10) << "Doesn't find gradients";
      return std::move(graph);
    }

    std::unordered_map<std::string, ir::Node *> vars;
    for (ir::Node *node : result.Nodes()) {
      if (node->IsVar() && node->Var()) {
        // Note: The graph may have the same name node. For example, parameter
        // is the input of operator and it also is the output of optimizer;
        vars.emplace(node->Var()->Name(), node);
      }
    }

    // Note: Maybe the order of the parameters and gradient should be changed.

    // Set Gradients as Persistable to prevent this var becoming reusable.
    auto dtype = kDefaultDtype;
    for (auto &p_g : params_grads) {
      // Get gradient var
      auto iter = vars.find(p_g.second);
      PADDLE_ENFORCE(iter != vars.end(), "%s is not found.", p_g.second);
      iter->second->Var()->SetPersistable(true);

      PADDLE_ENFORCE(IsSupportedVarType(iter->second->Var()->GetType()));

      // Get Dtype
      auto ele_dtype = iter->second->Var()->GetDataType();
      if (dtype == kDefaultDtype) {
        dtype = ele_dtype;
        PADDLE_ENFORCE_NE(ele_dtype, kDefaultDtype);
      }
      PADDLE_ENFORCE_EQ(ele_dtype, dtype);
    }

    // Create a FusedVars Set to avoid duplicating names for fused_var in other
    // pass.
    if (!result.Has(kFusedVars)) {
      result.Set(kFusedVars, new FusedVars);
    }
    // the kFusedGrads is used be fuse_optimizer_op_pass.
    if (!result.Has(kFusedGrads)) {
      result.Set(kFusedGrads, new FusedGrads);
    }
    // the fused_var_name should be unique.
    const std::string prefix(kFusedVarNamePrefix);
    auto fused_var_name = prefix + "@GRAD@" + params_grads.begin()->second;
    result.Get<FusedGrads>(kFusedGrads) = fused_var_name;
    auto &fused_var_set = result.Get<FusedVars>(kFusedVars);
    PADDLE_ENFORCE_EQ(fused_var_set.count(fused_var_name), 0,
                      "%s is duplicate in FusedVars.", fused_var_name);
    fused_var_set.insert(fused_var_name);

    InitFusedVarsAndAllocSpaceForVars(places, local_scopes, vars,
                                      fused_var_name, params_grads);

    return std::move(graph);
  }

  template <typename AttrType>
  void ResetAttribute(const std::string &attr_name, ir::Graph *graph) const {
    if (graph->Has(attr_name)) {
      VLOG(10) << attr_name << " is reset.";
      graph->Erase(attr_name);
    }
    graph->Set(attr_name, new AttrType);
  }

 private:
  bool IsSupportedVarType(const proto::VarType::Type &type) const {
    // Current only support LOD_TENSOR.
    return type == proto::VarType::LOD_TENSOR;
  }

  void AppendAllocSpaceForVarsOp(const std::vector<std::string> &params_name,
                                 const std::vector<std::string> &grads_name,
                                 const std::string &fused_var_name,
                                 BlockDesc *global_block) const {
    auto op_desc = global_block->AppendOp();
    op_desc->SetType("alloc_continuous_space");
    op_desc->SetInput("Input", params_name);
    op_desc->SetOutput("Output", grads_name);
    op_desc->SetOutput("FusedOutput", {fused_var_name});
  }

  void RecordParamsAndGrads(ir::Node *node,
                            ParamsAndGrads *params_grads) const {
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

      for (size_t i = 0; i < backward_vars.size(); i += 2) {
        VLOG(10) << "Trainable parameter: " << backward_vars[i]
                 << ", gradient: " << backward_vars[i + 1];

        params_grads->emplace_back(std::make_pair(
            backward_vars[i] /*param*/, backward_vars[i + 1] /*grad*/));
      }
    } catch (boost::bad_get e) {
    }
  }

  void InitFusedVarsAndAllocSpaceForVars(
      const std::vector<platform::Place> &places,
      const std::vector<Scope *> &local_scopes,
      const std::unordered_map<std::string, ir::Node *> &vars,
      const std::string &fused_var_name,
      const ParamsAndGrads &params_grads) const {
    // Init Gradients and FusedVars.
    VLOG(10) << "Init FusedVars.";
    // Alloc parameters and auxiliary vars in the respective scope.
    size_t idx = local_scopes.size();
    for (auto iter = local_scopes.rbegin(); iter != local_scopes.rend();
         ++iter, --idx) {
      auto &scope = *iter;
      VLOG(10) << "Find var in scope " << idx << " " << fused_var_name;
      PADDLE_ENFORCE(scope->FindVar(fused_var_name) == nullptr,
                     "%s has exist in scope[%d]", fused_var_name, idx);
      scope->Var(fused_var_name)->GetMutable<LoDTensor>();
    }

    VLOG(10) << "Init Gradients.";
    for (auto iter = local_scopes.rbegin(); iter != local_scopes.rend();
         ++iter) {
      auto &scope = *iter;
      for (auto &p_g : params_grads) {
        auto var_iter = vars.find(p_g.second);
        PADDLE_ENFORCE(var_iter != vars.end());
        PADDLE_ENFORCE_NOT_NULL(var_iter->second->Var());
        PADDLE_ENFORCE_EQ(var_iter->second->Var()->GetType(),
                          proto::VarType::LOD_TENSOR);
        scope->Var(p_g.second)->GetMutable<LoDTensor>();
      }
    }

    // Alloc continuous space for vars.
    std::vector<std::string> grads_name;
    std::vector<std::string> params_name;
    grads_name.reserve(params_grads.size());
    params_name.reserve(params_grads.size());
    for (auto &p_g : params_grads) {
      params_name.emplace_back(p_g.first);
      grads_name.emplace_back(p_g.second);
    }
    framework::ProgramDesc program_desc;
    AppendAllocSpaceForVarsOp(params_name, grads_name, fused_var_name,
                              program_desc.MutableBlock(0));

    for (size_t i = 0; i < local_scopes.size(); ++i) {
      for (auto &op_desc : program_desc.Block(0).AllOps()) {
        auto op = OpRegistry::CreateOp(*op_desc);
        op->Run(*local_scopes[i], places[i]);
      }
    }
  }
};

}  // namespace details
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(alloc_continuous_space_for_grad_pass,
              paddle::framework::details::AllocContinuousSpaceForGradPass)
    .RequirePassAttr(paddle::framework::details::kPlaces)
    .RequirePassAttr(paddle::framework::details::kLocalScopes);
