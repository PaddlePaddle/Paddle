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

#include "paddle/fluid/framework/ir/alloc_continuous_space_for_grad_pass.h"
#include <algorithm>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/details/build_strategy.h"
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/op_registry.h"

DEFINE_double(fuse_parameter_memory_size, -1.0,  // MBytes
              "fuse_parameter_memory_size is up limited memory size(MB)"
              "of one group parameters' gradient which is the input "
              "of communication calling(e.g NCCLAllReduce). "
              "The default value is 0, it means that "
              "not set group according to memory_size.");
DEFINE_int32(
    fuse_parameter_groups_size, 1,
    "fuse_parameter_groups_size is the up limited size of one group "
    "parameters' gradient. "
    "The default value is a experimental result. If the "
    "fuse_parameter_groups_size is 1, it means that the groups size is "
    "the number of parameters' gradient. If the fuse_parameter_groups_size is "
    "-1, it means that there are only one group. The default value is 3, it is "
    "an experimental value.");

namespace paddle {
namespace framework {
namespace ir {
// unit of the FLAGS_fuse_parameter_memory_size.
static constexpr double kMB = 1048576.0;

// SetFuseParameterGroupsSize and SetFuseParameterMemorySize are used in unit
// test, because it is invalid that seting 'FLAGS_fuse_parameter_memory_size'
// and 'FLAGS_fuse_parameter_groups_size' in unit test.
void SetFuseParameterGroupsSize(int group_size) {
  FLAGS_fuse_parameter_groups_size = group_size;
}

int GetFuseParameterGroupsSize() { return FLAGS_fuse_parameter_groups_size; }

void SetFuseParameterMemorySize(double memory_size) {
  FLAGS_fuse_parameter_memory_size = memory_size;
}

double GetFuseParameterMemorySize() { return FLAGS_fuse_parameter_memory_size; }

static framework::proto::VarType::Type kDefaultDtype =
    framework::proto::VarType::Type::VarType_Type_BOOL;

class AllocContinuousSpaceForGradPass : public ir::Pass {
 protected:
  void ApplyImpl(ir::Graph *graph) const {
    ir::Graph &result = *graph;

    auto &places = Get<const std::vector<platform::Place>>(details::kPlaces);
    auto &local_scopes = Get<const std::vector<Scope *>>(details::kLocalScopes);

    ResetAttribute<details::ParamsAndGrads>(details::kParamsAndGrads, &result);
    ResetAttribute<details::GroupGradsAndParams>(details::kGroupGradsAndParams,
                                                 &result);

    // NOTE: The operator nodes should be in topology order.
    std::vector<ir::Node *> topo_nodes = ir::TopologySortOperations(result);
    auto &params_grads =
        result.Get<details::ParamsAndGrads>(details::kParamsAndGrads);
    for (auto &node : topo_nodes) {
      RecordParamsAndGrads(node, &params_grads);
    }

    if (params_grads.size() == 0) {
      LOG(WARNING) << "Doesn't find gradients";
      return;
    }

    std::unordered_map<std::string, ir::Node *> vars;
    for (ir::Node *node : result.Nodes()) {
      if (node->IsVar() && node->Var()) {
        // Note: The graph may have the same name node. For example, parameter
        // is the input of operator and it also is the output of optimizer;
        vars.emplace(node->Var()->Name(), node);
      }
    }

    auto &group_grads_params =
        result.Get<details::GroupGradsAndParams>(details::kGroupGradsAndParams);

    // Note: the order of params_grads may be changed by SetGroupGradsAndParams.
    SetGroupGradsAndParams(vars, params_grads, &group_grads_params);

    params_grads.clear();
    for (auto &group_p_g : group_grads_params) {
      params_grads.insert(params_grads.begin(), group_p_g.begin(),
                          group_p_g.end());
    }
    for (auto &p_g : params_grads) {
      std::swap(p_g.first, p_g.second);
    }

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
        PADDLE_ENFORCE_NE(ele_dtype, kDefaultDtype,
                          "The data type should not be bool.");
      }
      PADDLE_ENFORCE_EQ(ele_dtype, dtype,
                        "The data type of input is not consistent.");
    }

    // Create a FusedVarsSet to avoid duplicating names for fused_var in other
    // pass.
    if (!result.Has(details::kFusedVars)) {
      result.Set(details::kFusedVars, new details::FusedVars);
    }
    // the kFusedGrads is used be fuse_optimizer_op_pass.
    result.Set(details::kFusedGrads, new details::FusedGrads);

    // the fused_var_name should be unique, so it appends
    // params_grads.begin()->second.
    auto fused_var_name = std::string(details::kFusedVarNamePrefix) + "@GRAD@" +
                          params_grads.begin()->second;
    result.Get<details::FusedGrads>(details::kFusedGrads) = fused_var_name;
    auto &fused_var_set = result.Get<details::FusedVars>(details::kFusedVars);
    PADDLE_ENFORCE_EQ(fused_var_set.count(fused_var_name), 0,
                      "%s is duplicate in FusedVars.", fused_var_name);
    fused_var_set.insert(fused_var_name);

    InitFusedVarsAndAllocSpaceForVars(places, local_scopes, vars,
                                      fused_var_name, params_grads);
  }

  template <typename AttrType>
  void ResetAttribute(const std::string &attr_name, ir::Graph *graph) const {
    if (graph->Has(attr_name)) {
      VLOG(10) << attr_name << " is reset.";
      graph->Erase(attr_name);
    }
    graph->Set(attr_name, new AttrType);
  }

  void SetGroupGradsAndParams(
      const std::unordered_map<std::string, ir::Node *> &var_nodes,
      const details::ParamsAndGrads &params_grads,
      details::GroupGradsAndParams *group_grads_params) const {
    SetGroupAccordingToLayers(var_nodes, params_grads, group_grads_params);
    SetGroupAccordingToMemorySize(var_nodes, group_grads_params);
  }

  void SetGroupAccordingToLayers(
      const std::unordered_map<std::string, ir::Node *> &var_nodes,
      const details::ParamsAndGrads &params_grads,
      details::GroupGradsAndParams *group_grads_params) const {
    std::unordered_map<std::string, std::vector<int>> layer_params;

    for (size_t i = 0; i < params_grads.size(); ++i) {
      auto pos = params_grads[i].first.find_first_of(".");
      if (pos == std::string::npos) {
        layer_params[params_grads[i].first].emplace_back(i);
      } else {
        layer_params[params_grads[i].first.substr(0, pos)].emplace_back(i);
      }
    }

    group_grads_params->reserve(layer_params.size());
    for (size_t i = 0; i < params_grads.size(); ++i) {
      auto pos = params_grads[i].first.find_first_of(".");
      std::string key = params_grads[i].first;
      if (pos != std::string::npos) {
        key = params_grads[i].first.substr(0, pos);
      }
      auto iter = layer_params.find(key);
      if (iter == layer_params.end()) continue;

      group_grads_params->emplace_back();
      auto &local_group_grads_params = group_grads_params->back();
      for (auto &idx : iter->second) {
        local_group_grads_params.emplace_back(
            std::make_pair(params_grads[idx].second, params_grads[idx].first));
      }
      layer_params.erase(iter);
    }

    VLOG(10) << "SetGroupAccordingToLayers: ";
    if (VLOG_IS_ON(10)) {
      PrintGroupInfo(var_nodes, group_grads_params);
    }
  }

  void PrintGroupInfo(
      const std::unordered_map<std::string, ir::Node *> &var_nodes,
      details::GroupGradsAndParams *group_grads_params) const {
    for (size_t i = 0; i < group_grads_params->size(); ++i) {
      VLOG(10) << "group " << i;
      std::stringstream out;
      size_t gps_size = 0;
      for (auto &g_p : group_grads_params->at(i)) {
        auto iter = var_nodes.find(g_p.second);
        PADDLE_ENFORCE(iter != var_nodes.end(), "%s is not found.", g_p.second);
        auto shape = iter->second->Var()->GetShape();
        size_t size = framework::SizeOfType(iter->second->Var()->GetDataType());
        std::for_each(shape.begin(), shape.end(),
                      [&size](const int64_t &n) { size *= n; });
        gps_size += size;
        out << string::Sprintf("(%s(%d), %s)", g_p.second, size, g_p.first);
      }
      VLOG(10) << out.str()
               << ", group size:" << group_grads_params->at(i).size()
               << ", group memory size:" << static_cast<double>(gps_size) / kMB
               << "(MB)";
    }
  }

  void SetGroupAccordingToMemorySize(
      const std::unordered_map<std::string, ir::Node *> &var_nodes,
      details::GroupGradsAndParams *group_grads_params) const {
    const double group_memory_size = GetFuseParameterMemorySize();
    if (group_memory_size <= 0.0) {
      return;
    }
    details::GroupGradsAndParams local_group_grads_params;
    size_t j = 0;
    while (j < group_grads_params->size()) {
      local_group_grads_params.emplace_back();
      auto &group_p_g = local_group_grads_params.back();
      size_t local_group_memory_size = 0;
      while (j < group_grads_params->size()) {
        std::for_each(
            group_grads_params->at(j).begin(), group_grads_params->at(j).end(),
            [&local_group_memory_size,
             &var_nodes](const std::pair<std::string, std::string> &g_p) {
              auto iter = var_nodes.find(g_p.second);
              PADDLE_ENFORCE(iter != var_nodes.end(), "%s is not found.",
                             g_p.second);
              auto shape = iter->second->Var()->GetShape();
              size_t size =
                  framework::SizeOfType(iter->second->Var()->GetDataType());
              std::for_each(shape.begin(), shape.end(),
                            [&size](const int64_t &n) { size *= n; });
              local_group_memory_size += size;
            });
        group_p_g.insert(group_p_g.end(), group_grads_params->at(j).begin(),
                         group_grads_params->at(j).end());
        ++j;
        if (GetFuseParameterGroupsSize() > 1 &&
            group_p_g.size() >
                static_cast<size_t>(GetFuseParameterGroupsSize())) {
          break;
        }

        if (static_cast<double>(local_group_memory_size) / kMB >=
            group_memory_size) {
          break;
        }
      }
    }

    std::swap(*group_grads_params, local_group_grads_params);

    VLOG(10) << string::Sprintf(
        "SetGroupAccordingToMemorySize(memory_size: %f):", group_memory_size);

    if (VLOG_IS_ON(10)) {
      PrintGroupInfo(var_nodes, group_grads_params);
    }
  }

 private:
  bool IsSupportedVarType(const proto::VarType::Type &type) const {
    // Current only support LOD_TENSOR.
    return type == proto::VarType::LOD_TENSOR;
  }

  void RecordParamsAndGrads(ir::Node *node,
                            details::ParamsAndGrads *params_grads) const {
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
      const details::ParamsAndGrads &params_grads) const {
    //  Init Gradients and FusedVars
    VLOG(10) << "Init FusedVars and Gradients.";
    for (auto it = local_scopes.rbegin(); it != local_scopes.rend(); ++it) {
      auto &scope = *it;

      PADDLE_ENFORCE(scope->FindVar(fused_var_name) == nullptr,
                     "%s has existed in scope.", fused_var_name);
      scope->Var(fused_var_name)->GetMutable<LoDTensor>();

      for (auto &p_g : params_grads) {
        auto iter = vars.find(p_g.second);
        PADDLE_ENFORCE(iter != vars.end());
        PADDLE_ENFORCE_NOT_NULL(iter->second->Var());
        PADDLE_ENFORCE_EQ(iter->second->Var()->GetType(),
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
};
}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(alloc_continuous_space_for_grad_pass,
              paddle::framework::ir::AllocContinuousSpaceForGradPass)
    .RequirePassAttr(paddle::framework::details::kPlaces)
    .RequirePassAttr(paddle::framework::details::kLocalScopes);
