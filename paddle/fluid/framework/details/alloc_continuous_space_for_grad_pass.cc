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
DEFINE_uint32(fuse_parameter_memory_size, 0,  // 0 KB
              "fuse_parameter_memory_size is up limited memory size "
              "of one group parameters' gradient which is the input "
              "of communication calling(e.g NCCLAllReduce). "
              "The default value is 0, it means that "
              "not set group according to memory_size.");
DEFINE_int32(
    fuse_parameter_groups_size, 3,
    "fuse_parameter_groups_size is the size of one group parameters' gradient. "
    "The default value is a experimental result. If the "
    "fuse_parameter_groups_size is 1, it means that the groups size is "
    "the number of parameters' gradient. If the fuse_parameter_groups_size is "
    "-1, it means that there are only one group. The default value is 3, it is "
    "an experimental value.");

namespace paddle {
namespace framework {
namespace details {

static const char kUnKnow[] = "@UNKNOW@";
static framework::proto::VarType::Type kDefaultDtype =
    framework::proto::VarType::Type::VarType_Type_BOOL;

class AllocContinuousSpaceForGradPass : public ir::Pass {
 protected:
  ir::Graph *ApplyImpl(ir::Graph *graph) const override {
    ir::Graph &result = *graph;

    auto &places = Get<const std::vector<platform::Place>>(kPlaces);
    auto &local_scopes = Get<const std::vector<Scope *>>(kLocalScopes);

    ResetAttribute<ParamsAndGrads>(kParamsAndGrads, &result);
    ResetAttribute<GroupGradsAndParams>(kGroupGradsAndParams, &result);

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

    auto &group_grads_params =
        result.Get<GroupGradsAndParams>(kGroupGradsAndParams);

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
        PADDLE_ENFORCE_NE(ele_dtype, kDefaultDtype);
      }
      PADDLE_ENFORCE_EQ(ele_dtype, dtype);
    }

    // Create the fused variable name.
    if (!result.Has(kFusedVars)) {
      result.Set(kFusedVars, new FusedVars);
    }
    const std::string prefix(kFusedVarNamePrefix);
    // The fused_var_name should be unique.
    auto fused_var_name = prefix + "GRAD@" + params_grads[0].second;
    auto &fused_var_set = result.Get<FusedVars>(kFusedVars);
    PADDLE_ENFORCE_EQ(fused_var_set.count(fused_var_name), 0);
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

  void SetGroupGradsAndParams(
      const std::unordered_map<std::string, ir::Node *> &var_nodes,
      const ParamsAndGrads &params_grads,
      GroupGradsAndParams *group_grads_params) const {
    SetGroupAccordingToLayers(var_nodes, params_grads, group_grads_params);
    SetGroupAccordingToMemorySize(var_nodes, group_grads_params);
    SetGroupAccordingToGroupSize(var_nodes, group_grads_params);
  }

  void SetGroupAccordingToLayers(
      const std::unordered_map<std::string, ir::Node *> &var_nodes,
      const ParamsAndGrads &params_grads,
      GroupGradsAndParams *group_grads_params) const {
    std::unordered_map<std::string, std::vector<int>> layer_params;

    for (size_t i = 0; i < params_grads.size(); ++i) {
      auto pos = params_grads[i].first.find_first_of(".");
      if (pos == std::string::npos) {
        layer_params[std::string(kUnKnow)].emplace_back(i);
      } else {
        layer_params[params_grads[i].first.substr(0, pos)].emplace_back(i);
      }
    }

    group_grads_params->reserve(layer_params.size());
    for (size_t i = 0; i < params_grads.size(); ++i) {
      auto pos = params_grads[i].first.find_first_of(".");
      std::string key = kUnKnow;
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
    for (size_t i = 0; i < group_grads_params->size(); ++i) {
      VLOG(10) << "group " << i;
      std::stringstream out;
      for (auto &p_g : group_grads_params->at(i)) {
        out << "(" << p_g.second << ", " << p_g.first << "), ";
      }
      VLOG(10) << out.str();
    }
  }

  void SetGroupAccordingToMemorySize(
      const std::unordered_map<std::string, ir::Node *> &var_nodes,
      GroupGradsAndParams *group_grads_params) const {
    if (FLAGS_fuse_parameter_memory_size == 0) {
      return;
    }
    size_t group_memory_size =
        static_cast<size_t>(FLAGS_fuse_parameter_memory_size);
    GroupGradsAndParams local_group_grads_params;

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
        if (local_group_memory_size >= group_memory_size) {
          break;
        }
      }
    }

    std::swap(*group_grads_params, local_group_grads_params);

    VLOG(10) << string::Sprintf(
        "SetGroupAccordingToMemorySize(memory_size: %d):",
        FLAGS_fuse_parameter_memory_size);
    for (size_t i = 0; i < group_grads_params->size(); ++i) {
      VLOG(10) << "group " << i;
      std::stringstream out;
      for (auto &g_p : group_grads_params->at(i)) {
        auto iter = var_nodes.find(g_p.second);
        PADDLE_ENFORCE(iter != var_nodes.end(), "%s is not found.", g_p.second);
        auto shape = iter->second->Var()->GetShape();
        size_t size = framework::SizeOfType(iter->second->Var()->GetDataType());
        std::for_each(shape.begin(), shape.end(),
                      [&size](const int64_t &n) { size *= n; });
        out << string::Sprintf("(%s(%d), %s)", g_p.second, size, g_p.first);
      }
      VLOG(10) << out.str();
    }
  }

  void SetGroupAccordingToGroupSize(
      const std::unordered_map<std::string, ir::Node *> &var_nodes,
      GroupGradsAndParams *group_grads_params) const {
    if (FLAGS_fuse_parameter_groups_size == 1) {
      return;
    }
    size_t group_size = static_cast<size_t>(FLAGS_fuse_parameter_groups_size);
    if (FLAGS_fuse_parameter_groups_size == -1) {
      group_size = group_grads_params->size();
    }
    PADDLE_ENFORCE_GT(group_size, 1);
    size_t groups = (group_grads_params->size() + group_size - 1) / group_size;
    GroupGradsAndParams local_group_grads_params;
    local_group_grads_params.reserve(groups);

    size_t j = 0;
    for (size_t i = 0; i < groups; ++i) {
      local_group_grads_params.emplace_back();
      auto &group_p_g = local_group_grads_params.back();
      group_p_g.reserve(group_size);
      while (j < group_grads_params->size()) {
        group_p_g.insert(group_p_g.end(), group_grads_params->at(j).begin(),
                         group_grads_params->at(j).end());
        ++j;
        if (j % group_size == 0) break;
      }
    }
    std::swap(*group_grads_params, local_group_grads_params);

    VLOG(10) << "SetGroupAccordingToGroupSize(group_size: " << group_size
             << "): ";
    for (size_t i = 0; i < group_grads_params->size(); ++i) {
      VLOG(10) << "group " << i;
      std::stringstream out;
      for (auto &p_g : group_grads_params->at(i)) {
        out << "(" << p_g.second << ", " << p_g.first << "), ";
      }
      VLOG(10) << out.str();
    }
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

    // Run Only Once Programs
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
