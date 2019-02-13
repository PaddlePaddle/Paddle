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

class AllocContinuousSpaceForGrad : public ir::Pass {
 protected:
  std::unique_ptr<ir::Graph> ApplyImpl(
      std::unique_ptr<ir::Graph> graph) const override {
    ir::Graph& result = *graph;
    if (result.Has(kParamsAndGrads)) {
      VLOG(10) << kParamsAndGrads << " are reset.";
      result.Erase(kParamsAndGrads);
    }
    result.Set(kParamsAndGrads, new ParamsAndGrads);

    // Record parameters and gradients
    std::unordered_map<std::string, ir::Node*> vars;
    for (ir::Node* node : result.Nodes()) {
      if (node->IsVar()) {
        if (node->Var()) {
          // Note: The graph may have the same name node. For example, parameter
          // is the input of operator and it also is the output of optimizer;
          vars.emplace(node->Var()->Name(), node);
        }
      } else {
        GetTrainingGradVarName(node, &result);
      }
    }

    // Note: Sort the parameters and gradient variables according
    // to parameters' name to make variables' name correspond correctly.
    auto& params_grads = result.Get<ParamsAndGrads>(kParamsAndGrads);
    std::sort(params_grads.begin(), params_grads.end(),
              [](const std::pair<std::string, std::string>& a,
                 const std::pair<std::string, std::string>& b) -> bool {
                return a.first < b.first;
              });

    // Set Gradients as Persistable to prevent this var becoming reusable.
    auto dtype = static_cast<proto::VarType::Type>(0);
    for (auto& p_g : params_grads) {
      // Get gradient var
      auto iter = vars.find(p_g.second);
      PADDLE_ENFORCE(iter != vars.end(), "%s is not found.", p_g.second);
      iter->second->Var()->SetPersistable(true);

      PADDLE_ENFORCE(IsSupportedVarType(iter->second->Var()->GetType()));

      // Get Dtype
      auto ele_dtype = iter->second->Var()->GetDataType();
      if (dtype == static_cast<proto::VarType::Type>(0)) {
        dtype = ele_dtype;
        PADDLE_ENFORCE_NE(ele_dtype, static_cast<proto::VarType::Type>(0));
      }
      PADDLE_ENFORCE_EQ(ele_dtype, dtype);
    }

    std::vector<std::string> grads_name;
    std::vector<std::string> params_name;
    grads_name.reserve(params_grads.size());
    params_name.reserve(params_grads.size());
    for (auto& p_g : params_grads) {
      params_name.emplace_back(p_g.first);
      grads_name.emplace_back(p_g.second);
    }

    // Create the fused variable name.
    const std::string prefix(kGradVarSuffix);
    auto fused_var_name = prefix + "_GRAD";
    if (!result.Has(kFusedVars)) {
      result.Set(kFusedVars, new FusedVars);
    }
    result.Get<FusedVars>(kFusedVars).emplace_back(fused_var_name);

    // Insert alloc_continuous_space_for_grad to RunOnlyOnceProgram,
    // which is executed before running the model with ParallelExecutor.
    if (!result.Has(kRunOnlyOnceProgram)) {
      result.Set(kRunOnlyOnceProgram, new RunOnlyOnceProgram);
    }
    result.Get<RunOnlyOnceProgram>(kRunOnlyOnceProgram).emplace_back();
    auto& program_desc =
        result.Get<RunOnlyOnceProgram>(kRunOnlyOnceProgram).back();
    auto* global_block = program_desc.MutableBlock(0);

    AppendAllocSpaceForVarsOp(params_name, grads_name, fused_var_name,
                              global_block);

    return std::move(graph);
  }

 private:
  bool IsSupportedVarType(const proto::VarType::Type& type) const {
    // Current only support LOD_TENSOR.
    return type == proto::VarType::LOD_TENSOR;
  }

  void AppendAllocSpaceForVarsOp(const std::vector<std::string>& params_name,
                                 const std::vector<std::string>& grads_name,
                                 const std::string& fused_var_name,
                                 BlockDesc* global_block) const {
    auto op_desc = global_block->AppendOp();
    op_desc->SetType("alloc_continuous_space_for_grad");
    op_desc->SetInput("Parameters", params_name);
    op_desc->SetOutput("Gradients", grads_name);
    op_desc->SetOutput("FusedOutput", {fused_var_name});
  }

  void GetTrainingGradVarName(ir::Node* node, ir::Graph* result) const {
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
      PADDLE_ENFORCE_EQ(backward_vars.size() % 2, 0);

      for (size_t i = 0; i < backward_vars.size(); i += 2) {
        VLOG(10) << "Trainable parameter: " << backward_vars[i]
                 << ", gradient: " << backward_vars[i + 1];
        result->Get<ParamsAndGrads>(kParamsAndGrads)
            .emplace_back(std::make_pair(backward_vars[i] /*param*/,
                                         backward_vars[i + 1] /*grad*/));
      }
    } catch (boost::bad_get e) {
    }
  }
};

}  // namespace details
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(alloc_continuous_space_for_grad_pass,
              paddle::framework::details::AllocContinuousSpaceForGrad);
