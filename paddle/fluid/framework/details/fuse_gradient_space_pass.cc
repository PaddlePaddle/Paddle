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

#include "paddle/fluid/framework/details/fuse_gradient_space_pass.h"
#include <algorithm>
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace framework {
namespace details {

std::unique_ptr<ir::Graph> FuseGradientSpacePass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  ir::Graph& result = *graph;
  graph->Set(kParamsAndGrads, new ParamsAndGrads);

  std::unordered_map<std::string, ir::Node*> vars;
  std::unordered_map<std::string, ir::Node*> ops;

  // Get Vars and Ops
  VLOG(10) << "Record parameters and gradients.";
  for (ir::Node* node : result.Nodes()) {
    if (node->IsVar()) {
      if (node->Var()) {
        auto var_name = node->Var()->Name();
        // Note: The graph may have the same name node. For example, parameter
        // is the input of operator and it also is the output of optimizer;
        vars.emplace(var_name, node);
      }
    } else {
      GetTrainingGradVarName(node, &ops, &result);
    }
  }

  auto& params_grads = result.Get<ParamsAndGrads>(kParamsAndGrads);
  // Note: The sort of parameter is impotant
  std::sort(params_grads.begin(), params_grads.end(),
            [](const std::pair<std::string, std::string>& a,
               const std::pair<std::string, std::string>& b) -> bool {
              return a.first < b.first;
            });

  // Set Gradients as Persistable
  auto dtype = static_cast<proto::VarType::Type>(0);
  for (auto& p_g : params_grads) {
    // Get gradient var
    auto iter = vars.find(p_g.second);
    PADDLE_ENFORCE(iter != vars.end());

    // Set Persistable to prevent this var becoming reusable.
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

  if (!result.Has(kRunOnlyOnceProgram)) {
    result.Set(kRunOnlyOnceProgram, new RunOnlyOnceProgram);
  }
  result.Get<RunOnlyOnceProgram>(kRunOnlyOnceProgram).emplace_back();
  auto& program_desc =
      result.Get<RunOnlyOnceProgram>(kRunOnlyOnceProgram).back();
  auto* global_block = program_desc.MutableBlock(0);
  AppendAllocSpaceForVarsOp(params_name, grads_name, global_block);

  return std::move(graph);
}

void FuseGradientSpacePass::GetTrainingGradVarName(
    ir::Node* node, std::unordered_map<std::string, ir::Node*>* ops,
    ir::Graph* result) const {
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
      ops->emplace(backward_vars[i + 1], node);
      result->Get<ParamsAndGrads>(kParamsAndGrads)
          .emplace_back(std::make_pair(backward_vars[i] /*param*/,
                                       backward_vars[i + 1] /*grad*/));
    }
  } catch (boost::bad_get e) {
  }
}

void FuseGradientSpacePass::AppendAllocSpaceForVarsOp(
    const std::vector<std::string>& params_name,
    const std::vector<std::string>& grads_name, BlockDesc* global_block) const {
  auto op_desc = global_block->AppendOp();
  op_desc->SetType("alloc_space_for_vars");
  op_desc->SetInput("Parameters", params_name);
  op_desc->SetOutput("Gradients", grads_name);
}

bool FuseGradientSpacePass::IsSupportedVarType(
    const proto::VarType::Type& type) const {
  // Current only support LOD_TENSOR.
  return type == proto::VarType::LOD_TENSOR;
}

}  // namespace details
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fuse_gradient_space_pass,
              paddle::framework::details::FuseGradientSpacePass);
