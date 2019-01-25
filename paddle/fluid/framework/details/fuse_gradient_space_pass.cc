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
#include "paddle/fluid/framework/details/fuse_gradient_space_pass.h"
#include <algorithm>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace framework {
namespace details {

std::unique_ptr<ir::Graph> FuseGradientSpacePass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  ir::Graph& result = *graph;

  result.Set(kParamsAndGrads, new ParamsAndGrads);
  std::unordered_map<std::string, ir::Node*> vars;
  std::unordered_map<std::string, ir::Node*> ops;
  // Get parameters and gradients
  for (ir::Node* node : graph->Nodes()) {
    if (node->IsVar()) {
      if (node->Var()) {
        auto var_name = node->Var()->Name();
        PADDLE_ENFORCE_EQ(vars.count(var_name), static_cast<size_t>(0));
        vars.emplace(var_name, node);
      }
    } else {
      try {
        bool is_bk_op =
            static_cast<bool>(boost::get<int>(node->Op()->GetAttr(
                                  OpProtoAndCheckerMaker::OpRoleAttrName())) &
                              static_cast<int>(OpRole::kBackward));
        if (!is_bk_op) continue;

        // Currently, we assume that once gradient is generated, it can be
        // broadcast, and each gradient is only broadcast once.
        auto backward_vars =
            boost::get<std::vector<std::string>>(node->Op()->GetNullableAttr(
                OpProtoAndCheckerMaker::OpRoleVarAttrName()));
        PADDLE_ENFORCE_EQ(backward_vars.size() % 2, 0);

        for (size_t i = 0; i < backward_vars.size(); i += 2) {
          result.Get<ParamsAndGrads>(kParamsAndGrads)
              .emplace(backward_vars[i] /*param*/,
                       backward_vars[i + 1] /*grad*/);
          ops.emplace(backward_vars[i + 1], node);
        }
      } catch (boost::bad_get e) {
      }
    }
  }

  std::vector<std::string> grads_name;
  // Set Gradients as Persistable
  proto::VarType::Type fuse_space_type = static_cast<proto::VarType::Type>(0);
  auto& params_grads = result.Get<ParamsAndGrads>(kParamsAndGrads);
  for (auto& p_g : params_grads) {
    auto iter = vars.find(p_g.second);
    PADDLE_ENFORCE(iter != vars.end());
    // Set Persistable
    iter->second->Var()->SetPersistable(true);

    // The Gradient can be SeletedRows and LoDTensor
    bool valid_type =
        (iter->second->Var()->GetType() == proto::VarType::LOD_TENSOR);
    PADDLE_ENFORCE(valid_type);
    // Get Dtype
    auto dtype = iter->second->Var()->GetDataType();
    if (fuse_space_type == static_cast<proto::VarType::Type>(0)) {
      fuse_space_type = dtype;
      PADDLE_ENFORCE_NE(dtype, static_cast<proto::VarType::Type>(0));
    }
    PADDLE_ENFORCE_EQ(dtype, fuse_space_type);

    grads_name.emplace_back(p_g.second);
  }

  OpDesc desc;
  desc.SetType("alloc_space_for_vars");
  desc.SetInput("Input", grads_name);
  desc.SetOutput("Output", grads_name);

  auto alloc_space_node = result.CreateOpNode(&desc);
  // Need Insert alloc_space_node's input
  // we should know the deep of the ops

  // Insert alloc_space_node's output
  for (auto& op : ops) {
    auto ctl_node = result.CreateControlDepVar();
    alloc_space_node->outputs.emplace_back(ctl_node);
    ctl_node->inputs.emplace_back(alloc_space_node);
    op.second->inputs.emplace_back(ctl_node);
    ctl_node->outputs.emplace_back(op.second);
  }

  return std::move(graph);
}

}  // namespace details
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fuse_gradient_space_pass,
              paddle::framework::details::FuseGradientSpacePass);
