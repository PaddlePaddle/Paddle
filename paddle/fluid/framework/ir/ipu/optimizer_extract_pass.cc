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

#include "paddle/fluid/framework/ir/ipu/optimizer_extract_pass.h"

#include "paddle/fluid/framework/ir/pass_tester_helper.h"
#include "paddle/fluid/platform/device/ipu/ipu_backend.h"

namespace paddle {
namespace framework {
namespace ir {

void IpuOptimizerExtractPass::ApplyImpl(ir::Graph* graph) const {
  VLOG(10) << "enter IpuOptimizerExtractPass::ApplyImpl";
  VLOG(10) << "Raw Graph: ";
  VLOG(10) << DebugString(graph);

  auto ipu_backend = paddle::platform::ipu::IpuBackend::GetInstance();

  for (auto* node : graph->Nodes()) {
    if (node->IsOp() && node->Op()) {
      int op_role = BOOST_GET_CONST(
          int, node->Op()->GetAttr(
                   framework::OpProtoAndCheckerMaker::OpRoleAttrName()));

      // graph usually have multiple optimizer node for different parameter,
      // and these node have the same type and attr value usually
      if ((op_role == static_cast<int>(framework::OpRole::kOptimize))) {
        ipu_backend->GetExecutor().SetOptimizerType(node->Op()->Type());
        VLOG(10) << "found optimizer type: " << node->Op()->Type();

        for (const std::string& attr_name : node->Op()->AttrNames()) {
          auto attr_type = node->Op()->GetAttrType(attr_name);
          // with adam, attr are float
          if (attr_type == proto::AttrType::FLOAT) {
            auto attr_value =
                BOOST_GET_CONST(float, node->Op()->GetAttr(attr_name));
            ipu_backend->GetExecutor().SetOptimizerAttr(attr_name, attr_value);
          } else {
            VLOG(10) << "Skip " << attr_type;
          }
        }

        auto lr_var_name = node->Op()->Input("LearningRate");
        PADDLE_ENFORCE_EQ(lr_var_name.size(), 1u,
                          platform::errors::InvalidArgument(
                              "In op(%s), find input(LearningRate) failed.",
                              node->Op()->Type()));

        ipu_backend->GetExecutor().SetLRVarName(lr_var_name[0]);
      }

      if ((op_role == static_cast<int>(framework::OpRole::kLoss))) {
        VLOG(10) << "found loss op type: " << node->Op()->Type();
        auto outputs = node->Op()->Outputs();
        PADDLE_ENFORCE_EQ(
            outputs.size(), 1,
            platform::errors::InvalidArgument("Can only support one loss key"));

        auto losses_name = outputs.begin()->second;
        PADDLE_ENFORCE_EQ(losses_name.size(), 1,
                          platform::errors::InvalidArgument(
                              "Can only support one loss name"));

        ipu_backend->GetExecutor().SetLoss(losses_name[0]);
      }
    }
  }

  VLOG(10) << "Post Graph: ";
  VLOG(10) << DebugString(graph);
  VLOG(10) << "leave IpuOptimizerExtractPass::ApplyImpl";
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(optimizer_extract_pass,
              paddle::framework::ir::IpuOptimizerExtractPass);
