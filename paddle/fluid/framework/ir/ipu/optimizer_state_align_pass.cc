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

#include "paddle/fluid/framework/ir/ipu/optimizer_state_align_pass.h"
#include "paddle/fluid/framework/ir/pass_tester_helper.h"
#include "paddle/fluid/platform/device/ipu/common.h"
#include "paddle/fluid/platform/device/ipu/ipu_backend.h"

namespace paddle {
namespace framework {
namespace ir {

using paddle::platform::ipu::IpuBackend;
using framework::ir::Graph;
using framework::ir::Node;

void IpuOptimizerStateAlignPass::ApplyImpl(ir::Graph* graph) const {
  VLOG(10) << "enter IpuOptimizerStateAlignPass::ApplyImpl";
  VLOG(10) << "Raw Graph: ";
  VLOG(10) << DebugString(graph);

  auto ipu_backend = IpuBackend::GetInstance();
  const auto* scope_ = ipu_backend->GetScope();

  for (auto* node : graph->Nodes()) {
    if (node->IsOp() && node->Op()) {
      int op_role = BOOST_GET_CONST(
          int, node->Op()->GetAttr(
                   framework::OpProtoAndCheckerMaker::OpRoleAttrName()));

      if ((op_role == static_cast<int>(framework::OpRole::kOptimize))) {
        auto inputs = node->Op()->Inputs();
        if (inputs.count(platform::ipu::sBeta1Pow)) {
          auto var = scope_->GetVar(inputs.at(platform::ipu::sBeta1Pow)[0]);
          auto data = var->GetMutable<framework::LoDTensor>()->data<float>();
          auto beta = BOOST_GET_CONST(
              float, node->Op()->GetAttr(platform::ipu::sBeta1));

          // ensure current save with beta1pow, rather than step.
          // beta1pow = beta1 ^ (step + 1). Just set beta1pow because popart
          // support single Step__
          bool save_with_beta1pow = (data[0] < 1.0f) && (data[0] > 0.0f);
          float step = 0;
          float beta_acc = beta;
          while (beta_acc > data[0] && save_with_beta1pow) {
            beta_acc *= beta;
            step += 1;
          }

          if (save_with_beta1pow) {
            data[0] = step;
          }
        }
      }
    }
  }

  VLOG(10) << "Post Graph: ";
  VLOG(10) << DebugString(graph);
  VLOG(10) << "leave IpuOptimizerStateAlignPass::ApplyImpl";
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(optimizer_state_align_pass,
              paddle::framework::ir::IpuOptimizerStateAlignPass);
