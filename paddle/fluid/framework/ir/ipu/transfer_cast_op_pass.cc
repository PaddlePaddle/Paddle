// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/ipu/transfer_cast_op_pass.h"

#include "paddle/fluid/framework/ir/pass_tester_helper.h"
#include "paddle/fluid/platform/device/ipu/ipu_backend.h"

namespace paddle {
namespace framework {
namespace ir {

// Transfer the target dtype of Cast Op to FP16 if the original target is FP32
// and enable FP16 mode.
void TransferCastOpPass::ApplyImpl(ir::Graph* graph) const {
  VLOG(10) << "enter TransferCastOpPass::ApplyImpl";
  VLOG(10) << "Raw Graph: ";
  VLOG(10) << DebugString(graph);

  auto ipu_backend = platform::ipu::IpuBackend::GetInstance();
  auto enable_fp16 = ipu_backend->GetIpuStrategy()->enable_fp16;
  auto transfer_cast_op = ipu_backend->GetIpuStrategy()->transfer_cast_op;
  if (enable_fp16 && transfer_cast_op) {
    for (auto* node : graph->Nodes()) {
      if (node->IsOp() && node->Op()->Type() == "popart_cast") {
        if (BOOST_GET_CONST(std::string, node->Op()->GetAttr("to")) ==
            "FLOAT") {
          node->Op()->SetAttr("to", std::string("FLOAT16"));
        }
      }
    }
  }

  VLOG(10) << "Post Graph: ";
  VLOG(10) << DebugString(graph);
  VLOG(10) << "leave TransferCastOpPass::ApplyImpl";
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(transfer_cast_op_pass, paddle::framework::ir::TransferCastOpPass);
