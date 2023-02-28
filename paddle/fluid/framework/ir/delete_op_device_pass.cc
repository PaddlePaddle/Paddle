// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>
#include "paddle/fluid/framework/ir/pass.h"

namespace phi {
class DenseTensor;
}  // namespace phi

namespace paddle {
namespace framework {
class Scope;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace ir {

// "op_device" attr is only used in model training. "op_device" attr will change
// place of op kernel, so we use "delete_op_device_pass" to remove it.
class DeleteOpDevicePass : public Pass {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;
};

void DeleteOpDevicePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));
  int found_subgraph_count = 0;
  for (auto* node : graph->Nodes()) {
    if (!node->IsOp() || !node->Op()->HasAttr("op_device")) continue;
    node->Op()->RemoveAttr("op_device");
    found_subgraph_count++;
  }
  if (found_subgraph_count > 0) {
    LOG(INFO) << "---  detected " << found_subgraph_count << " subgraphs";
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(delete_op_device_pass, paddle::framework::ir::DeleteOpDevicePass);
