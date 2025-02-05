/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/ir/cudnn_placement_pass.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle::framework::ir {

bool CUDNNPlacementPass::IsSupport(const Node* op) const {
  std::string attr_name = GetAttrName();

  if (!(op->Op()->HasAttr(attr_name) || op->Op()->HasProtoAttr(attr_name)))
    return false;

  auto& all_kernels = OperatorWithKernel::AllOpKernels();
  auto it = all_kernels.find(op->Op()->Type());
  if (it == all_kernels.end()) {
    // All control operators don't have kernel.
    return false;
  }
  for (auto& kernel_pair : it->second) {
    if (phi::is_gpu_place(kernel_pair.first.place_) &&
        (kernel_pair.first.library_type_ == LibraryType::kCUDNN)) {
      return true;
    }
  }
  return false;
}

}  // namespace paddle::framework::ir

REGISTER_PASS(cudnn_placement_pass, paddle::framework::ir::CUDNNPlacementPass)
    .RequirePassAttr("cudnn_enabled_op_types");
