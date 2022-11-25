/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/ir/mkldnn/mkldnn_placement_pass.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace framework {
namespace ir {

bool MKLDNNPlacementPass::IsSupport(const Node* op) const {
  auto op_type = op->Op()->Type();
  const auto excluded_op_types_list = GetExcludedOpTypesList();
  if (!excluded_op_types_list.empty() &&
      std::find(excluded_op_types_list.begin(),
                excluded_op_types_list.end(),
                op_type) != excluded_op_types_list.end()) {
    return false;
  }

  auto& all_kernels = OperatorWithKernel::AllOpKernels();
  auto it = all_kernels.find(op_type);
  if (it != all_kernels.end()) {
    for (auto& kernel_pair : it->second) {
      if (platform::is_cpu_place(kernel_pair.first.place_) &&
          (kernel_pair.first.library_type_ == LibraryType::kMKLDNN)) {
        if (op->inputs.size() > 0 && op->inputs[0]->IsVar() &&
            kernel_pair.first.data_type_ == op->inputs[0]->Var()->GetDataType())
          return true;
        return true;
      }
    }
  }

  auto phi_kernels = phi::KernelFactory::Instance().SelectKernelMap(
      phi::TransToPhiKernelName(op_type));

  for (auto& kernel_pair : phi_kernels) {
    if (kernel_pair.first.backend() == phi::Backend::ONEDNN) {
      if (op->inputs.size() > 0 && op->inputs[0]->IsVar() &&
          kernel_pair.first.dtype() == framework::TransToPhiDataType(
                                           op->inputs[0]->Var()->GetDataType()))
        return true;
      return true;
    }
  }
  return false;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(mkldnn_placement_pass, paddle::framework::ir::MKLDNNPlacementPass)
    .RequirePassAttr("mkldnn_enabled_op_types")
    .DefaultPassAttr("mkldnn_excluded_op_types",
                     new std::unordered_set<std::string>());
