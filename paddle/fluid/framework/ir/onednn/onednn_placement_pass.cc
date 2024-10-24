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

#include "paddle/fluid/framework/ir/onednn/onednn_placement_pass.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle::framework::ir {

inline bool FoundOneDNNKernelWithCorrectDataType(
    const framework::ir::Node* op) {
  const auto op_type = op->Op()->Type();
  auto& all_kernels = framework::OperatorWithKernel::AllOpKernels();
  auto it = all_kernels.find(op_type);
  if (it != all_kernels.end()) {
    for (auto& kernel_pair : it->second) {
      if (phi::is_cpu_place(kernel_pair.first.place_) &&
          (kernel_pair.first.library_type_ ==
           framework::LibraryType::kMKLDNN)) {
        if (!op->inputs.empty()) {
          if (op->inputs[0]->IsVar() &&
              op->inputs[0]->Var()->Name() != "feed" &&
              kernel_pair.first.data_type_ ==
                  op->inputs[0]->Var()->GetDataType())
            return true;
        } else {
          return true;
        }
      }
    }
  }
  return false;
}

inline bool FoundPhiOneDNNKernelWithCorrectDataType(
    const framework::ir::Node* op) {
  auto op_type = op->Op()->Type();
  auto phi_kernels = phi::KernelFactory::Instance().SelectKernelMap(
      phi::TransToPhiKernelName(op_type));

  for (auto& kernel_pair : phi_kernels) {
    if (kernel_pair.first.backend() == phi::Backend::ONEDNN) {
      if (!op->inputs.empty()) {
        if (op->inputs[0]->IsVar() && op->inputs[0]->Var()->Name() != "feed" &&
            kernel_pair.first.dtype() ==
                phi::TransToPhiDataType(op->inputs[0]->Var()->GetDataType()))
          return true;
      } else {
        return true;
      }
    }
  }
  return false;
}

bool MKLDNNPlacementPass::IsSupport(const Node* op) const {
  if (FoundOneDNNKernelWithCorrectDataType(op) ||
      FoundPhiOneDNNKernelWithCorrectDataType(op)) {
    // For interpolate ops, there's a little difference between Paddle and
    // DNNL.
    // If run DNNL interpolate ops, manual set AnalysisConfig and apply
    // the corresponding pass.
    const std::vector<std::string> not_default_op_types = {"bilinear_interp",
                                                           "nearest_interp",
                                                           "trilinear_interp",
                                                           "bicubic_interp",
                                                           "linear_interp",
                                                           "bilinear_interp_v2",
                                                           "linear_interp_v2"};
    bool is_interpolate_op =
        std::find(not_default_op_types.begin(),
                  not_default_op_types.end(),
                  op->Op()->Type()) != not_default_op_types.end();
    return !is_interpolate_op;
  }
  return false;
}

}  // namespace paddle::framework::ir

REGISTER_PASS(onednn_placement_pass, paddle::framework::ir::MKLDNNPlacementPass)
    .RequirePassAttr("mkldnn_enabled_op_types");

REGISTER_PASS_CAPABILITY(onednn_placement_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().LE(
            "fusion_gru", 1));
