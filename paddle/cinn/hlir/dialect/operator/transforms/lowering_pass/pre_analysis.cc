// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/cinn/hlir/dialect/operator/transforms/lowering_pass/pre_analysis.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/lowering_pass/utils.h"

namespace cinn::hlir::dialect::details {
void FusionOpAnalysis::GatherGroup(pir::Operation* fusion_op) {
  OpLoweringGroupPtr group_ptr = RebuildGroup(fusion_op, is_dy_shape_);
  VLOG(6) << "Gather Group " << group_ptr->FuncName()
          << " for fusion_op : " << fusion_op->id();
  pre_analysis_info_->group_infos.insert({fusion_op, group_ptr});
  if (is_dy_shape_) {
    auto broadcast_tree_info = std::make_shared<BroadcastTreeInfo>(group_ptr);
    pre_analysis_info_->broadcast_tree_infos.insert(
        {group_ptr, broadcast_tree_info});
  }
}

void FusionOpAnalysis::RunImpl(pir::Operation* op) {
  if (op->isa<cinn::dialect::FusionOp>()) {
    GatherGroup(op);
    return;
  }
  for (uint32_t i = 0; i < op->num_regions(); ++i) {
    for (auto& block : op->region(i)) {
      for (auto& op : block) {
        RunImpl(&op);
      }
    }
  }
}

void FusionOpAnalysis::PreCompileGroup() {
  std::vector<OpLoweringGroupPtr> groups;
  const auto& EnqueueGroup = [&](const OpLoweringGroupPtr& group) {
    const bool has_broadcast_tree =
        pre_analysis_info_->broadcast_tree_infos.count(group) > 0;
    if (has_broadcast_tree) {
      const auto broadcast_tree =
          pre_analysis_info_->broadcast_tree_infos.at(group);
      if (broadcast_tree->HasMultiBranch()) {
        return;  // do nothing
      }
    }
    groups.push_back(group);
  };
  for (auto& group_info : pre_analysis_info_->group_infos) {
    EnqueueGroup(group_info.second);
  }
  // Build and trigger compilaion cache.
  VLOG(4) << "Parallel Pre-Compile for Group with size: " << groups.size();
  PirCompiler pir_compiler(cinn::common::DefaultNVGPUTarget());
  pir_compiler.Build(groups);
}
}  // namespace cinn::hlir::dialect::details
