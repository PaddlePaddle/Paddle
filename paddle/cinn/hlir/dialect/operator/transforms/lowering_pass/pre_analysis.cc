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
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/lowering_pass/utils.h"
#include "paddle/cinn/hlir/framework/pir_compiler.h"
#include "paddle/common/flags.h"

PD_DECLARE_bool(enable_cinn_compile_cache);

namespace cinn::dialect::ir::details {
using cinn::hlir::framework::PirCompiler;

void FusionOpAnalysis::GatherGroup(pir::Operation* fusion_op) {
  OpLoweringGroupPtr group_ptr = BuildOpLoweringGroup(fusion_op);
  VLOG(6) << "Gather Group " << group_ptr->FuncName()
          << " for fusion_op : " << fusion_op->id();
  group_infos_->insert({fusion_op, group_ptr});
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
  // Make compilation into lazy mode while
  // FLAGS_enable_cinn_compile_cache=false.
  if (!FLAGS_enable_cinn_compile_cache) return;

  std::vector<OpLoweringGroupPtr> groups;
  for (auto& group_info : *group_infos_) {
    groups.push_back(group_info.second);
  }
  // Build and trigger compilaion cache.
  VLOG(4) << "Parallel Pre-Compile for Group with size: " << groups.size();
  PirCompiler pir_compiler(cinn::common::DefaultDeviceTarget());
  pir_compiler.Build(groups);
}
}  // namespace cinn::dialect::ir::details
