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

#include "paddle/cinn/hlir/dialect/operator/transforms/lowering_pass/utils.h"

#include "paddle/cinn/adt/generate_map_expr.h"
#include "paddle/cinn/hlir/dialect/operator/ir/attribute_storage.h"
#include "paddle/cinn/hlir/dialect/operator/ir/generate_shape_util.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_attribute.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/lowering_pass/collect_sym_expr.h"
#include "paddle/cinn/hlir/dialect/runtime/ir/jit_kernel_op.h"
#include "paddle/cinn/hlir/dialect/runtime/ir/runtime_dialect.h"
#include "paddle/cinn/hlir/framework/pir/compilation_cache.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/cinn/hlir/framework/pir_compiler.h"
#include "paddle/cinn/runtime/flags.h"

PD_DECLARE_bool(cinn_enable_map_expr);
PD_DECLARE_bool(enable_cinn_compile_cache);

namespace cinn::dialect::ir::details {

using cinn::hlir::framework::CompilationCache;
using cinn::hlir::framework::PirCompiler;
using cinn::hlir::framework::pir::CINNKernelInfo;
using cinn::hlir::framework::pir::CompatibleInfo;

std::unordered_map<std::string, ::pir::Attribute> GetJitKernelAttr(
    const OpLoweringGroupPtr& group) {
  const auto& CreateKernelInfo = [&]() -> CINNKernelInfo {
    const auto& CreateFromCache = [&]() {
      hlir::framework::pir::FusionInfo fusion_info(*group);
      return CompilationCache::Instance().GetKernelInfo(fusion_info);
    };
    const auto& CreateFromNewCompile = [&]() {
      PirCompiler pir_compiler(cinn::common::DefaultDeviceTarget());
      return pir_compiler.Build({group})[0];
    };

    if (FLAGS_enable_cinn_compile_cache) {
      return CreateFromCache();
    } else {
      return CreateFromNewCompile();
    }
  };
  std::unordered_map<std::string, ::pir::Attribute> attrs{
      {cinn::dialect::JitKernelOp::kAttrName,
       cinn::dialect::CINNKernelInfoAttribute::get(pir::IrContext::Instance(),
                                                   CreateKernelInfo())}};
  return attrs;
}

OpLoweringGroupPtr BuildOpLoweringGroup(pir::Operation* fusion_op_ptr) {
  auto fusion_op = fusion_op_ptr->dyn_cast<cinn::dialect::FusionOp>();
  std::vector<::pir::Operation*> ops;
  auto group_op_kind = cinn::hlir::framework::OpPatternKind::kElementWise;
  // Rebuild ops of the group
  for (auto op : fusion_op.GetOperators()) {
    if (!op->isa<::pir::YieldOp>()) {
      ops.push_back(op);
      group_op_kind = static_cast<int>(CompatibleInfo::OpKind(*op)) >
                              static_cast<int>(group_op_kind)
                          ? CompatibleInfo::OpKind(*op)
                          : group_op_kind;
    }
  }

  PADDLE_ENFORCE_GT(fusion_op.attributes().count("group_info"),
                    0UL,
                    ::common::errors::InvalidArgument(
                        "fusion_op should have group_info attribute."));

  const auto attr = fusion_op.attribute("group_info")
                        .dyn_cast<cinn::dialect::GroupInfoAttribute>()
                        .data();

  const auto& fn_name = attr.fn_name;
  auto group = std::make_shared<OpLoweringGroup>(
      ops,
      fn_name,
      fusion_op_ptr->attribute("fusion_tracker")
          .dyn_cast<cinn::dialect::FusionTrackerPtrAttribute>()
          .data());

  group_op_kind =
      static_cast<int>(attr.op_pattern_kind) > static_cast<int>(group_op_kind)
          ? attr.op_pattern_kind
          : group_op_kind;
  group->set_loop_ranges(attr.loop_ranges);
  group->set_loop_ranges_expr(attr.loop_ranges_expr);
  group->set_reduce_axis(attr.reduce_axis);
  group->set_alignment_schedule_info(attr.alignment_schedule_info);
  group->set_op_pattern_kind(group_op_kind);

  // Rebuild output_ops and input_ops of the group
  auto yield_op = fusion_op.GetOperators().back();
  for (size_t i = 0; i < yield_op->num_operands(); ++i) {
    auto in = yield_op->operand_source(i);
    group->mut_output_values().push_back(in);
    group->mut_output_ops().insert(in.defining_op());
  }

  // Because the group is rebuilt, the order of group.output_values generated
  // by BuildCUDAJITInfo may not be same with the order bound in the yield op,
  // so a mapping is required.
  if (FLAGS_cinn_enable_map_expr) {
    cinn::adt::TryGenerateMapExprFromGroup(group);
  }
  // Rebuild other information
  // TODO(zhangyuqin1998): Do we need group.master_ops?
  return group;
}

void UpdateGroupShapeOrDataExprs(OpLoweringGroupPtr group) {
  auto& shape_analysis =
      pir::ShapeAnalysisManager::Instance().Get(group->GetParentProgram());
  group->set_value_to_shape_or_data_exprs(
      CreateGroupShapeOrDataExprs(group, shape_analysis));
}

}  // namespace cinn::dialect::ir::details
