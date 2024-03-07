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

#include "paddle/cinn/hlir/dialect/operator/transforms/add_cinn_pass.h"

#include "paddle/common/errors.h"
#include "paddle/common/flags.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/dialect/shape/ir/shape_dialect.h"
#include "paddle/pir/include/pass/pass_manager.h"

#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/add_broadcast_to_elementwise_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/add_store_in_fusion_op_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/cinn_group_cluster_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/dynamic_reshape_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/fuse_shape_ops_into_generate_shape_op_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/check_infer_symbolic_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/convert_0d_to_1d_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/convert_dynamic_to_static_dim_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/convert_static_dim_to_dynamic_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/divide_group_op_to_fusion_op_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/move_generate_shape_ops_to_prologue_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/simplify_dim_expr_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/substitute_dim_expr_based_on_constraints_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/insert_broadcast_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/lower_cinn_fusion_op_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/pd_to_cinn_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/remove_unchanged_reshape_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/replace_dynamic_expand_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/split_generate_shape_into_shape_ops_pass.h"
#include "paddle/fluid/pir/transforms/build_cinn_pass.h"
#include "paddle/fluid/pir/transforms/dead_code_elimination_pass.h"
#include "paddle/fluid/pir/transforms/shape_optimization_pass.h"

COMMON_DECLARE_bool(print_ir);
COMMON_DECLARE_bool(check_infer_symbolic);
PD_DECLARE_bool(group_schedule_tiling_first);

namespace cinn::dialect::ir {

namespace {
bool HasDynamicShape(const pir::Program& program) {
  for (const auto& op : *program.block()) {
    if (op.isa<pir::CombineOp>()) {
      continue;
    }
    for (uint32_t i = 0; i < op.num_results(); ++i) {
      if (op.result(i) && op.result(i).type()) {
        auto shape_type =
            op.result(i).type().dyn_cast<pir::ShapedTypeInterface>();
        if (shape_type && shape_type.IsDynamicShape()) {
          return true;
        }
      }
    }
  }
  return false;
}
}  // namespace

void ApplyCinnPreprocessPass(
    ::pir::Program* program,
    const std::function<std::shared_ptr<::pir::PassManager>()>&
        CreatePassManager) {
  std::shared_ptr<pir::PassManager> pass_manager = CreatePassManager();
  bool has_dynamic_shape = HasDynamicShape(*program);

  pass_manager->AddPass(cinn::dialect::ir::CreateConvert0DTo1DPass());
  if (!has_dynamic_shape && FLAGS_check_infer_symbolic) {
    pass_manager->AddPass(pir::CreateShapeOptimizationPass());
    pass_manager->AddPass(cinn::dialect::ir::CreateCheckInferSymbolicPass());
  }
  pass_manager->AddPass(cinn::dialect::ir::CreatePdOpToCinnOpPass());

  pass_manager->AddPass(
      cinn::dialect::ir::CreateAddBroadcastToElementwisePass());
  pass_manager->AddPass(pir::CreateDeadCodeEliminationPass());

  if (has_dynamic_shape) {
    pass_manager->AddPass(cinn::dialect::ir::CreateConvert0DTo1DPass());
    pass_manager->AddPass(pir::CreateShapeOptimizationPass());
    pass_manager->AddPass(cinn::dialect::ir::CreateConvert0DTo1DPass());
    pass_manager->AddPass(
        cinn::dialect::ir::CreateFuseShapeOpsIntoGenerateShapeOpPass());
    pass_manager->AddPass(pir::CreateDeadCodeEliminationPass());
  }
  pass_manager->AddPass(cinn::dialect::ir::CreateRemoveUnchangedReshapePass());

  pass_manager->Run(program);
}

void ApplyBuildGroupOpPass(
    ::pir::Program* program,
    const std::function<std::shared_ptr<pir::PassManager>()>&
        CreatePassManager) {
  std::shared_ptr<pir::PassManager> pass_manager = CreatePassManager();
  pass_manager->AddPass(pir::CreateBuildCinnPass());
  if (HasDynamicShape(*program)) {
    pass_manager->AddPass(pir::CreateShapeOptimizationPass());
    pass_manager->AddPass(cinn::dialect::ir::CreateInsertBroadcastPass());
  }
  pass_manager->Run(program);
}

void ApplyGroupOpPass(::pir::Program* program,
                      const std::function<std::shared_ptr<pir::PassManager>()>&
                          CreatePassManager) {
  std::shared_ptr<pir::PassManager> pass_manager = CreatePassManager();
  if (HasDynamicShape(*program)) {
    pass_manager->AddPass(::pir::CreateShapeOptimizationPass());
    pass_manager->AddPass(
        cinn::dialect::ir::CreateFuseShapeOpsIntoGenerateShapeOpPass());
    pass_manager->AddPass(
        cinn::dialect::ir::CreateMoveGenerateShapeOpsToProloguePass());
    pass_manager->AddPass(
        cinn::dialect::ir::CreateSubstituteDimExprBasedOnConstraintsPass());
    pass_manager->AddPass(cinn::dialect::ir::CreateSimplifyDimExprPass());
  }

  pass_manager->AddPass(cinn::dialect::ir::CreateDynamicReshapeOpPass());
  pass_manager->AddPass(cinn::dialect::ir::CreateReplaceDynamicExpandOpPass());
  pass_manager->AddPass(pir::CreateDeadCodeEliminationPass());

  pass_manager->Run(program);
}

void ApplyDivideGroupOpToFusionOpPass(
    ::pir::Program* program,
    const std::function<std::shared_ptr<pir::PassManager>()>&
        CreatePassManager) {
  std::shared_ptr<pir::PassManager> pass_manager = CreatePassManager();
  if (FLAGS_group_schedule_tiling_first) {
    pass_manager->AddPass(cinn::dialect::ir::CreateCinnGroupClusterPass());
    pass_manager->AddPass(cinn::dialect::ir::CreateAddStoreInFusionOpPass());
  } else {
    pass_manager->AddPass(
        cinn::dialect::ir::CreateDivideGroupOpToFusionOpPass());
  }
  pass_manager->Run(program);
}

void ApplyCinnLowerPass(
    ::pir::Program* program,
    const std::function<std::shared_ptr<pir::PassManager>()>&
        CreatePassManager) {
  std::shared_ptr<pir::PassManager> pass_manager = CreatePassManager();

  bool has_dynamic_shape = HasDynamicShape(*program);

  bool force_static_shape = false;
  if (auto pass = cinn::dialect::ir::CreateConvertDynamicToStaticDimPass()) {
    pass_manager->AddPass(std::move(pass.value()));
    force_static_shape = true;
  }
  if (auto pass = cinn::dialect::ir::CreateConvertStaticDimToDynamicPass()) {
    pass_manager->AddPass(std::move(pass.value()));
  }

  if (has_dynamic_shape && !force_static_shape) {
    pass_manager->AddPass(
        cinn::dialect::ir::CreateLowerCinnDyShapeFusionOpPass());
  }

  pass_manager->AddPass(cinn::dialect::ir::CreateLowerCinnFusionOpPass());
  pass_manager->AddPass(
      cinn::dialect::ir::CreateSplitGenerateShapeIntoShapeOpsPass());

  pass_manager->Run(program);
}

void ApplyCinnPass(::pir::Program* program,
                   const std::function<std::shared_ptr<pir::PassManager>()>&
                       CreatePassManager) {
  ApplyCinnPreprocessPass(program, CreatePassManager);
  ApplyBuildGroupOpPass(program, CreatePassManager);
  ApplyGroupOpPass(program, CreatePassManager);
  ApplyDivideGroupOpToFusionOpPass(program, CreatePassManager);
  ApplyCinnLowerPass(program, CreatePassManager);
}

}  // namespace cinn::dialect::ir
