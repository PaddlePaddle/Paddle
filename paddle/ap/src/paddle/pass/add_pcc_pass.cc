// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/ap/include/paddle/pass/add_pcc_pass.h"

#include <chrono>
#include "paddle/common/errors.h"
#include "paddle/common/flags.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/utils/shape_analysis_utils.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/dialect/shape/ir/shape_dialect.h"
#include "paddle/pir/include/dialect/shape/transforms/shape_optimization_pass.h"
#include "paddle/pir/include/dialect/shape/utils/shape_analysis.h"
#include "paddle/pir/include/pass/pass_manager.h"

#include "paddle/ap/include/memory/guard.h"
#include "paddle/ap/include/paddle/pass/ap_generic_drr_pass.h"
#include "paddle/ap/include/paddle/pass/convert_pd_facade_to_ap_facade.h"
#include "paddle/ap/include/paddle/pass/fallback_fusion_op_to_phi_pass.h"
#include "paddle/ap/include/paddle/pass/fuse_ap_trivial_pass.h"
#include "paddle/ap/include/paddle/pass/put_ap_trivial_ops_in_range_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/fuse_shape_ops_into_generate_shape_op_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/move_generate_shape_ops_to_prologue_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/remove_redundant_full_int_array_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/split_generate_shape_into_shape_ops_pass.h"
#include "paddle/fluid/pir/transforms/general/common_subexpression_elimination_pass.h"
#include "paddle/fluid/pir/transforms/general/dead_code_elimination_pass.h"
#include "paddle/pir/include/core/ir_printer.h"

namespace ap::paddle {

void ApplyShapeOptimizationPass(
    ::pir::Program* program,
    const std::function<std::shared_ptr<::pir::PassManager>()>&
        CreatePassManager) {
  std::shared_ptr<pir::PassManager> pass_manager = CreatePassManager();
  pir::OriginalAttributesFilter::Instance().SetOriginalAttributesMap(
      ::paddle::dialect::GetAllOpOriginalAttributes());

  pass_manager->AddPass(pir::CreateShapeOptimizationPass());
  pass_manager->Run(program);
}

void ApplyApGenericDrrPass(
    ::pir::Program* program,
    const std::function<std::shared_ptr<pir::PassManager>()>&
        CreatePassManager) {
  ap::memory::Guard guard{};
  if (auto pass = ap::paddle::CreateApGenericClassicDrrPass(
          guard.circlable_ref_list())) {
    std::shared_ptr<pir::PassManager> pass_manager = CreatePassManager();
    pass_manager->AddPass(std::move(pass.value()));
    pass_manager->AddPass(pir::CreateDeadCodeEliminationPass());
    pass_manager->Run(program);
  }
  if (auto pass = ap::paddle::CreateApGenericAbstractDrrPass(
          guard.circlable_ref_list())) {
    std::shared_ptr<pir::PassManager> pass_manager = CreatePassManager();
    pass_manager->AddPass(std::move(pass.value()));
    pass_manager->AddPass(pir::CreateDeadCodeEliminationPass());
    pass_manager->Run(program);
  }
}

void ApplyApFacadePass(::pir::Program* program,
                       const std::function<std::shared_ptr<pir::PassManager>()>&
                           CreatePassManager) {
  std::shared_ptr<pir::PassManager> pass_manager = CreatePassManager();
  pass_manager->AddPass(CreateConvertPdFacadeToApFacadePass());
  pass_manager->Run(program);
}

void ApplyFuseApTrivialPass(
    ::pir::Program* program,
    const std::function<std::shared_ptr<pir::PassManager>()>&
        CreatePassManager) {
  {
    std::shared_ptr<pir::PassManager> pass_manager = CreatePassManager();
    pass_manager->AddPass(CreateFuseApTrivialPass());
    pass_manager->Run(program);
  }
  {
    std::shared_ptr<pir::PassManager> pass_manager = CreatePassManager();
    pass_manager->AddPass(CreatePutApTrivialOpsInRangePass());
    pass_manager->AddPass(CreateFuseApTrivialPass());
    pass_manager->Run(program);
  }
}

void ApplyFallbackToPhiPass(
    ::pir::Program* program,
    const std::function<std::shared_ptr<pir::PassManager>()>&
        CreatePassManager) {
  {
    std::shared_ptr<pir::PassManager> pass_manager = CreatePassManager();
    pass_manager->AddPass(CreateFallbackFusionOpToPhiPass());
    pass_manager->Run(program);
  }
}

namespace {

struct FinishLogger {
  pir::Program* program_;
  int seq_no_{0};

  void operator()(const std::string& stage_name) {
    pir::IrPrinter(LOG(ERROR)
                   << seq_no_++ << ") after " << stage_name << "():\n")
        .PrintProgram(program_);
  }
};

}  // namespace

void ApplyPccPass(
    ::pir::Program* program,
    const std::function<std::shared_ptr<pir::PassManager>()>& CreatePassManager,
    bool is_train_mode) {
  LOG_FIRST_N(INFO, 1) << "Compiling subgraph with PCC backend ...";
  const uint32_t origin_num_ops = program->num_ops();
  if (origin_num_ops == 0) return;

  if (is_train_mode) {
    // Skip infer symbol shape in inference, because we have run this pass in
    // the previous process
    ApplyShapeOptimizationPass(program, CreatePassManager);
  }
  FinishLogger Logger{program};
  ApplyApFacadePass(program, CreatePassManager);
  Logger("ApplyApFacadePass");
  ApplyFuseApTrivialPass(program, CreatePassManager);
  Logger("ApplyFuseApTrivialPass");
  ApplyApGenericDrrPass(program, CreatePassManager);
  Logger("ApplyApGenericDrrPass");
  ApplyFallbackToPhiPass(program, CreatePassManager);
  Logger("ApplyFallbackToPhiPass");
}

}  // namespace ap::paddle
