// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/frontend/optimize.h"

#include <memory>
#include <string>
#include <unordered_set>

#include "paddle/cinn/common/target.h"
#include "paddle/cinn/frontend/decomposer/use_decomposer.h"
#include "paddle/cinn/frontend/pass/use_program_pass.h"
#include "paddle/cinn/frontend/program_pass.h"
#include "paddle/cinn/frontend/syntax.h"
#include "paddle/cinn/hlir/framework/graph.h"
#include "paddle/cinn/hlir/framework/pass.h"
#include "paddle/cinn/hlir/framework/visualize_helper.h"
#include "paddle/cinn/hlir/pass/use_pass.h"
#include "paddle/cinn/runtime/flags.h"

PD_DECLARE_bool(cinn_use_fill_constant_folding);
PD_DECLARE_bool(cinn_use_op_fusion);
PD_DECLARE_bool(cinn_use_common_subexpression_elimination);
PD_DECLARE_string(cinn_check_fusion_accuracy_pass);
PD_DECLARE_bool(cinn_use_custom_call);
PD_DECLARE_bool(use_reduce_split_pass);
PD_DECLARE_bool(cinn_use_dense_merge_pass);
PD_DECLARE_string(cinn_custom_call_deny_ops);
PD_DECLARE_bool(general_fusion_merge_pass);

namespace cinn {
namespace frontend {

OptimizeOptions DefaultTrainingOptimizeOptions() {
  OptimizeOptions options;
  options.program_passes.emplace_back("ExpandZeroDim");
  options.program_passes.emplace_back("AutoCast");
  options.program_passes.emplace_back("Decomposer");
  options.program_passes.emplace_back("RemoveIdentity");

  options.program_passes.emplace_back("CastCollapsing");
  options.program_passes.emplace_back("TransposeCollapsing");
  options.program_passes.emplace_back("RemoveIdentity");

#ifdef CINN_WITH_CUDA
  auto can_find_custom_call_deny_op = [](const std::string& op) {
    return FLAGS_cinn_custom_call_deny_ops.find(op) != std::string::npos;
  };
  bool is_gemm_use_cublas = FLAGS_cinn_use_custom_call &&
                            !can_find_custom_call_deny_op("matmul") &&
                            !can_find_custom_call_deny_op("cublas_gemm") &&
                            !can_find_custom_call_deny_op("cublas_matmul");
  if (is_gemm_use_cublas) {
    options.program_passes.emplace_back("TransposeFoldingInput");
    options.program_passes.emplace_back("GemmRewriter");
    options.program_passes.emplace_back("TransposeFoldingOutput");
    options.program_passes.emplace_back("GemmRewriter");
  }
#endif

  options.program_passes.emplace_back("AutoBroadcast");
  options.program_passes.emplace_back("FillConstantRewriter");
  if (FLAGS_cinn_use_fill_constant_folding) {
    options.program_passes.emplace_back("FillConstantFolding");
  }
  options.program_passes.emplace_back("RemoveIdentity");
  options.program_passes.emplace_back("DeadCodeEliminate");

  options.graph_passes = {"ConstantFolding"};
  if (FLAGS_cinn_use_dense_merge_pass) {
    options.graph_passes.push_back("DenseMergePass");
  }

  if (FLAGS_cinn_use_custom_call) {
    options.graph_passes.emplace_back("TransToCustomCallPass");
  }

  if (FLAGS_cinn_use_common_subexpression_elimination) {
    options.graph_passes.emplace_back("CommonSubexpressionEliminationPass");
  }

  // this pass should be applied before merge
  if (FLAGS_use_reduce_split_pass) {
    options.graph_passes.emplace_back("ReduceSplit");
  }

  if (FLAGS_cinn_use_op_fusion) {
    options.graph_passes.emplace_back("OpFusionPass");
    if (FLAGS_general_fusion_merge_pass) {
      options.graph_passes.emplace_back("GeneralFusionMergePass");
    } else {
      options.graph_passes.emplace_back("FusionMergePass");
    }
  } else {
    options.graph_passes.emplace_back("BuildNonFusedGroupsPass");
  }

#ifdef CINN_WITH_CUDA
  options.graph_passes.emplace_back("SingleGroupOptimizePass");
#endif

  // WARNING: the pass must be the last pass !!!
  if (!cinn::runtime::CheckStringFlagFalse(
          FLAGS_cinn_check_fusion_accuracy_pass)) {
    // Check the correct of fusion kernels, if the results not satisfied
    // 'allclose(rtol=1e-05f, atol=1e-08f)', report error and exited.
    options.graph_passes.emplace_back("CheckFusionAccuracyPass");
    options.graph_passes.emplace_back("TransToCustomCallPass");
  }
  return options;
}

std::vector<std::string> DefaultOpFusionPasses() {
  std::vector<std::string> passes;
  if (FLAGS_cinn_use_op_fusion) {
    passes = {"OpFusionPass", "FusionMergePass"};
  }
  return passes;
}

std::shared_ptr<hlir::framework::Graph> Optimize(
    frontend::Program* program,
    const std::unordered_set<std::string>& fetch_ids,
    common::Target target,
    const OptimizeOptions& options) {
  cinn::hlir::framework::PassPrinter::GetInstance()->Begin(fetch_ids);
  // Apply program passes
  VLOG(3) << "Before frontend::ProgramPass::Apply";
  frontend::ProgramPass::Apply(
      program, fetch_ids, target, options.program_passes);
  // Apply graph passes
  auto graph =
      std::make_shared<hlir::framework::Graph>(*program, fetch_ids, target);

  VLOG(3) << "Before hlir::framework::ApplyPasses";
  hlir::framework::ApplyPasses(graph.get(), options.graph_passes);
  cinn::hlir::framework::PassPrinter::GetInstance()->End();
  return graph;
}

std::shared_ptr<hlir::framework::Graph> Optimize(
    frontend::Program* program,
    const std::unordered_set<std::string>& fetch_ids,
    common::Target target,
    const std::vector<std::string>& passes) {
  OptimizeOptions options;

  bool enbale_fusion = false;
  if (!passes.empty()) {
    for (const auto& pass : passes) {
      auto* p_pass = ProgramPassRegistry::Global()->Find(pass);
      auto* g_pass =
          Registry<hlir::framework::PassFunctionRegister>::Global()->Find(pass);
      if (p_pass) {
        options.program_passes.emplace_back(pass);
      } else if (g_pass) {
        options.graph_passes.emplace_back(pass);
        if (pass == "OpFusionPass" || pass == "FusionMergePass") {
          enbale_fusion = true;
        }
      } else {
        LOG(FATAL) << "Pass " << pass
                   << " unsupported in CINN! Please check.\n";
      }
    }

    if (!enbale_fusion) {
      options.graph_passes.emplace_back("BuildNonFusedGroupsPass");
    }
  } else {
    // if pass empty, default enable all pass
    options = DefaultTrainingOptimizeOptions();
  }

  return Optimize(program, fetch_ids, target, options);
}

}  // namespace frontend
}  // namespace cinn
