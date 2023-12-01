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

#include "paddle/fluid/pir/transforms/constant_folding_pass.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/new_executor/interpretercore.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/pir/dialect/operator/interface/op_yaml_info.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/transforms/pd_op_to_kernel_pass.h"
#include "paddle/fluid/pir/transforms/transform_general_functions.h"

#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"

#include "paddle/pir/core/builtin_attribute.h"
#include "paddle/pir/core/builtin_op.h"
#include "paddle/pir/core/ir_context.h"
#include "paddle/pir/core/op_result.h"
#include "paddle/pir/core/op_trait.h"
#include "paddle/pir/core/operation.h"
#include "paddle/pir/core/parameter.h"
#include "paddle/pir/core/program.h"
#include "paddle/pir/pass/pass.h"
#include "paddle/pir/pattern_rewrite/frozen_rewrite_pattern_set.h"
#include "paddle/pir/pattern_rewrite/pattern_match.h"
#include "paddle/pir/pattern_rewrite/pattern_rewrite_driver.h"

namespace {

class ConstantFoldingPattern : public pir::RewritePattern {
 public:
  ConstantFoldingPattern(
      pir::IrContext* context,
      size_t* counter,
      const phi::Place& place,
      paddle::framework::Scope* scope,
      paddle::framework::interpreter::ExecutionConfig* exe_config,
      std::vector<std::string>* deleted_vars)
      : RewritePattern(MatchAnyOpTypeTag(),
                       1 /*benefit*/,
                       context,
                       {} /*generated_names*/),
        counter_(counter),
        place_(place),
        scope_(scope),
        exe_config_(exe_config),
        deleted_vars_(deleted_vars) {
    exe_config_->create_local_scope = false;
  }

  bool Match(pir::Operation* op) const override {
    // 1. Some ops do not need to be processed
    if (op->HasTrait<pir::SideEffectTrait>() ||
        op->isa<pir::ConstantTensorOp>() || op->isa<pir::ParameterOp>() ||
        op->isa<paddle::dialect::FeedOp>())
      return false;

    for (uint32_t i = 0; i < op->num_operands(); i++) {
      // 2. inputs must come from parameter op or constant op
      // 3. inputs must be a dense tensor type
      if (!(pir::GetDefiningOpForInput(op, i)->isa<pir::ParameterOp>() ||
            pir::GetDefiningOpForInput(op, i)->isa<pir::ConstantTensorOp>()) ||
          !op->operand_source(i)
               .type()
               .isa<paddle::dialect::DenseTensorType>()) {
        return false;
      }
    }

    for (uint32_t i = 0; i < op->num_results(); i++) {
      // 4. outputs must be a dense tensor type
      if (!op->result(i).type().isa<paddle::dialect::DenseTensorType>()) {
        return false;
      }
    }

    return true;
  }

  void Rewrite(pir::Operation* op,
               pir::PatternRewriter& rewriter) const override {  // NOLINT
    VLOG(4) << "constant_folding_pass applys on [" << op->name() << "] op";
    pir::Program new_program(rewriter.ir_context());
    auto output_var_name =
        BuildProgramFromOperation(op, &new_program, rewriter);

    // execute program
    exe_config_->skip_gc_vars.insert(output_var_name);
    auto kernel_program =
        paddle::dialect::PdOpLowerToKernelPass(&new_program, place_);
    paddle::framework::InterpreterCore core(
        place_, {}, kernel_program->block(), scope_, *exe_config_);

    core.Run({});

    rewriter.SetInsertionPointToStart(rewriter.block());
    // TODO(liuyuanle): support multiple output
    if (ReplaceResultByParameter(op)) {
      auto parameter_op = rewriter.Build<pir::ParameterOp>(
          output_var_name, op->result(0).type());
      parameter_op->set_attribute(
          kAttrIsPersisable, rewriter.array_attr({rewriter.bool_attr(true)}));

      rewriter.ReplaceAllUsesWith(op->result(0), parameter_op->result(0));
    } else {
      auto constant_op = rewriter.Build<pir::ConstantTensorOp>(
          rewriter.tensor_name_attr(output_var_name), op->result(0).type());
      constant_op->set_attribute(
          kAttrIsPersisable, rewriter.array_attr({rewriter.bool_attr(true)}));

      rewriter.ReplaceAllUsesWith(op->result(0), constant_op->result(0));
    }
    VLOG(4) << "constant_folding_pass applied on [" << op->name() << "] op";
    rewriter.EraseOp(op);
  }

 private:
  bool CheckUseOps(const std::vector<pir::Operation*>& use_ops) const {
    for (auto* use_op : use_ops) {
      if (use_op->isa<pir::CombineOp>()) {
        if (!ReplaceResultByParameter(use_op)) return false;
      } else if (use_op->HasInterface<paddle::dialect::OpYamlInfoInterface>()) {
        auto [input_infos, _1, _2, _3, _4] =
            use_op->dyn_cast<paddle::dialect::OpYamlInfoInterface>()
                .GetOpInfo();
        for (const auto& input_info : input_infos) {
          if (input_info.type_name.find("IntArrayAttribute") !=
                  std::string::npos ||
              input_info.type_name.find("ScalarAttribute") !=
                  std::string::npos) {
            return false;
          }
        }
      }
    }
    return true;
  }

  bool ReplaceResultByParameter(pir::Operation* op) const {
    for (uint32_t i = 0; i < op->num_results(); i++) {
      auto use_ops = pir::GetUseOpsForOutput(op, i);
      if (!CheckUseOps(use_ops)) return false;
    }
    return true;
  }

  std::string BuildProgramFromOperation(
      pir::Operation* op,
      pir::Program* new_program,
      pir::PatternRewriter& rewriter) const {  // NOLINT
    pir::Builder builder =
        pir::Builder(rewriter.ir_context(), new_program->block());

    // prepare op inputs
    std::vector<pir::Value> op_inputs;
    for (uint32_t i = 0; i < op->num_operands(); i++) {
      const auto& param_name =
          pir::GetParameterNameFromValue(op->operand_source(i));
      auto* param_var = scope_->FindVar(param_name);
      PADDLE_ENFORCE_NOT_NULL(
          param_var,
          phi::errors::InvalidArgument("Parameter var not in scope."));

      auto parameter_op = builder.Build<pir::ParameterOp>(
          param_name, op->operand_source(i).type());
      if (op->operand_source(i).use_count() <= 1) {
        deleted_vars_->push_back(param_name);
      } else {
        parameter_op->set_attribute(
            kAttrIsPersisable, rewriter.array_attr({rewriter.bool_attr(true)}));
      }
      op_inputs.push_back(parameter_op->result(0));
    }

    // prepare op outputs
    std::vector<pir::Type> output_types;
    for (uint32_t i = 0; i < op->num_results(); i++) {
      output_types.push_back(op->result(i).type());
    }

    auto* temp_op =
        builder.Build(op_inputs, op->attributes(), output_types, op->info());

    // TODO(liuyuanle): support multiple output
    // for (uint32_t i = 0; i < op->num_results(); i++) {
    std::stringstream ss;
    ss << std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::string output_var_name =
        "constant_folding@_" + ss.str() + std::to_string((*counter_)++);

    builder.Build<pir::ShadowOutputOp>(temp_op->result(0), output_var_name);
    // }

    return output_var_name;
  }

 private:
  size_t* counter_;
  phi::Place place_;
  paddle::framework::Scope* scope_;
  paddle::framework::interpreter::ExecutionConfig* exe_config_;
  std::vector<std::string>* deleted_vars_;
};

class ConstantFoldingPass : public pir::Pass {
 public:
  explicit ConstantFoldingPass(const phi::Place& place,
                               paddle::framework::Scope* scope)
      : pir::Pass("constant_folding_pass", 1), place_(place), scope_(scope) {
    PADDLE_ENFORCE_NOT_NULL(
        scope_, phi::errors::InvalidArgument("scope can not be nullptr"));
  }

 private:
  bool Initialize(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);
    ps.Add<ConstantFoldingPattern>(
        context, &counter_, place_, scope_, &exe_config_, &deleted_vars_);
    patterns_ = pir::FrozenRewritePatternSet(std::move(ps));
    return true;
  }

  void Run(pir::Operation* op) override {
    size_t op_nums = op->GetParentProgram()->block()->size();
    pir::GreedyRewriteConfig cfg;
    cfg.use_top_down_traversal = true;
    cfg.max_iterations = 10;
    pir::ApplyPatternsGreedily(op->region(0), patterns_, cfg);

    // delete old parameter var
    scope_->EraseVars(deleted_vars_);
    LOG(INFO) << " ------ constant_folding_pass done: [" << counter_ << "/"
              << op_nums << "]";
  }

  bool CanApplyOn(pir::Operation* op) const override {
    // TODO(liuyuanle): remove op->isa<::pir::ModuleOp>()
    return op->isa<::pir::ModuleOp>() && op->num_regions() > 0;
  }

 private:
  size_t counter_{0};
  phi::Place place_;
  paddle::framework::Scope* scope_{nullptr};
  paddle::framework::interpreter::ExecutionConfig exe_config_{};
  std::vector<std::string> deleted_vars_;

  pir::FrozenRewritePatternSet patterns_;
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateConstantFoldingPass(
    const phi::Place& place, paddle::framework::Scope* scope) {
  return std::make_unique<ConstantFoldingPass>(place, scope);
}

}  // namespace pir
