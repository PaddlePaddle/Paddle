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
    VLOG(4) << "constant_folding_pass applys match on [" << op->name()
            << "] op";
    // 1. Some ops do not need to be processed
    if (op->HasTrait<pir::SideEffectTrait>() ||
        op->isa<pir::ConstantTensorOp>() || op->isa<pir::ParameterOp>() ||
        op->isa<paddle::dialect::FeedOp>()) {
      return false;
    }

    for (uint32_t i = 0; i < op->num_operands(); i++) {
      if (!op->operand_source(i) || !op->operand_source(i).type()) {
        continue;
      }
      // 2. inputs must come from parameter op or constant op
      if (!(pir::GetDefiningOpForInput(op, i)->isa<pir::ParameterOp>() ||
            pir::GetDefiningOpForInput(op, i)->isa<pir::ConstantTensorOp>())) {
        return false;
      }
      // 3. inputs must be a dense tensor type
      if (!op->operand_source(i)
               .type()
               .isa<paddle::dialect::DenseTensorType>()) {
        return false;
      }
    }

    for (uint32_t i = 0; i < op->num_results(); i++) {
      if (!op->result(i) || !op->result(i).type()) {
        continue;
      }
      // 4. outputs must be a dense tensor type
      if (!op->result(i).type().isa<paddle::dialect::DenseTensorType>()) {
        return false;
      }
    }

    // 5. maybe affect performence
    if (op->isa<paddle::dialect::FullOp>()) {
      auto next_ops = pir::GetUseOpsForOutput(op, 0);
      for (auto [next_op, _] : next_ops) {
        if (next_op->isa<paddle::dialect::FullWithTensorOp>() ||
            next_op->isa<paddle::dialect::LinspaceOp>()) {
          return false;
        }
      }
    }

    VLOG(4) << "constant_folding_pass applied match on [" << op->name()
            << "] op";
    return true;
  }

  void Rewrite(pir::Operation* op,
               pir::PatternRewriter& rewriter) const override {  // NOLINT
    VLOG(4) << "constant_folding_pass applys rewrite on [" << op->name()
            << "] op";
    pir::Program new_program(rewriter.ir_context());
    auto output_var_names =
        BuildProgramFromOperation(op, &new_program, rewriter);

    // execute program
    for (auto output_var_name : output_var_names) {
      exe_config_->skip_gc_vars.insert(output_var_name);
    }
    auto kernel_program =
        paddle::dialect::PdOpLowerToKernelPass(&new_program, place_);
    paddle::framework::InterpreterCore core(
        place_, {}, kernel_program->block(), scope_, *exe_config_);

    core.Run({});

    rewriter.SetInsertionPointToStart(rewriter.block());

    bool use_parameter_op = ReplaceResultByParameterOp(op);

    for (uint32_t i = 0; i < op->num_results(); i++) {
      if (!op->result(i) || !op->result(i).type()) {
        continue;
      }
      std::string output_var_name = output_var_names[i];
      auto* output_var = scope_->FindVar(output_var_name);
      PADDLE_ENFORCE_NOT_NULL(
          output_var,
          phi::errors::InvalidArgument("Parameter var [%s] not in scope.",
                                       output_var_name));

      if (use_parameter_op) {
        if (output_var->IsType<phi::DenseTensor>()) {
          auto* output_tensor = output_var->GetMutable<phi::DenseTensor>();
          if (output_tensor->IsInitialized() &&
              output_tensor->place().GetType() != place_.GetType()) {
            phi::DenseTensor temp_tensor;
            temp_tensor.Resize(output_tensor->dims());
            paddle::framework::TensorCopySync(
                *output_tensor, phi::CPUPlace{}, &temp_tensor);
            output_tensor->clear();
            paddle::framework::TensorCopySync(
                temp_tensor, place_, output_tensor);
          }
          auto parameter_op = rewriter.Build<pir::ParameterOp>(
              output_var_name, op->result(i).type());
          parameter_op->set_attribute(
              kAttrIsPersisable,
              rewriter.array_attr({rewriter.bool_attr(true)}));

          rewriter.ReplaceAllUsesWith(op->result(i), parameter_op->result(0));
        }
      } else {
        if (output_var->IsType<phi::DenseTensor>()) {
          auto* output_tensor = output_var->GetMutable<phi::DenseTensor>();
          if (output_tensor->place().GetType() != phi::AllocationType::CPU) {
            phi::DenseTensor temp_tensor;
            temp_tensor.Resize(output_tensor->dims());
            paddle::framework::TensorCopySync(
                *output_tensor, phi::CPUPlace{}, &temp_tensor);
            output_tensor->clear();
            paddle::framework::TensorCopySync(
                temp_tensor, phi::CPUPlace{}, output_tensor);
          }
        }

        auto constant_op = rewriter.Build<pir::ConstantTensorOp>(
            rewriter.tensor_name_attr(output_var_name), op->result(i).type());
        constant_op->set_attribute(
            kAttrIsPersisable, rewriter.array_attr({rewriter.bool_attr(true)}));

        rewriter.ReplaceAllUsesWith(op->result(i), constant_op->result(0));
      }
    }
    rewriter.EraseOp(op);
    VLOG(4) << "constant_folding_pass applied rewrite on [" << op->name()
            << "] op";
  }

 private:
  bool CheckUseOps(
      const std::vector<std::pair<pir::Operation*, int32_t>>& use_ops) const {
    for (auto [use_op, idx] : use_ops) {
      if (use_op->isa<pir::CombineOp>()) {
        if (!ReplaceResultByParameterOp(use_op)) return false;
      } else if (use_op->HasInterface<paddle::dialect::OpYamlInfoInterface>()) {
        auto [input_infos, _1, _2, _3, _4] =
            use_op->dyn_cast<paddle::dialect::OpYamlInfoInterface>()
                .GetOpInfo();
        if (input_infos[idx].type_name.find("IntArrayAttribute") !=
                std::string::npos ||
            input_infos[idx].type_name.find("ScalarAttribute") !=
                std::string::npos) {
          return false;
        }
      }
    }
    return true;
  }

  bool ReplaceResultByParameterOp(pir::Operation* op) const {
    for (uint32_t i = 0; i < op->num_results(); i++) {
      auto use_ops = pir::GetUseOpsForOutput(op, i);
      if (!CheckUseOps(use_ops)) return false;
    }
    return true;
  }

  std::vector<std::string> BuildProgramFromOperation(
      pir::Operation* op,
      pir::Program* new_program,
      pir::PatternRewriter& rewriter) const {  // NOLINT
    pir::Builder builder =
        pir::Builder(rewriter.ir_context(), new_program->block());

    // prepare op inputs
    std::vector<pir::Value> op_inputs;
    for (uint32_t i = 0; i < op->num_operands(); i++) {
      if (op->operand_source(i)) {
        const auto& param_name =
            pir::GetParameterNameFromValue(op->operand_source(i));
        auto* param_var = scope_->FindVar(param_name);
        PADDLE_ENFORCE_NOT_NULL(
            param_var,
            phi::errors::InvalidArgument("Parameter var [%s] not in scope.",
                                         param_name));

        auto parameter_op = builder.Build<pir::ParameterOp>(
            param_name, op->operand_source(i).type());
        if (op->operand_source(i).use_count() <= 1) {
          deleted_vars_->push_back(param_name);
        } else {
          parameter_op->set_attribute(
              kAttrIsPersisable,
              rewriter.array_attr({rewriter.bool_attr(true)}));
        }
        op_inputs.push_back(parameter_op->result(0));
      } else {
        op_inputs.push_back(
            op->operand_source(i).dyn_cast<pir::OpResult>() /*nullptr*/);
      }
    }

    // prepare op outputs
    std::vector<pir::Type> output_types;
    for (uint32_t i = 0; i < op->num_results(); i++) {
      output_types.push_back(op->result(i).type());
    }

    auto* temp_op =
        builder.Build(op_inputs, op->attributes(), output_types, op->info());

    std::vector<std::string> output_var_names;
    for (uint32_t i = 0; i < op->num_results(); i++) {
      if (!temp_op->result(i) || !temp_op->result(i).type()) {
        continue;
      }
      std::stringstream ss;
      ss << std::chrono::high_resolution_clock::now()
                .time_since_epoch()
                .count();
      std::string output_var_name =
          "constant_folding@_" + ss.str() + std::to_string((*counter_)++);

      builder.Build<pir::ShadowOutputOp>(temp_op->result(i), output_var_name);
      output_var_names.push_back(output_var_name);
    }

    return output_var_names;
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
    pir::ApplyPatternsGreedily(op, patterns_, cfg);
    PrintStatistics(counter_, op_nums);
    // delete old parameter var
    scope_->EraseVars(deleted_vars_);
    if (place_.GetType() != phi::AllocationType::CPU) {
      paddle::memory::Release(place_);
    }
    paddle::memory::Release(phi::CPUPlace{});
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
