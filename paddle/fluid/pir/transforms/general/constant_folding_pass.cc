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

#include "paddle/fluid/pir/transforms/general/constant_folding_pass.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/new_executor/interpretercore.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/pir/dialect/operator/interface/op_yaml_info.h"
#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/fluid/pir/transforms/pd_op_to_kernel_pass.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/common/errors.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"

#include "paddle/pir/include/core/builder.h"
#include "paddle/pir/include/core/builtin_attribute.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/op_trait.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/core/parameter.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/core/region.h"
#include "paddle/pir/include/core/value.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"
#include "paddle/pir/include/pattern_rewrite/frozen_rewrite_pattern_set.h"
#include "paddle/pir/include/pattern_rewrite/pattern_match.h"
#include "paddle/pir/include/pattern_rewrite/pattern_rewrite_driver.h"

namespace {

class ConstantFoldingPattern : public pir::RewritePattern {
 public:
  ConstantFoldingPattern(
      pir::IrContext* context,
      size_t* suffix,
      const phi::Place& place,
      paddle::framework::Scope* scope,
      paddle::framework::interpreter::ExecutionConfig* exe_config)
      : RewritePattern(MatchAnyOpTypeTag(),
                       1 /*benefit*/,
                       context,
                       {} /*generated_names*/),
        suffix_(suffix),
        place_(place),
        scope_(scope),
        exe_config_(exe_config) {
    exe_config_->create_local_scope = false;
  }

  bool Match(pir::Operation* op) const override {
    VLOG(4) << "constant_folding_pass applies match on [" << op->name()
            << "] op";
    // 1. Some ops do not need to be processed
    if (op->HasTrait<pir::SideEffectTrait>() ||
        op->isa<pir::ConstantTensorOp>() || op->isa<pir::ParameterOp>() ||
        op->isa<paddle::dialect::FeedOp>() ||
        op->isa<paddle::dialect::DataOp>() ||
        op->isa<paddle::dialect::AssignValueOp>()) {
      return false;
    }

    for (uint32_t i = 0; i < op->num_operands(); i++) {
      if (!op->operand_source(i) || !op->operand_source(i).type()) {
        continue;
      }
      // 2. inputs must come from ParameterOp/ConstantTensorOp/CombineOp
      auto* prev_op = pir::GetDefiningOpForInput(op, i);
      if (!prev_op || !(prev_op->isa<pir::ParameterOp>() ||
                        prev_op->isa<pir::ConstantTensorOp>() ||
                        prev_op->isa<pir::CombineOp>())) {
        return false;
      }
      if (prev_op->isa<pir::CombineOp>()) {
        if (prev_op->result(0).use_count() > 1) {
          return false;
        }
        for (uint32_t i = 0; i < prev_op->num_operands(); i++) {
          if (!prev_op->operand_source(i) ||
              !prev_op->operand_source(i).type()) {
            continue;
          }
          // 3. for combine's prev op, inputs must come from
          // ParameterOp/ConstantTensorOp
          auto* prev_prev_op = pir::GetDefiningOpForInput(prev_op, i);
          if (!prev_prev_op || !(prev_prev_op->isa<pir::ParameterOp>() ||
                                 prev_prev_op->isa<pir::ConstantTensorOp>())) {
            return false;
          }
          if (!prev_op->operand_source(i)
                   .type()
                   .isa<paddle::dialect::DenseTensorType>()) {
            return false;
          }
        }
      } else {
        // 4. inputs must be a dense tensor type
        if (!op->operand_source(i)
                 .type()
                 .isa<paddle::dialect::DenseTensorType>()) {
          return false;
        }
      }
    }

    for (uint32_t i = 0; i < op->num_results(); i++) {
      if (!op->result(i) || !op->result(i).type()) {
        continue;
      }
      // 5. outputs must be a dense tensor type
      if (!op->result(i).type().isa<paddle::dialect::DenseTensorType>()) {
        return false;
      }
      // 6. next op should not be a while op
      for (auto it = op->result(i).use_begin(); it != op->result(i).use_end();
           ++it) {
        if (it.owner()->isa<paddle::dialect::WhileOp>()) {
          return false;
        }
      }
    }

    // 7. maybe affect performance
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
    VLOG(4) << "constant_folding_pass applies rewrite on [" << op->name()
            << "] op";
    auto output_var_names = RunOp(op, rewriter);

    // ParameterOp and ConstantTensorOp should be created in the top-level block
    rewriter.SetInsertionPointToStart(
        rewriter.block()->parent_program()->block());

    bool use_parameter_op = ReplaceResultByParameterOp(op);

    for (uint32_t i = 0; i < op->num_results(); i++) {
      if (!op->result(i) || !op->result(i).type()) {
        continue;
      }
      std::string output_var_name = output_var_names[i];
      auto* output_var = scope_->FindVar(output_var_name);
      PADDLE_ENFORCE_NOT_NULL(
          output_var,
          common::errors::InvalidArgument("Parameter var [%s] not in scope.",
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
        }

        auto parameter_op = rewriter.Build<pir::ParameterOp>(
            output_var_name, op->result(i).type());
        parameter_op->set_attribute(
            kAttrIsPersistable,
            rewriter.array_attr({rewriter.bool_attr(true)}));

        rewriter.ReplaceAllUsesWith(op->result(i), parameter_op->result(0));

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
            output_var_name, op->result(i).type());
        constant_op->set_attribute(
            kAttrIsPersistable,
            rewriter.array_attr({rewriter.bool_attr(true)}));

        rewriter.ReplaceAllUsesWith(op->result(i), constant_op->result(0));
      }
    }
    rewriter.EraseOp(op);

    // NOTE(liuyuanle): Here, we release one useless variable after another to
    // effectively reduce peak memory usage.
    if (deleted_vars_.size() > 0) {
      scope_->EraseVars(deleted_vars_);
      deleted_vars_.clear();
    }
    VLOG(4) << "constant_folding_pass applied rewrite on [" << op->name()
            << "] op";
  }

 private:
  bool CheckUseOps(
      const std::vector<std::pair<pir::Operation*, int32_t>>& use_ops) const {
    for (auto [use_op, idx] : use_ops) {
      if (use_op->isa<pir::CombineOp>() ||
          use_op->isa<paddle::dialect::StackOp>()) {
        if (!ReplaceResultByParameterOp(use_op)) {
          return false;
        }
      } else if (use_op->isa<paddle::dialect::MemcpyH2dOp>()) {
        return false;
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
    if (op->isa<paddle::dialect::MemcpyD2hOp>()) {
      return false;
    }
    for (uint32_t i = 0; i < op->num_results(); i++) {
      auto use_ops = pir::GetUseOpsForOutput(op, i);
      if (!CheckUseOps(use_ops)) return false;
    }
    return true;
  }

 protected:
  std::vector<std::string> RunOp(
      pir::Operation* op,
      pir::PatternRewriter& rewriter) const {  // NOLINT
    pir::Program new_program(rewriter.ir_context());
    auto output_var_names =
        BuildProgramFromOperation(op, &new_program, rewriter);

    // execute program
    for (const auto& output_var_name : output_var_names) {
      exe_config_->skip_gc_vars.insert(output_var_name);
    }
    if (VLOG_IS_ON(3)) {
      std::cout << "IR before lowering = " << new_program << std::endl;
    }
    auto kernel_program =
        paddle::dialect::PdOpLowerToKernelPass(&new_program, place_);
    if (VLOG_IS_ON(3)) {
      std::cout << "IR after lowering = " << *kernel_program << std::endl;
    }
    paddle::framework::InterpreterCore core(
        place_, {}, kernel_program->block(), scope_, *exe_config_);
    core.Run({});
    return output_var_names;
  }

  template <typename Op>
  Op BuildParameterOrConstantTensorOP(
      uint32_t index,
      pir::Operation* op,
      pir::Builder& builder,                   // NOLINT
      pir::PatternRewriter& rewriter) const {  // NOLINT
    const auto& var_name =
        pir::GetParameterNameFromValue(op->operand_source(index));
    auto* var = scope_->FindVar(var_name);
    PADDLE_ENFORCE_NOT_NULL(
        var,
        common::errors::InvalidArgument("Persistable var [%s] not in scope.",
                                        var_name));
    auto from_op =
        builder.Build<Op>(var_name, op->operand_source(index).type());
    if (op->operand_source(index).use_count() > 1) {
      from_op->set_attribute(kAttrIsPersistable,
                             rewriter.array_attr({rewriter.bool_attr(true)}));
    } else {
      deleted_vars_.push_back(var_name);
    }
    return from_op;
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
        auto* prev_op = pir::GetDefiningOpForInput(op, i);
        if (prev_op->isa<pir::CombineOp>()) {
          // prepare combine op inputs
          std::vector<pir::Value> combine_op_inputs;
          for (uint32_t j = 0; j < prev_op->num_operands(); j++) {
            auto* prev_prev_op = pir::GetDefiningOpForInput(prev_op, j);
            if (prev_prev_op->isa<pir::ParameterOp>()) {
              auto parameter_op =
                  BuildParameterOrConstantTensorOP<pir::ParameterOp>(
                      j, prev_op, builder, rewriter);
              combine_op_inputs.push_back(parameter_op->result(0));
            } else if (prev_prev_op->isa<pir::ConstantTensorOp>()) {
              auto constant_op =
                  BuildParameterOrConstantTensorOP<pir::ConstantTensorOp>(
                      j, prev_op, builder, rewriter);
              combine_op_inputs.push_back(constant_op->result(0));
            } else {
              PADDLE_THROW(common::errors::Fatal(
                  "Not support %s before builtin.combine op!",
                  prev_prev_op->name()));
            }
          }
          auto combine_op = builder.Build<pir::CombineOp>(combine_op_inputs);
          op_inputs.push_back(combine_op->result(0));
        } else if (prev_op->isa<pir::ParameterOp>()) {
          auto parameter_op =
              BuildParameterOrConstantTensorOP<pir::ParameterOp>(
                  i, op, builder, rewriter);
          op_inputs.push_back(parameter_op->result(0));
        } else if (prev_op->isa<pir::ConstantTensorOp>()) {
          auto constant_op =
              BuildParameterOrConstantTensorOP<pir::ConstantTensorOp>(
                  i, op, builder, rewriter);
          op_inputs.push_back(constant_op->result(0));
        } else {
          PADDLE_THROW(common::errors::Fatal(
              "Not support %s before matched op!", prev_op->name()));
        }
      } else {
        op_inputs.push_back(nullptr);
      }
    }

    // prepare op outputs
    std::vector<pir::Type> op_output_types;
    for (uint32_t i = 0; i < op->num_results(); i++) {
      op_output_types.push_back(op->result(i).type());
    }

    auto* op_copy =
        builder.Build(op_inputs, op->attributes(), op_output_types, op->info());

    std::vector<std::string> output_var_names;
    for (uint32_t i = 0; i < op_copy->num_results(); i++) {
      if (!op_copy->result(i) || !op_copy->result(i).type()) {
        continue;
      }

      auto cast_op = builder.Build<paddle::dialect::CastOp>(
          op_copy->result(i),
          paddle::dialect::GetValueDataType(op_copy->result(i)));

      std::stringstream ss;
      ss << std::chrono::high_resolution_clock::now()
                .time_since_epoch()
                .count();
      std::string output_var_name =
          "constant_folding@_" + ss.str() + std::to_string((*suffix_)++);

      builder.Build<pir::ShadowOutputOp>(cast_op.result(0), output_var_name);
      output_var_names.push_back(output_var_name);
    }

    return output_var_names;
  }

 protected:
  size_t* suffix_;
  phi::Place place_;
  paddle::framework::Scope* scope_;
  paddle::framework::interpreter::ExecutionConfig* exe_config_;
  mutable std::vector<std::string> deleted_vars_;
};

class ConstantFoldingPatternForTrain : public ConstantFoldingPattern {
 public:
  ConstantFoldingPatternForTrain(
      pir::IrContext* context,
      size_t* suffix,
      const phi::Place& place,
      paddle::framework::Scope* scope,
      paddle::framework::interpreter::ExecutionConfig* exe_config)
      : ConstantFoldingPattern(context, suffix, place, scope, exe_config) {}

  bool Match(pir::Operation* op) const override {
    VLOG(4) << "constant_folding_pass applies match on [" << op->name()
            << "] op";
    if (!ConstantFoldingPattern::Match(op)) {
      return false;
    }
    for (uint32_t i = 0; i < op->num_operands(); i++) {
      // inputs must come from or constant op
      auto* prev_op = pir::GetDefiningOpForInput(op, i);
      if (!prev_op || !prev_op->isa<pir::ConstantTensorOp>()) {
        return false;
      }
    }
    return true;
  }

  void Rewrite(pir::Operation* op,
               pir::PatternRewriter& rewriter) const override {  // NOLINT
    VLOG(4) << "constant_folding_pass for train applies rewrite on ["
            << op->name() << "] op";

    auto output_var_names = RunOp(op, rewriter);

    // ConstantTensorOp should be created in the top-level block
    rewriter.SetInsertionPointToStart(
        rewriter.block()->parent_program()->block());

    for (uint32_t i = 0; i < op->num_results(); i++) {
      if (!op->result(i) || !op->result(i).type()) {
        continue;
      }
      std::string output_var_name = output_var_names[i];
      PADDLE_ENFORCE_NOT_NULL(
          scope_->FindVar(output_var_name),
          common::errors::InvalidArgument("Parameter var [%s] not in scope.",
                                          output_var_name));

      auto constant_op = rewriter.Build<pir::ConstantTensorOp>(
          output_var_name, op->result(i).type());
      constant_op->set_attribute(
          kAttrIsPersistable, rewriter.array_attr({rewriter.bool_attr(true)}));

      rewriter.ReplaceAllUsesWith(op->result(i), constant_op->result(0));
    }
    rewriter.EraseOp(op);
    VLOG(4) << "constant_folding_pass for train applied rewrite on ["
            << op->name() << "] op";
  }
};

class ConstantFoldingPass : public pir::Pass {
 public:
  ConstantFoldingPass() : pir::Pass("constant_folding_pass", 1) {}

 private:
  bool Initialize(pir::IrContext* context) override {
    PADDLE_ENFORCE_EQ(
        Has(pir::Pass::kPlaceAttr),
        true,
        common::errors::InvalidArgument(
            "Pass initialize failed."
            "When using ConstantFoldingPass, place attribute is required!"
            "Use Set method to set the place attribute."));
    PADDLE_ENFORCE_EQ(
        Has(pir::Pass::kParamScopeAttr),
        true,
        common::errors::InvalidArgument(
            "Pass initialize failed."
            "When using ConstantFoldingPass, scope attribute is required!"
            "Use Set method to set the scope attribute."));

    place_ = Get<phi::Place>(pir::Pass::kPlaceAttr);
    scope_ = &Get<paddle::framework::Scope>(pir::Pass::kParamScopeAttr);

    PADDLE_ENFORCE_NOT_NULL(
        scope_, common::errors::InvalidArgument("scope can not be nullptr"));

    pir::RewritePatternSet ps(context);

    if (Has("train_mode") && Get<bool>("train_mode")) {
      ps.Add<ConstantFoldingPatternForTrain>(
          context, &suffix_, phi::CPUPlace{}, scope_, &exe_config_);
    } else {
      ps.Add<ConstantFoldingPattern>(
          context, &suffix_, place_, scope_, &exe_config_);
    }
    patterns_ = pir::FrozenRewritePatternSet(std::move(ps));
    return true;
  }

  void Run(pir::Operation* op) override {
    int64_t num_ops{0};
    for (uint32_t i = 0; i < op->num_regions(); ++i) {
      auto& region = op->region(i);
      for (auto& block : region) {
        num_ops += block.size();
      }
    }
    pir::GreedyRewriteConfig cfg;
    cfg.use_top_down_traversal = true;
    cfg.max_iterations = 10;
    auto [_, num_rewrites] = pir::ApplyPatternsGreedily(op, patterns_, cfg);
    AddStatistics(num_rewrites, num_ops);
  }

 private:
  size_t suffix_{0};
  phi::Place place_{phi::CPUPlace{}};
  paddle::framework::Scope* scope_{nullptr};
  paddle::framework::interpreter::ExecutionConfig exe_config_{};

  pir::FrozenRewritePatternSet patterns_;
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateConstantFoldingPass() {
  return std::make_unique<ConstantFoldingPass>();
}

}  // namespace pir

REGISTER_IR_PASS(constant_folding_pass, ConstantFoldingPass);
