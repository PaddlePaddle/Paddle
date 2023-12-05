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

#pragma once

#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/cinn_group_lowering_pass.h"

#include <unordered_map>

#include "paddle/cinn/adt/generate_map_expr.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_attribute.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/op_with_group_merge_pass.h"
#include "paddle/cinn/hlir/dialect/runtime/ir/jit_kernel_op.h"
#include "paddle/cinn/hlir/dialect/runtime/ir/runtime_dialect.h"
#include "paddle/cinn/hlir/framework/pir_compiler.h"
#include "paddle/cinn/runtime/flags.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/core/program.h"
#include "paddle/pir/dialect/control_flow/ir/cf_op.h"
#include "paddle/pir/pass/pass_registry.h"
#include "paddle/pir/pattern_rewrite/frozen_rewrite_pattern_set.h"

PD_DECLARE_bool(cinn_enable_map_expr);

namespace {

std::vector<pir::Value> GetBlockOutsideInput(
    const std::vector<pir::Operation*> op_list) {
  std::vector<pir::Value> vec_res;
  std::unordered_set<::pir::Value> block_inner_output;
  for (size_t k = 0; k < op_list.size(); ++k) {
    for (size_t i = 0; i < op_list[k]->num_results(); ++i) {
      block_inner_output.insert(op_list[k]->result(i));
    }
  }

  std::unordered_set<::pir::Value> insert_value;
  for (size_t k = 0; k < op_list.size(); ++k) {
    for (size_t i = 0; i < op_list[k]->num_operands(); ++i) {
      if (!block_inner_output.count(op_list[k]->operand_source(i)) &&
          !insert_value.count(op_list[k]->operand_source(i))) {
        vec_res.push_back(op_list[k]->operand_source(i));
        insert_value.insert(op_list[k]->operand_source(i));
      }
    }
  }
  return vec_res;
}

std::vector<pir::Value> GetBlockOutsideOutput(
    const std::vector<pir::Operation*> op_list,
    const std::vector<pir::Operation*> group_all_list) {
  assert(group_all_list.size() >= 2);
  assert(group_all_list.back()->isa<pir::YieldOp>());

  auto yeild_op = group_all_list.back()->dyn_cast<pir::YieldOp>();

  std::unordered_set<pir::Value> yeild_inputs;
  for (size_t i = 0; i < yeild_op.num_operands(); ++i) {
    yeild_inputs.insert(yeild_op.operand_source(i));
  }

  std::unordered_set<pir::Operation*> innner_op_set(op_list.begin(),
                                                    op_list.end());
  std::unordered_set<pir::Operation*> outside_group_set;

  for (size_t i = 0; i < group_all_list.size(); ++i) {
    if (!innner_op_set.count(group_all_list[i])) {
      outside_group_set.insert(group_all_list[i]);
    }
  }

  std::vector<pir::Value> vec_res;

  for (auto* op : op_list) {
    for (size_t i = 0; i < op->num_results(); ++i) {
      if (yeild_inputs.count(op->result(i))) {
        vec_res.push_back(op->result(i));
      } else {
        for (auto it = op->result(i).use_begin(); it != op->result(i).use_end();
             ++it) {
          if (outside_group_set.count(it->owner())) {
            vec_res.push_back(op->result(i));
            break;
          }
        }
      }
    }
  }
  return vec_res;
}

std::vector<pir::Operation*> GetOpListNotIncludeYield(
    const std::vector<pir::Operation*>& op_list) {
  std::vector<pir::Operation*> vec_res;
  for (size_t i = 0; i < op_list.size(); ++i) {
    if (!op_list[i]->isa<pir::YieldOp>()) {
      vec_res.push_back(op_list[i]);
    }
  }

  return vec_res;
}

std::vector<pir::Operation*> GetOutputOpList(
    const std::vector<pir::Operation*>& op_list) {
  std::vector<pir::Operation*> vec_res;
  auto yield_op = op_list.back();

  for (size_t i = 0; i < yield_op->num_operands(); ++i) {
    vec_res.push_back(
        yield_op->operand(i).source().dyn_cast<pir::OpResult>().owner());
  }

  return vec_res;
}

class GroupOpPattern : public pir::OpRewritePattern<cinn::dialect::GroupOp> {
 public:
  GroupOpPattern(
      ::pir::IrContext* context,
      const std::shared_ptr<pir::ShapeConstraintIRAnalysis>& shape_analysis)
      : pir::OpRewritePattern<cinn::dialect::GroupOp>(context),
        shape_analysis_(shape_analysis) {}

  bool MatchAndRewrite(cinn::dialect::GroupOp group_op,
                       pir::PatternRewriter& rewriter) const override {
    ::pir::IrContext* ctx = ::pir::IrContext::Instance();
    auto target = cinn::common::DefaultNVGPUTarget();
    auto* program = group_op->GetParentProgram();
    VLOG(4) << "Before GroupOpPattern: " << *program;
    // TODO(Aurelius84): Remove scope after cleaning PirCompiler usless Build
    // Interface
    auto scope = std::make_shared<cinn::hlir::framework::Scope>();

    VLOG(4) << "start Lowering Group Op: " << group_op;
    // using yield op to sort
    std::unordered_map<::pir::Value, size_t> value2id;
    auto yeild_op = group_op.ops().back();
    for (size_t i = 0; i < yeild_op->num_operands(); ++i) {
      value2id[yeild_op->operand_source(i)] = i;
    }
    std::unordered_map<pir::Value, pir::Value> value_map;

    // op fusion
    auto op_fusion = cinn::dialect::ir::OpFusionPassInternal(
        GetOpListNotIncludeYield(group_op.ops()),
        GetOutputOpList(group_op.ops()),
        shape_analysis_);

    // fusion merge
    auto group_list = cinn::dialect::ir::GeneralFusionMergePassInternal(
        op_fusion, shape_analysis_);

    for (auto group : group_list) {
      auto ir_compiler = cinn::hlir::framework::PirCompilerManager::Create(
          *program, target, scope);
      group->shape_analysis = shape_analysis_;
      if (FLAGS_cinn_enable_map_expr) {
        cinn::adt::TryGenerateMapExprFromGroup(group);
      }

      auto fn_ptr_res = ir_compiler->BuildCUDAJITInfo({group});
      std::unordered_map<std::string, ::pir::Attribute> op_attrs{
          {cinn::dialect::JitKernelOp::kAttrName,
           cinn::dialect::CINNKernelInfoAttribute::get(ctx, fn_ptr_res[0])},
      };

      // Generate jit kernel op input and output
      auto vec_ins = GetBlockOutsideInput(group->ops);
      for (size_t i = 0; i < vec_ins.size(); ++i) {
        if (value_map.find(vec_ins[i]) != value_map.end()) {
          vec_ins[i] = value_map.at(vec_ins[i]);
        }
      }

      std::vector<pir::Type> vec_types;
      for (size_t i = 0; i < group->output_values.size(); ++i) {
        vec_types.push_back(group->output_values[i].type());
      }

      auto jit_kernel_op = rewriter.Build<cinn::dialect::JitKernelOp>(
          vec_ins, op_attrs, vec_types);
      for (size_t i = 0; i < jit_kernel_op.num_results(); ++i) {
        auto find_it = value2id.find(group->output_values[i]);
        if (find_it != value2id.end()) {
          rewriter.ReplaceAllUsesWith(group_op.result(find_it->second),
                                      jit_kernel_op.result(i));
        }
        value_map[group->output_values[i]] = jit_kernel_op.result(i);
      }
    }
    value_map.clear();
    VLOG(4) << "Before GroupOpPattern.EraseOp: " << *program;
    rewriter.EraseOp(group_op);
    return true;
  }

 private:
  std::shared_ptr<pir::ShapeConstraintIRAnalysis> shape_analysis_;
};

class CinnGroupLoweringPass : public pir::PatternRewritePass {
 public:
  CinnGroupLoweringPass(
      const std::shared_ptr<pir::ShapeConstraintIRAnalysis>& shape_analysis)
      : pir::PatternRewritePass("cinn_group_lowering", 1),
        shape_analysis_(shape_analysis) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    context->GetOrRegisterDialect<cinn::dialect::RuntimeDialect>();
    context->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();
    context->GetOrRegisterDialect<paddle::dialect::KernelDialect>();

    pir::RewritePatternSet ps(context);
    ps.Add<GroupOpPattern>(context, shape_analysis_);

    return ps;
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->isa<pir::ModuleOp>() && op->num_regions() > 0;
  }

 private:
  const std::shared_ptr<pir::ShapeConstraintIRAnalysis>& shape_analysis_;
};

}  // namespace

namespace cinn {
namespace dialect {
namespace ir {

std::unique_ptr<::pir::Pass> CreateCinnGroupLoweringPass(
    const std::shared_ptr<pir::ShapeConstraintIRAnalysis>& shape_analysis) {
  return std::make_unique<CinnGroupLoweringPass>(shape_analysis);
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn

// REGISTER_IR_PASS(cinn_group_lowering, CinnGroupLoweringPass);
