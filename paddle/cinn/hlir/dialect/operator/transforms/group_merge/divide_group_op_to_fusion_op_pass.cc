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

#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/divide_group_op_to_fusion_op_pass.h"

#include <unordered_map>

#include "paddle/cinn/adt/generate_map_expr.h"
#include "paddle/cinn/common/broadcast_tree.h"
#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/generate_shape_util.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_attribute.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/op_with_group_merge_pass.h"
#include "paddle/cinn/hlir/dialect/runtime/ir/runtime_dialect.h"
#include "paddle/cinn/hlir/framework/pir_compiler.h"
#include "paddle/cinn/runtime/flags.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"
#include "paddle/pir/include/pass/pass_registry.h"
#include "paddle/pir/include/pattern_rewrite/frozen_rewrite_pattern_set.h"

#include "paddle/fluid/pir/dialect/operator/ir/manual_op.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr.h"

PD_DECLARE_bool(cinn_enable_map_expr);

namespace {
using GroupPtr = cinn::dialect::ir::GroupPtr;

std::vector<pir::Value> GetBlockOutsideOutput(
    const std::vector<pir::Operation*>& op_list) {
  std::vector<pir::Value> vec_res;

  std::unordered_set<pir::Value> block_inner_inputs;
  for (auto op : op_list) {
    for (size_t i = 0; i < op->num_operands(); ++i) {
      block_inner_inputs.insert(op->operand_source(i));
    }
  }

  std::unordered_set<pir::Value> insert_value;
  for (auto op : op_list) {
    for (size_t i = 0; i < op->num_results(); ++i) {
      if (!block_inner_inputs.count(op->result(i)) &&
          !insert_value.count(op->result(i))) {
        vec_res.push_back(op->result(i));
        insert_value.insert(op->result(i));
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
    vec_res.push_back(yield_op->operand(i).source().defining_op());
  }

  return vec_res;
}

class GroupOpPattern : public pir::OpRewritePattern<cinn::dialect::GroupOp> {
 public:
  explicit GroupOpPattern(::pir::IrContext* context)
      : pir::OpRewritePattern<cinn::dialect::GroupOp>(context) {}

  bool MatchAndRewrite(cinn::dialect::GroupOp group_op,
                       pir::PatternRewriter& rewriter) const override {
    ::pir::IrContext* ctx = ::pir::IrContext::Instance();
    auto* program = group_op->GetParentProgram();
    VLOG(4) << "Before GroupOpPattern: " << *program;

    // step 1: op fusion & fusion merge
    auto group_list = cinn::dialect::ir::OpFusionPassInternal(
        GetOpListNotIncludeYield(group_op.GetOperators()),
        GetOutputOpList(group_op.GetOperators()));
    auto merged_group_list =
        cinn::dialect::ir::GeneralFusionMergePassInternal(group_list);

    // step 2: Prepare necessary map information for ReplaceAllUsesWith and
    // infer dynamic symbolic shape.
    const auto output_value2id = [&]() {
      std::unordered_map<::pir::Value, size_t> output_value2id;
      auto yield_op = group_op.GetOperators().back();
      for (size_t i = 0; i < yield_op->num_operands(); ++i) {
        output_value2id[yield_op->operand_source(i)] = i;
      }
      return output_value2id;
    }();

    auto& shape_analysis =
        pir::ShapeAnalysisManager::Instance().Get(group_op->GetParentProgram());
    // Record map info for yield value to each fusion_op's result
    std::unordered_map<::pir::Value, ::pir::Value> fusion_yield_values;

    const auto& TryReplaceOperandSource = [&](::pir::Operation* op) {
      for (auto& operand : op->operands()) {
        const auto value = operand.source();
        if (fusion_yield_values.find(value) != fusion_yield_values.end()) {
          operand.set_source(fusion_yield_values.at(value));
        }
      }
    };

    const auto& CreateFusionOp =
        [&](const std::vector<pir::Value>& vec_outs,
            GroupPtr& group) -> cinn::dialect::FusionOp {
      std::vector<pir::Type> output_types;
      for (auto& value : vec_outs) {
        output_types.emplace_back(value.type());
      }
      auto fusion_op = rewriter.Build<cinn::dialect::FusionOp>(output_types);
      pir::Block* fusion_block = fusion_op.block();
      for (auto op : group->ops) {
        TryReplaceOperandSource(op);
        op->MoveTo(fusion_block, fusion_block->end());
      }
      return fusion_op;
    };

    // step 3: Create Fusion Op for each divided sub group.
    for (auto group : merged_group_list) {
      const std::vector<::pir::Value> vec_outs =
          group->GenerateGroupOutputValues();
      auto fusion_op = CreateFusionOp(vec_outs, group);

      for (size_t i = 0; i < fusion_op.num_results(); ++i) {
        CHECK(fusion_yield_values.insert({vec_outs[i], fusion_op.result(i)})
                  .second)
            << "fusion_yield_values already has key!";
        const auto& shape_expr =
            shape_analysis.GetShapeOrDataForValue(vec_outs[i]);
        // TODO(Hongqing-work): delete this after fix bug of
        // cinn_dynamic_reshape_op_pass
        shape_analysis.SetShapeOrDataForValue(fusion_op.result(i), shape_expr);
        auto find_it = output_value2id.find(vec_outs[i]);
        if (find_it != output_value2id.end()) {
          // If it's an output of group_op, YieldOp is needed to find the real
          // user
          rewriter.ReplaceAllUsesWith(group_op.result(find_it->second),
                                      fusion_op.result(i));
        }
      }
      rewriter.SetInsertionPointToBlockEnd(fusion_op.block());
      rewriter.Build<::pir::YieldOp>(vec_outs);
      rewriter.SetInsertionPointAfter(fusion_op);
    }
    rewriter.EraseOp(group_op);
    VLOG(4) << "after GroupOpPattern: " << *program;
    return true;
  }
};

class DivideGroupOpToFusionOpPass : public pir::PatternRewritePass {
 public:
  DivideGroupOpToFusionOpPass()
      : pir::PatternRewritePass("divide_group_op_to_fusion_op", 1) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    context->GetOrRegisterDialect<cinn::dialect::RuntimeDialect>();
    context->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();
    context->GetOrRegisterDialect<paddle::dialect::KernelDialect>();

    pir::RewritePatternSet ps(context);
    ps.Add<GroupOpPattern>(context);

    return ps;
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->num_regions() > 0;
  }
};

}  // namespace

namespace cinn {
namespace dialect {
namespace ir {

std::unique_ptr<::pir::Pass> CreateDivideGroupOpToFusionOpPass() {
  return std::make_unique<DivideGroupOpToFusionOpPass>();
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
