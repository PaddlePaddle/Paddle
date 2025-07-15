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

#include "paddle/ap/include/paddle/pass/move_trivial_fusion_range_to_fusion_op_pass.h"
#include "paddle/ap/include/paddle/hlir/manual_op.h"

#include "paddle/ap/include/axpr/abstract_list.h"
#include "paddle/ap/include/axpr/anf_expr_util.h"
#include "paddle/ap/include/axpr/atomic.h"
#include "paddle/ap/include/axpr/builtin_frame_util.h"
#include "paddle/ap/include/axpr/builtin_serializable_attr_map_to_axpr_helper.h"
#include "paddle/ap/include/axpr/data_type_util.h"
#include "paddle/ap/include/axpr/interpreter.h"
#include "paddle/ap/include/axpr/lambda_expr_builder.h"
#include "paddle/ap/include/paddle/builtin_frame_util.h"
#include "paddle/ap/include/paddle/pir/ap_pir_attribute.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/utils/general_functions.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/ir_mapping.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace ap::paddle {

namespace adt = ap::adt;

namespace {

template <typename ContainerOp>
class MoveTrivialFusionRangeToContainerOpPattern
    : public pir::OpRewritePattern<::paddle::dialect::ApTrivialFusionEndOp> {
 public:
  using pir::OpRewritePattern<
      ::paddle::dialect::ApTrivialFusionEndOp>::OpRewritePattern;

  bool MatchAndRewrite(
      ::paddle::dialect::ApTrivialFusionEndOp ap_trivial_fusion_end_op,
      pir::PatternRewriter& rewriter) const override {
    const auto& ret = TryMatchAndRewrite(ap_trivial_fusion_end_op, &rewriter);
    PADDLE_ENFORCE_EQ(
        ret.HasError(),
        false,
        common::errors::Fatal(
            "MoveTrivialFusionRangeToFusionOpPattern::MatchAndRewrite failed. "
            "\nTraceback (most recent call "
            "last):\n%s\n%s: %s. ",
            ret.GetError().CallStackToString(),
            ret.GetError().class_name(),
            ret.GetError().msg()));
    return ret.GetOkValue();
  }

  adt::Result<bool> TryMatchAndRewrite(
      ::paddle::dialect::ApTrivialFusionEndOp ap_trivial_fusion_end_op,
      pir::PatternRewriter* rewriter) const {
    rewriter->SetInsertionPointAfter(ap_trivial_fusion_end_op);
    ADT_LET_CONST_REF(ap_trivial_fusion_begin_op,
                      GetApTrivialFusionBeginOp(ap_trivial_fusion_end_op));
    ADT_LET_CONST_REF(
        old_outputs,
        GetUsedOutputs(ap_trivial_fusion_begin_op, ap_trivial_fusion_end_op));
    auto fusion_op = rewriter->Build<ContainerOp>([&] {
      std::vector<pir::Type> output_types{};
      output_types.reserve(old_outputs.size());
      for (pir::Value output : old_outputs) {
        output_types.emplace_back(output.type());
      }
      return output_types;
    }());
    pir::IrMapping ir_mapping{};
    {
      ADT_LET_CONST_REF(external_inputs,
                        GetExternalInputs(ap_trivial_fusion_begin_op,
                                          ap_trivial_fusion_end_op));
      for (pir::Value value : external_inputs) {
        ir_mapping.Add(value, value);
      }
    }
    std::list<pir::Operation*> reversed_old_ops;
    {
      auto clone_options = pir::CloneOptions(true, true, true);
      for (auto iter =
               ++ap_trivial_fusion_begin_op->operator pir::Block::Iterator();
           iter != ap_trivial_fusion_end_op->operator pir::Block::Iterator();
           ++iter) {
        fusion_op.block()->push_back(iter->Clone(ir_mapping, clone_options));
        reversed_old_ops.push_front(iter);
      }
    }
    {
      std::vector<pir::Value> yield_inputs{};
      yield_inputs.reserve(fusion_op->num_results());
      for (int i = 0; i < fusion_op->num_results(); ++i) {
        yield_inputs.push_back(ir_mapping.Lookup(old_outputs.at(i)));
      }
      pir::Builder builder{pir::IrContext::Instance(), fusion_op.block()};
      builder.Build<pir::YieldOp>(yield_inputs);
    }
    for (int i = 0; i < fusion_op->num_results(); ++i) {
      rewriter->ReplaceAllUsesWith(old_outputs.at(i), fusion_op->result(i));
    }
    {
      rewriter->EraseOp(ap_trivial_fusion_end_op);
      for (auto* old_op : reversed_old_ops) {
        rewriter->EraseOp(old_op);
      }
      rewriter->EraseOp(ap_trivial_fusion_begin_op);
    }
    return true;
  }

  adt::Result<::paddle::dialect::ApTrivialFusionBeginOp>
  GetApTrivialFusionBeginOp(
      ::paddle::dialect::ApTrivialFusionEndOp ap_trivial_fusion_end_op) const {
    auto* block = ap_trivial_fusion_end_op->GetParent();
    auto ap_trivial_fusion_end_iter =
        ap_trivial_fusion_end_op->operator pir::Block::Iterator();
    ADT_CHECK(ap_trivial_fusion_end_iter != block->begin());
    std::size_t depth = 1;
    auto iter = ap_trivial_fusion_end_iter;
    do {
      --iter;
      if (iter->isa<::paddle::dialect::ApTrivialFusionEndOp>()) {
        ++depth;
      } else if (iter->isa<::paddle::dialect::ApTrivialFusionBeginOp>()) {
        --depth;
        if (depth == 0)
          return iter->dyn_cast<::paddle::dialect::ApTrivialFusionBeginOp>();
      } else {
        // Do nothing.
      }
    } while (iter != block->begin());
    return adt::errors::NotImplementedError{
        "no pd_op.ap_trivial_fusion_begin matched."};
  }

  adt::Result<std::unordered_set<pir::Value>> GetExternalInputs(
      ::paddle::dialect::ApTrivialFusionBeginOp ap_trivial_fusion_begin_op,
      ::paddle::dialect::ApTrivialFusionEndOp ap_trivial_fusion_end_op) const {
    using IterT = pir::Block::Iterator;
    ADT_LET_CONST_REF(
        all_inputs,
        GetInputsInRange(
            ++ap_trivial_fusion_begin_op->operator IterT(),
            ap_trivial_fusion_end_op->operator pir::Block::Iterator()));
    ADT_LET_CONST_REF(
        all_outputs,
        GetOutputsInRange(
            ++ap_trivial_fusion_begin_op->operator IterT(),
            ap_trivial_fusion_end_op->operator pir::Block::Iterator()));
    std::unordered_set<pir::Value> ret;
    for (pir::Value input : all_inputs) {
      if (std::find(all_outputs.begin(), all_outputs.end(), input) ==
          all_outputs.end()) {
        ret.insert(input);
      }
    }
    return ret;
  }

  adt::Result<std::vector<pir::Value>> GetUsedOutputs(
      ::paddle::dialect::ApTrivialFusionBeginOp ap_trivial_fusion_begin_op,
      ::paddle::dialect::ApTrivialFusionEndOp ap_trivial_fusion_end_op) const {
    using IterT = pir::Block::Iterator;
    ADT_LET_CONST_REF(
        tmp_and_outputs,
        GetOutputsInRange(
            (++ap_trivial_fusion_begin_op->operator IterT()),
            ap_trivial_fusion_end_op->operator pir::Block::Iterator()));
    ADT_LET_CONST_REF(inputs_after_end,
                      GetInputsAfter(ap_trivial_fusion_end_op));
    std::vector<pir::Value> used_outputs;
    used_outputs.reserve(tmp_and_outputs.size());
    for (pir::Value value : tmp_and_outputs) {
      if (inputs_after_end.count(value)) {
        used_outputs.push_back(value);
      }
    }
    return used_outputs;
  }

  adt::Result<std::vector<pir::Value>> GetOutputsBefore(
      pir::Operation* op) const {
    auto* block = op->GetParent();
    pir::Block::Iterator end = op->operator pir::Block::Iterator();
    return GetOutputsInRange(block->begin(), end);
  }

  adt::Result<std::vector<pir::Value>> GetOutputsInRange(
      pir::Block::Iterator begin, pir::Block::Iterator end) const {
    std::vector<pir::Value> outputs;
    for (auto iter = begin; iter != end; ++iter) {
      for (int i = 0; i < iter->num_results(); ++i) {
        outputs.push_back(iter->result(i));
      }
    }
    return outputs;
  }

  adt::Result<std::unordered_set<pir::Value>> GetInputsAfter(
      pir::Operation* op) const {
    auto* block = op->GetParent();
    using IterT = pir::Block::Iterator;
    auto pos = op->operator IterT();
    return GetInputsInRange((++pos), block->end());
  }

  adt::Result<std::unordered_set<pir::Value>> GetInputsInRange(
      pir::Block::Iterator begin, pir::Block::Iterator end) const {
    std::unordered_set<pir::Value> inputs;
    auto TryInsertInput = [&](pir::Value input) {
      if (!input) return;
      inputs.insert(input);
    };
    for (auto iter = begin; iter != end; ++iter) {
      for (int i = 0; i < iter->num_operands(); ++i) {
        TryInsertInput(iter->operand_source(i));
      }
      for (pir::Value free_value : pir::GetUsedExternalValue(*iter)) {
        TryInsertInput(free_value);
      }
    }
    return inputs;
  }
};

class MoveTrivialFusionRangeToFusionOpPass : public pir::PatternRewritePass {
 public:
  MoveTrivialFusionRangeToFusionOpPass()
      : pir::PatternRewritePass("move_trivial_fusion_range_to_fusion_op", 1) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);
    ps.Add<
        MoveTrivialFusionRangeToContainerOpPattern<::cinn::dialect::FusionOp>>(
        context);
    return ps;
  }
};

class MoveTrivialFusionRangeToGroupOpPass : public pir::PatternRewritePass {
 public:
  MoveTrivialFusionRangeToGroupOpPass()
      : pir::PatternRewritePass("move_trivial_fusion_range_to_group_op", 1) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);
    ps.Add<
        MoveTrivialFusionRangeToContainerOpPattern<::cinn::dialect::GroupOp>>(
        context);
    return ps;
  }
};

}  // namespace

std::unique_ptr<::pir::Pass> CreateMoveTrivialFusionRangeToFusionOpPass() {
  return std::make_unique<MoveTrivialFusionRangeToFusionOpPass>();
}

}  // namespace ap::paddle
