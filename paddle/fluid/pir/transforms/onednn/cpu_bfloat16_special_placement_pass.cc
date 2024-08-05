// REGISTER_IR_PASS(cpu_bfloat16_placement_pass, OneDNNPlacementBf16Pass);
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

#include "paddle/fluid/pir/transforms/onednn/cpu_bfloat16_special_placement_pass.h"

#include "paddle/fluid/pir/dialect/operator/interface/op_yaml_info.h"
#include "paddle/fluid/pir/dialect/operator/ir/onednn_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_parser.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_util.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"
#include "paddle/pir/include/pattern_rewrite/pattern_match.h"

#include "paddle/fluid/pir/dialect/kernel/ir/kernel_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/pattern_rewrite/pattern_match.h"

namespace {
class CpuBfloat16PlacementPattern : public pir::RewritePattern {
 public:
  explicit CpuBfloat16PlacementPattern(pir::IrContext* context)
      : pir::RewritePattern(MatchAnyOpTypeTag(),
                            1 /*benefit*/,
                            context,
                            {} /*generated_names*/) {}
  bool Match(pir::Operation* op) const override {
    auto attributes = op->attributes();
    if (attributes.find("mkldnn_data_type") != attributes.end()) {
      auto mkldnn_data_type = attributes.at("mkldnn_data_type")
                                  .dyn_cast<pir::StrAttribute>()
                                  .AsString();
      if (mkldnn_data_type == "int8") {
        return false;
      }
    }

    if (!op->isa<paddle::onednn::dialect::CastOp>() &&
        !op->isa<paddle::onednn::dialect::Cast_Op>() &&
        !op->isa<paddle::onednn::dialect::ClipOp>() &&
        !op->isa<paddle::onednn::dialect::Clip_Op>() &&
        !op->isa<paddle::onednn::dialect::AddOp>() &&
        !op->isa<paddle::onednn::dialect::Add_Op>() &&
        !op->isa<paddle::onednn::dialect::MultiplyOp>() &&
        !op->isa<paddle::onednn::dialect::Multiply_Op>() &&
        !op->isa<paddle::onednn::dialect::ReluOp>() &&
        !op->isa<paddle::onednn::dialect::Relu_Op>() &&
        !op->isa<paddle::onednn::dialect::Reshape_Op>() &&
        !op->isa<paddle::onednn::dialect::ReshapeOp>() &&
        !op->isa<paddle::onednn::dialect::ScaleOp>() &&
        !op->isa<paddle::onednn::dialect::Scale_Op>() &&
        !op->isa<paddle::onednn::dialect::SigmoidOp>() &&
        !op->isa<paddle::onednn::dialect::Sigmoid_Op>() &&
        !op->isa<paddle::onednn::dialect::SoftmaxOp>() &&
        !op->isa<paddle::onednn::dialect::Softmax_Op>() &&
        !op->isa<paddle::onednn::dialect::SqueezeOp>() &&
        !op->isa<paddle::onednn::dialect::Squeeze_Op>() &&
        !op->isa<paddle::onednn::dialect::TransposeOp>() &&
        !op->isa<paddle::onednn::dialect::Transpose_Op>()) {
      return false;
    }

    // The prev op should be quant op.
    bool pre_op = false;
    for (size_t i = 0; i < op->num_operands(); ++i) {
      paddle::onednn::dialect::QuantizeOp quant_op =
          pir::GetDefiningOpForInput(op, i)
              ->dyn_cast<paddle::onednn::dialect::QuantizeOp>();
      if (quant_op) {
        pre_op = true;
        break;
      }
    }
    if (!pre_op) {
      return false;
    }

    return true;
  }
  void Rewrite(pir::Operation* op,
               pir::PatternRewriter& rewriter) const override {  // NOLINT
    auto attributes = op->attributes();

    std::string target_op_name = op->name();

    auto op_info =
        pir::IrContext::Instance()->GetRegisteredOpInfo(target_op_name);
    if (op_info) {
      std::vector<pir::Type> op_item_inner_output_types;
      for (size_t i = 0; i < op->num_results(); ++i) {
        op_item_inner_output_types.push_back(op->result_type(i));
      }

      for (auto& attr : attributes) {
        if (attr.first == "mkldnn_data_type") {
          attributes[attr.first] =
              pir::StrAttribute::get(pir::IrContext::Instance(), "bfloat16");
        }
      }

      pir::Operation* op_item_inner = rewriter.Build(op->operands_source(),
                                                     attributes,
                                                     op_item_inner_output_types,
                                                     op_info);
      rewriter.ReplaceOp(op, op_item_inner->results());
    }
  }
};

class OneDNNBf16SpecialPlacementPass : public pir::PatternRewritePass {
 public:
  OneDNNBf16SpecialPlacementPass()
      : pir::PatternRewritePass("cpu_bfloat16_special_placement_pass", 3) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);
    ps.Add<CpuBfloat16PlacementPattern>(context);

    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateCpuBf16SpecialPlacementPatternPass() {
  return std::make_unique<OneDNNBf16SpecialPlacementPass>();
}
}  // namespace pir

REGISTER_IR_PASS(cpu_bfloat16_special_placement_pass,
                 OneDNNBf16SpecialPlacementPass);
