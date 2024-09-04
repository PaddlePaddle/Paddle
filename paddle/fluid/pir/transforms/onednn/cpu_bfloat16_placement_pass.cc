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

#include "paddle/fluid/pir/transforms/onednn/cpu_bfloat16_placement_pass.h"

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

namespace {
class OneDNNBf16PlacementPattern : public pir::RewritePattern {
 public:
  explicit OneDNNBf16PlacementPattern(pir::IrContext* context)
      : pir::RewritePattern(MatchAnyOpTypeTag(),
                            1 /*benefit*/,
                            context,
                            {} /*generated_names*/) {}

  bool Match(pir::Operation* op) const override {  // NOLINT
    if (!op->isa<paddle::onednn::dialect::BilinearInterpOp>() &&
        !op->isa<paddle::onednn::dialect::CastOp>() &&
        !op->isa<paddle::onednn::dialect::Cast_Op>() &&
        !op->isa<paddle::onednn::dialect::ClipOp>() &&
        !op->isa<paddle::onednn::dialect::Clip_Op>() &&
        !op->isa<paddle::onednn::dialect::Conv2dOp>() &&
        !op->isa<paddle::onednn::dialect::Conv2dTransposeOp>() &&
        !op->isa<paddle::onednn::dialect::Conv2dTransposeBiasOp>() &&
        !op->isa<paddle::onednn::dialect::AddOp>() &&
        !op->isa<paddle::onednn::dialect::Add_Op>() &&
        !op->isa<paddle::onednn::dialect::MultiplyOp>() &&
        !op->isa<paddle::onednn::dialect::Multiply_Op>() &&
        !op->isa<paddle::onednn::dialect::FcOp>() &&
        !op->isa<paddle::onednn::dialect::FusionGruOp>() &&
        !op->isa<paddle::onednn::dialect::GeluOp>() &&
        !op->isa<paddle::onednn::dialect::LayerNormOp>() &&
        !op->isa<paddle::onednn::dialect::MatmulOp>() &&
        !op->isa<paddle::onednn::dialect::Pool2dOp>() &&
        !op->isa<paddle::onednn::dialect::PreluOp>() &&
        !op->isa<paddle::onednn::dialect::ReluOp>() &&
        !op->isa<paddle::onednn::dialect::Relu_Op>() &&
        !op->isa<paddle::onednn::dialect::Reshape_Op>() &&
        !op->isa<paddle::onednn::dialect::ReshapeOp>() &&
        !op->isa<paddle::onednn::dialect::ScaleOp>() &&
        !op->isa<paddle::onednn::dialect::Scale_Op>() &&
        !op->isa<paddle::onednn::dialect::SigmoidOp>() &&
        !op->isa<paddle::onednn::dialect::Sigmoid_Op>() &&
        !op->isa<paddle::onednn::dialect::SliceOp>() &&
        !op->isa<paddle::onednn::dialect::SoftmaxOp>() &&
        !op->isa<paddle::onednn::dialect::Softmax_Op>() &&
        !op->isa<paddle::onednn::dialect::SqueezeOp>() &&
        !op->isa<paddle::onednn::dialect::Squeeze_Op>() &&
        !op->isa<paddle::onednn::dialect::SumOp>() &&
        !op->isa<paddle::onednn::dialect::TransposeOp>() &&
        !op->isa<paddle::onednn::dialect::Transpose_Op>() &&
        !op->isa<paddle::onednn::dialect::FusedConv2dOp>() &&
        !op->isa<paddle::onednn::dialect::FusedMatmulOp>()) {
      return false;
    }

    // The pass use HasOpINT8DataType to skip int8 op
    auto op_attr = op->attributes();
    if (op_attr.find("mkldnn_data_type") != op_attr.end()) {
      auto mkldnn_data_type = op_attr.at("mkldnn_data_type")
                                  .dyn_cast<pir::StrAttribute>()
                                  .AsString();
      if (mkldnn_data_type == "int8") {
        return false;
      }
    }

    if (op_attr.find("use_quantizer") != op_attr.end()) {
      if (op_attr.at("use_quantizer").dyn_cast<pir::BoolAttribute>().data()) {
        return false;
      }
    }
    return true;
  }

  void Rewrite(pir::Operation* op,
               pir::PatternRewriter& rewriter) const override {  // NOLINT
    std::string target_op_name = op->name();

    auto op_info =
        pir::IrContext::Instance()->GetRegisteredOpInfo(target_op_name);
    if (op_info) {
      std::vector<pir::Type> op_item_inner_output_types;
      for (size_t i = 0; i < op->num_results(); ++i) {
        op_item_inner_output_types.push_back(op->result_type(i));
      }
      auto attributes = op->attributes();
      for (auto& attr : attributes) {
        if (attr.first == "mkldnn_data_type") {
          VLOG(8) << "mkldnn_data_type set to bf16, op:" << target_op_name;
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

class RemoveOrphanedPattern : public pir::RewritePattern {
 public:
  explicit RemoveOrphanedPattern(pir::IrContext* context)
      : pir::RewritePattern(MatchAnyOpTypeTag(),
                            1 /*benefit*/,
                            context,
                            {} /*generated_names*/) {}

  // find orphaned bfloat16 operator that is between two float32 operators
  // revert mkldnn_data_type attr to float32
  bool Match(pir::Operation* op) const override {  // NOLINT
    if (!op->isa<paddle::onednn::dialect::BilinearInterpOp>() &&
        !op->isa<paddle::onednn::dialect::CastOp>() &&
        !op->isa<paddle::onednn::dialect::Cast_Op>() &&
        !op->isa<paddle::onednn::dialect::ClipOp>() &&
        !op->isa<paddle::onednn::dialect::Clip_Op>() &&
        !op->isa<paddle::onednn::dialect::Conv2dOp>() &&
        !op->isa<paddle::onednn::dialect::Conv2dTransposeOp>() &&
        !op->isa<paddle::onednn::dialect::Conv2dTransposeBiasOp>() &&
        !op->isa<paddle::onednn::dialect::AddOp>() &&
        !op->isa<paddle::onednn::dialect::Add_Op>() &&
        !op->isa<paddle::onednn::dialect::MultiplyOp>() &&
        !op->isa<paddle::onednn::dialect::Multiply_Op>() &&
        !op->isa<paddle::onednn::dialect::FcOp>() &&
        !op->isa<paddle::onednn::dialect::FusionGruOp>() &&
        !op->isa<paddle::onednn::dialect::GeluOp>() &&
        !op->isa<paddle::onednn::dialect::LayerNormOp>() &&
        !op->isa<paddle::onednn::dialect::MatmulOp>() &&
        !op->isa<paddle::onednn::dialect::Pool2dOp>() &&
        !op->isa<paddle::onednn::dialect::PreluOp>() &&
        !op->isa<paddle::onednn::dialect::ReluOp>() &&
        !op->isa<paddle::onednn::dialect::Relu_Op>() &&
        !op->isa<paddle::onednn::dialect::Reshape_Op>() &&
        !op->isa<paddle::onednn::dialect::ReshapeOp>() &&
        !op->isa<paddle::onednn::dialect::ScaleOp>() &&
        !op->isa<paddle::onednn::dialect::Scale_Op>() &&
        !op->isa<paddle::onednn::dialect::SigmoidOp>() &&
        !op->isa<paddle::onednn::dialect::Sigmoid_Op>() &&
        !op->isa<paddle::onednn::dialect::SliceOp>() &&
        !op->isa<paddle::onednn::dialect::SoftmaxOp>() &&
        !op->isa<paddle::onednn::dialect::Softmax_Op>() &&
        !op->isa<paddle::onednn::dialect::SqueezeOp>() &&
        !op->isa<paddle::onednn::dialect::Squeeze_Op>() &&
        !op->isa<paddle::onednn::dialect::SumOp>() &&
        !op->isa<paddle::onednn::dialect::TransposeOp>() &&
        !op->isa<paddle::onednn::dialect::Transpose_Op>() &&
        !op->isa<paddle::onednn::dialect::FusedConv2dOp>() &&
        !op->isa<paddle::onednn::dialect::FusedMatmulOp>()) {
      return false;
    }

    bool prev_fp32 = false;
    bool next_fp32 = false;
    const std::set<std::string> constant_op({"pd_op.data",
                                             "builtin.parameter",
                                             "pd_op.feed",
                                             "pd_op.fetch",
                                             "pd_op.assign"});

    if (op->num_operands()) {
      for (uint32_t i = 0; i < op->num_operands(); i++) {
        if (!op->operand_source(i) || !op->operand_source(i).type()) {
          continue;
        }
        auto* prev_op = pir::GetDefiningOpForInput(op, i);
        // if (!prev_op) continue;
        // Some ops do not need to be processed
        std::string prev_name = prev_op->name();
        if (constant_op.count(prev_name)) {
          continue;
        }

        auto op_attr = prev_op->attributes();
        if (op_attr.find("mkldnn_data_type") == op_attr.end()) {
          // data_type_is_missing
          prev_fp32 = true;
          break;
        }
        auto mkldnn_data_type = op_attr.at("mkldnn_data_type")
                                    .dyn_cast<pir::StrAttribute>()
                                    .AsString();

        if (mkldnn_data_type == "float32") {
          prev_fp32 = true;
          break;
        }
      }
    } else {
      // The first op in graph
      return false;
    }

    for (uint32_t i = 0; i < op->num_results(); i++) {
      if (!op->result(i) || !op->result(i).type()) {
        continue;
      }
      auto next_op_list = pir::GetUseOpsForOutput(op, i);
      for (auto const& [next_op, op_index] : next_op_list) {
        // Some ops do not need to be processed
        std::string next_op_name = next_op->name();
        if (constant_op.count(next_op_name)) {
          continue;
        }
        auto op_next_attr = next_op->attributes();
        if (op_next_attr.find("mkldnn_data_type") == op_next_attr.end()) {
          // data_type_is_missing
          VLOG(8) << "data_type_is_missing:" << next_op->name();
          next_fp32 = true;
          break;
        }
        auto mkldnn_data_type = op_next_attr.at("mkldnn_data_type")
                                    .dyn_cast<pir::StrAttribute>()
                                    .AsString();
        if (mkldnn_data_type == "float32") {
          VLOG(8) << "mkldnn_data_type is fp32:" << next_op->name();
          next_fp32 = true;
          break;
        }
      }
    }

    return prev_fp32 && next_fp32;
  }

  void Rewrite(pir::Operation* op,
               pir::PatternRewriter& rewriter) const override {  // NOLINT
    std::string target_op_name = op->name();
    auto op_info =
        pir::IrContext::Instance()->GetRegisteredOpInfo(target_op_name);
    if (op_info) {
      std::vector<pir::Type> op_item_inner_output_types;
      for (size_t i = 0; i < op->num_results(); ++i) {
        op_item_inner_output_types.push_back(op->result_type(i));
      }
      auto attributes = op->attributes();

      if (attributes.find("mkldnn_data_type") != attributes.end()) {
        attributes["mkldnn_data_type"] =
            pir::StrAttribute::get(pir::IrContext::Instance(), "float32");
      }

      pir::Operation* op_item_inner = rewriter.Build(op->operands_source(),
                                                     attributes,
                                                     op_item_inner_output_types,
                                                     op_info);
      rewriter.ReplaceOp(op, op_item_inner->results());
    }
  }
};

class RemoveUnsupportedOpPattern : public pir::RewritePattern {
 public:
  explicit RemoveUnsupportedOpPattern(pir::IrContext* context)
      : pir::RewritePattern(MatchAnyOpTypeTag(),
                            1 /*benefit*/,
                            context,
                            {} /*generated_names*/) {}

  bool Match(pir::Operation* op) const override {  // NOLINT
    if (!op->isa<paddle::onednn::dialect::BilinearInterpOp>() &&
        !op->isa<paddle::onednn::dialect::CastOp>() &&
        !op->isa<paddle::onednn::dialect::Cast_Op>() &&
        !op->isa<paddle::onednn::dialect::ClipOp>() &&
        !op->isa<paddle::onednn::dialect::Clip_Op>() &&
        !op->isa<paddle::onednn::dialect::Conv2dOp>() &&
        !op->isa<paddle::onednn::dialect::Conv2dTransposeOp>() &&
        !op->isa<paddle::onednn::dialect::Conv2dTransposeBiasOp>() &&
        !op->isa<paddle::onednn::dialect::AddOp>() &&
        !op->isa<paddle::onednn::dialect::Add_Op>() &&
        !op->isa<paddle::onednn::dialect::MultiplyOp>() &&
        !op->isa<paddle::onednn::dialect::Multiply_Op>() &&
        !op->isa<paddle::onednn::dialect::FcOp>() &&
        !op->isa<paddle::onednn::dialect::FusionGruOp>() &&
        !op->isa<paddle::onednn::dialect::GeluOp>() &&
        !op->isa<paddle::onednn::dialect::LayerNormOp>() &&
        !op->isa<paddle::onednn::dialect::MatmulOp>() &&
        !op->isa<paddle::onednn::dialect::Pool2dOp>() &&
        !op->isa<paddle::onednn::dialect::PreluOp>() &&
        !op->isa<paddle::onednn::dialect::ReluOp>() &&
        !op->isa<paddle::onednn::dialect::Relu_Op>() &&
        !op->isa<paddle::onednn::dialect::Reshape_Op>() &&
        !op->isa<paddle::onednn::dialect::ReshapeOp>() &&
        !op->isa<paddle::onednn::dialect::ScaleOp>() &&
        !op->isa<paddle::onednn::dialect::Scale_Op>() &&
        !op->isa<paddle::onednn::dialect::SigmoidOp>() &&
        !op->isa<paddle::onednn::dialect::Sigmoid_Op>() &&
        !op->isa<paddle::onednn::dialect::SliceOp>() &&
        !op->isa<paddle::onednn::dialect::SoftmaxOp>() &&
        !op->isa<paddle::onednn::dialect::Softmax_Op>() &&
        !op->isa<paddle::onednn::dialect::SqueezeOp>() &&
        !op->isa<paddle::onednn::dialect::Squeeze_Op>() &&
        !op->isa<paddle::onednn::dialect::SumOp>() &&
        !op->isa<paddle::onednn::dialect::TransposeOp>() &&
        !op->isa<paddle::onednn::dialect::Transpose_Op>() &&
        !op->isa<paddle::onednn::dialect::FusedConv2dOp>() &&
        !op->isa<paddle::onednn::dialect::FusedMatmulOp>()) {
      return false;
    }

    uint32_t num_operands = op->num_operands();
    for (uint32_t i = 0; i < num_operands; i++) {
      auto* pre_op = pir::GetDefiningOpForInput(op, i);
      if (pre_op->HasAttribute("mkldnn_data_type")) {
        return false;
      }
    }

    bool unsupported_op = false;
    for (auto& value : op->operands_source()) {
      pir::Type op_dtype = pir::GetDataTypeFromValue(value);
      // Only float input can be converted to bfloat16
      if (!op_dtype.isa<pir::Float32Type>()) {
        unsupported_op = true;
        break;
      }
    }
    if (!unsupported_op) {
      return false;
    }
    return true;
  }

  void Rewrite(pir::Operation* op,
               pir::PatternRewriter& rewriter) const override {  // NOLINT
    std::string target_op_name = op->name();
    auto op_info =
        pir::IrContext::Instance()->GetRegisteredOpInfo(target_op_name);
    if (op_info) {
      std::vector<pir::Type> op_item_inner_output_types;
      for (size_t i = 0; i < op->num_results(); ++i) {
        op_item_inner_output_types.push_back(op->result_type(i));
      }
      auto attributes = op->attributes();
      if (attributes.find("mkldnn_data_type") != attributes.end()) {
        attributes["mkldnn_data_type"] =
            pir::StrAttribute::get(pir::IrContext::Instance(), "float32");
      }
      pir::Operation* op_item_inner = rewriter.Build(op->operands_source(),
                                                     attributes,
                                                     op_item_inner_output_types,
                                                     op_info);
      rewriter.ReplaceOp(op, op_item_inner->results());
    }
  }
};

class OneDNNPlacementBf16Pass : public pir::PatternRewritePass {
 public:
  OneDNNPlacementBf16Pass()
      : pir::PatternRewritePass("cpu_bfloat16_placement_pass", 3) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);
    ps.Add<OneDNNBf16PlacementPattern>(context);
    ps.Add<RemoveOrphanedPattern>(context);
    ps.Add<RemoveUnsupportedOpPattern>(context);

    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateCpuBf16PlacementPass() {
  return std::make_unique<OneDNNPlacementBf16Pass>();
}

}  // namespace pir

REGISTER_IR_PASS(cpu_bfloat16_placement_pass, OneDNNPlacementBf16Pass);
