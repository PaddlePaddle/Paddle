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

#include "paddle/fluid/pir/transforms/onednn/onednn_placement_pass.h"

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
template <class IrType1, class IrType2>
static pir::Type create_type(pir::Type type,
                             const phi::Place &place,
                             pir::Type out_dtype,
                             pir::IrContext *ctx) {
  auto input_type = type.dyn_cast<IrType1>();
  return IrType2::get(ctx,
                      place,
                      out_dtype,
                      input_type.dims(),
                      input_type.data_layout(),
                      input_type.lod(),
                      input_type.offset());
}

template <typename OpType>
class OneDNNBf16PlacementPattern : public pir::OpRewritePattern<OpType> {
 public:
  using pir::OpRewritePattern<OpType>::OpRewritePattern;

  bool MatchAndRewrite(
      OpType op,
      pir::PatternRewriter &rewriter) const override {  // NOLINT

    // for (auto& value : op->operands_source()) {
    //   pir::Type op_dtype = pir::GetDataTypeFromValue(value);
    //    // Only float input can be converted to bfloat16
    //   if (!op_dtype.isa<pir::Float32Type>()) {
    //       return false;
    //   }
    // }
    // The pass use HasOpINT8DataType to skip int8 op
    auto op_attr = op->attributes();

    if (op_attr.find("mkldnn_data_type") != op_attr.end() &&
        op_attr.find("mkldnn_data_type")->second == "int8") {
      return false;
    }
    if (op_attr.find("use_quantizer") != op_attr.end() &&
        op_attr.find("use_quantizer")->second) {
      return false;
    }

    std::string target_op_name = op->name();
    target_op_name.replace(0, 5, "onednn_op");

    auto op_info =
        pir::IrContext::Instance()->GetRegisteredOpInfo(target_op_name);
    // pir::IrContext* ctx = pir::IrContext::Instance();
    if (op_info) {
      std::vector<pir::Type> op_item_inner_output_types;
      VLOG(0) << "This is op->num_results:" << op->num_results();
      for (size_t i = 0; i < op->num_results(); ++i) {
        // VLOG(0) << "This is result_type:" << op->result_type(i);
        // pir::Type type = op->result_type(i);
        // auto out_dtype = type.dyn_cast<pir::DenseTensorType>().dtype();
        // VLOG(0) << "This is dtype:" << out_dtype;
        // pir::Type new_type = create_type<pir::DenseTensorType,
        // paddle::dialect::AllocatedDenseTensorType>( type, phi::CPUPlace(),
        // pir::BFloat16Type::get(ctx), ctx);
        // op_item_inner_output_types.push_back(new_type);
        op_item_inner_output_types.push_back(op->result_type(i));
      }
      auto attributes = op->attributes();
      auto yaml_interface =
          op_info.GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>();
      paddle::dialect::OpRunTimeInfo runtime_info =
          std::get<3>(yaml_interface->get_op_info_(target_op_name));
      for (auto &attr : runtime_info.extra_args_default_value) {
        if (attr.first == "mkldnn_data_type") {
          VLOG(0) << "mkldnn_data_type set to bf16" << target_op_name;
          attributes[attr.first] =
              pir::StrAttribute::get(pir::IrContext::Instance(), "bfloat16");
        } else {
          attributes[attr.first] = attr.second;
        }
      }

      pir::Operation *op_item_inner = rewriter.Build(op->operands_source(),
                                                     attributes,
                                                     op_item_inner_output_types,
                                                     op_info);
      rewriter.ReplaceOp(op, op_item_inner->results());
    }

    return true;
  }
};

template <typename OpType>
class RemoveOrphanedPattern : public pir::OpRewritePattern<OpType> {
 public:
  using pir::OpRewritePattern<OpType>::OpRewritePattern;

  // find orphaned bfloat16 operator that is between two float32 operators
  // revert mkldnn_data_type attr to float32
  bool MatchAndRewrite(
      OpType op,
      pir::PatternRewriter &rewriter) const override {  // NOLINT

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
        auto *prev_op = pir::GetDefiningOpForInput(op, i);
        // Some ops do not need to be processed
        std::string prev_name = prev_op->name();
        if (constant_op.count(prev_name)) {
          continue;
        }

        // if (prev_op->HasTrait<pir::SideEffectTrait>() ||
        // prev_op->isa<pir::ConstantTensorOp>() ||
        // prev_op->isa<pir::ParameterOp>() ||
        // prev_op->isa<paddle::dialect::FeedOp>() ||
        // prev_op->isa<paddle::dialect::DataOp>()) {
        //       continue;
        //     }

        auto op_attr = prev_op->attributes();
        if (op_attr.find("mkldnn_data_type") == op_attr.end()) {
          // data_type_is_missing
          prev_fp32 = true;
          break;
        }
        if (op_attr.find("mkldnn_data_type")->second == "float32") {
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
      for (auto const &[next_op, op_index] : next_op_list) {
        // Some ops do not need to be processed
        std::string next_op_name = next_op->name();
        if (constant_op.count(next_op_name)) {
          continue;
        }
        auto op_next_attr = next_op->attributes();
        if (op_next_attr.find("mkldnn_data_type") == op_next_attr.end()) {
          // data_type_is_missing
          VLOG(0) << "data_type_is_missing:" << next_op->name();
          next_fp32 = true;
          break;
        }
        if (op_next_attr.find("mkldnn_data_type")->second == "float32") {
          VLOG(0) << "mkldnn_data_type is fp32:" << next_op->name();
          next_fp32 = true;
          break;
        }
      }
    }

    if (prev_fp32 && next_fp32) {
      VLOG(0) << "RemoveOrphanedOperators:" << op->name();
    } else {
      VLOG(0) << "Not RemoveOrphanedOperators:" << op->name();
      return false;
    }

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

      pir::Operation *op_item_inner = rewriter.Build(op->operands_source(),
                                                     attributes,
                                                     op_item_inner_output_types,
                                                     op_info);
      rewriter.ReplaceOp(op, op_item_inner->results());
    }

    return true;
  }
};

template <typename OpType>
class RemoveUnsupportedOpPattern : public pir::OpRewritePattern<OpType> {
 public:
  using pir::OpRewritePattern<OpType>::OpRewritePattern;

  bool MatchAndRewrite(
      OpType op,
      pir::PatternRewriter &rewriter) const override {  // NOLINT

    uint32_t num_operands = op->num_operands();
    for (uint32_t i = 0; i < num_operands; i++) {
      auto *pre_op = pir::GetDefiningOpForInput(op, i);
      if (pre_op->HasAttribute("mkldnn_data_type")) {
        return false;
      }
    }

    bool unsupported_op = false;
    for (auto &value : op->operands_source()) {
      pir::Type op_dtype = pir::GetDataTypeFromValue(value);
      // Only float input can be converted to bfloat16
      if (!op_dtype.isa<pir::Float32Type>()) {
        unsupported_op = true;
      }
    }
    if (!unsupported_op) {
      return false;
    }

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
      pir::Operation *op_item_inner = rewriter.Build(op->operands_source(),
                                                     attributes,
                                                     op_item_inner_output_types,
                                                     op_info);
      rewriter.ReplaceOp(op, op_item_inner->results());
    }

    return true;
  }
};

template <typename OpType>
class CpuBfloat16Pattern1 : public pir::OpRewritePattern<OpType> {
 public:
  using pir::OpRewritePattern<OpType>::OpRewritePattern;
  bool MatchAndRewrite(
      OpType op,
      pir::PatternRewriter &rewriter) const override {  // NOLINT
    auto op_attr = op->attributes();

    if (op_attr.find("mkldnn_data_type") != op_attr.end() &&
        op_attr.find("mkldnn_data_type")->second != "bfloat16") {
      return false;
    }
    auto op_info = pir::IrContext::Instance()->GetRegisteredOpInfo(op->name());
    pir::IrContext *ctx = pir::IrContext::Instance();
    VLOG(0) << "MatchAndRewrite===";
    if (op_info) {
      std::vector<pir::Type> op_item_inner_output_types;
      for (size_t i = 0; i < op->num_results(); ++i) {
        pir::Type type = op->result_type(i);
        pir::Type new_type =
            create_type<pir::DenseTensorType,
                        paddle::dialect::AllocatedDenseTensorType>(
                type, phi::CPUPlace(), pir::BFloat16Type::get(ctx), ctx);
        VLOG(0) << "This is new_type:" << new_type;
        op_item_inner_output_types.push_back(new_type);
        // op_item_inner_output_types.push_back(op->result_type(i));
      }
      auto attributes = op->attributes();

      pir::Operation *op_item_inner = rewriter.Build(op->operands_source(),
                                                     attributes,
                                                     op_item_inner_output_types,
                                                     op_info);
      rewriter.ReplaceOp(op, op_item_inner->results());
    }

    return true;
  }
};

class PatternCreator {
 public:
  explicit PatternCreator(pir::IrContext *context) : context(context) {}

  template <typename Op>
  void CreateBf16PlacementPatterns(pir::RewritePatternSet &patternSet) {
    auto pattern = std::make_unique<OneDNNBf16PlacementPattern<Op>>(
        context, benefit++, std::vector<std::string>{});
    patternSet.Add(std::move(pattern));
  }

  template <typename Op>
  void CreateRemoveOrphanedPatterns(pir::RewritePatternSet &patternSet) {
    auto pattern = std::make_unique<RemoveOrphanedPattern<Op>>(
        context, benefit++, std::vector<std::string>{});
    patternSet.Add(std::move(pattern));
  }

  template <typename Op>
  void CreateRUnsupportedOpPatterns(pir::RewritePatternSet &patternSet) {
    auto pattern = std::make_unique<RemoveUnsupportedOpPattern<Op>>(
        context, benefit++, std::vector<std::string>{});
    patternSet.Add(std::move(pattern));
  }

  void ClearBenefit() { benefit = 1; }

 private:
  pir::IrContext *context;
  int benefit = 1;
};

class OneDNNPlacementBf16Pass : public pir::PatternRewritePass {
 public:
  OneDNNPlacementBf16Pass()
      : pir::PatternRewritePass("cpu_bfloat16_placement_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);

    PatternCreator patternCreator(context);
    patternCreator
        .CreateBf16PlacementPatterns<paddle::dialect::BilinearInterpOp>(ps);  //
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::CastOp>(ps);  //
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::Cast_Op>(
        ps);  //
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::ClipOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::Clip_Op>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::ConcatOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::Conv2dOp>(
        ps);  //
    patternCreator
        .CreateBf16PlacementPatterns<paddle::dialect::Conv2dTransposeOp>(
            ps);  //
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::AddOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::Add_Op>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::MultiplyOp>(
        ps);  //
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::Multiply_Op>(
        ps);  //
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::FcOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::FusionGruOp>(
        ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::GeluOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::LayerNormOp>(
        ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::MatmulOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::Pool2dOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::PreluOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::ReluOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::Relu_Op>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::Reshape_Op>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::ReshapeOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::ScaleOp>(
        ps);  //
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::Scale_Op>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::SigmoidOp>(
        ps);  //
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::Sigmoid_Op>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::SliceOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::SoftmaxOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::Softmax_Op>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::SplitOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::SqueezeOp>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::Squeeze_Op>(ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::SumOp>(ps);  //
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::TransposeOp>(
        ps);
    patternCreator.CreateBf16PlacementPatterns<paddle::dialect::Transpose_Op>(
        ps);

    patternCreator.ClearBenefit();

    patternCreator.CreateRemoveOrphanedPatterns<
        paddle::onednn::dialect::BilinearInterpOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::CastOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::Cast_Op>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::ClipOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::Clip_Op>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::ConcatOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::Conv2dOp>(ps);
    patternCreator.CreateRemoveOrphanedPatterns<
        paddle::onednn::dialect::Conv2dTransposeOp>(ps);
    patternCreator.CreateRemoveOrphanedPatterns<paddle::onednn::dialect::AddOp>(
        ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::Add_Op>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::MultiplyOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::Multiply_Op>(ps);
    patternCreator.CreateRemoveOrphanedPatterns<paddle::onednn::dialect::FcOp>(
        ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::FusionGruOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::GeluOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::LayerNormOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::MatmulOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::Pool2dOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::PreluOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::ReluOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::Relu_Op>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::Reshape_Op>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::ReshapeOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::ScaleOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::Scale_Op>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::SigmoidOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::Sigmoid_Op>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::SliceOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::SoftmaxOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::Softmax_Op>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::SplitOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::SqueezeOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::Squeeze_Op>(ps);
    patternCreator.CreateRemoveOrphanedPatterns<paddle::onednn::dialect::SumOp>(
        ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::TransposeOp>(ps);
    patternCreator
        .CreateRemoveOrphanedPatterns<paddle::onednn::dialect::Transpose_Op>(
            ps);

    patternCreator.ClearBenefit();
    patternCreator.CreateRUnsupportedOpPatterns<
        paddle::onednn::dialect::BilinearInterpOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::CastOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::Cast_Op>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::ClipOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::Clip_Op>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::ConcatOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::Conv2dOp>(ps);
    patternCreator.CreateRUnsupportedOpPatterns<
        paddle::onednn::dialect::Conv2dTransposeOp>(ps);
    patternCreator.CreateRUnsupportedOpPatterns<paddle::onednn::dialect::AddOp>(
        ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::Add_Op>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::MultiplyOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::Multiply_Op>(ps);
    patternCreator.CreateRUnsupportedOpPatterns<paddle::onednn::dialect::FcOp>(
        ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::FusionGruOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::GeluOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::LayerNormOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::MatmulOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::Pool2dOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::PreluOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::ReluOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::Relu_Op>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::Reshape_Op>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::ReshapeOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::ScaleOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::Scale_Op>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::SigmoidOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::Sigmoid_Op>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::SliceOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::SoftmaxOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::Softmax_Op>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::SplitOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::SqueezeOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::Squeeze_Op>(ps);
    patternCreator.CreateRUnsupportedOpPatterns<paddle::onednn::dialect::SumOp>(
        ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::TransposeOp>(ps);
    patternCreator
        .CreateRUnsupportedOpPatterns<paddle::onednn::dialect::Transpose_Op>(
            ps);

    // ps.Add(paddle::drr::Create<CpuBfloat16Pattern1>(
    //       context, paddle::onednn::dialect::AddOp::name()));

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
