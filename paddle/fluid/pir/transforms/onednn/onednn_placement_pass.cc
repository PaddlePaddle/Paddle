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

namespace {

template <typename OpType>
class OneDNNPlacementPattern : public pir::OpRewritePattern<OpType> {
 public:
  using pir::OpRewritePattern<OpType>::OpRewritePattern;

  bool MatchAndRewrite(
      OpType op,
      pir::PatternRewriter &rewriter) const override {  // NOLINT
    std::string target_op_name = op->name();
    if (target_op_name == "pd_op.scale" || target_op_name == "pd_op.scale_" ||
        target_op_name == "pd_op.cast" || target_op_name == "pd_op.cast_") {
      auto input_type = pir::GetDataTypeFromValue(op->operand_source(0));
      if (!(pir::isa<pir::Float32Type>(input_type) ||
            pir::isa<pir::BFloat16Type>(input_type)))
        return false;
    }
    if (target_op_name == "pd_op.slice") {
      auto input_type = pir::GetDataTypeFromValue(op->operand_source(0));
      if (!(pir::isa<pir::Float32Type>(input_type) ||
            pir::isa<pir::BFloat16Type>(input_type) ||
            pir::isa<pir::UInt8Type>(input_type) ||
            pir::isa<pir::Int8Type>(input_type)))
        return false;
    }
    target_op_name.replace(0, 5, "onednn_op");

    auto op_info =
        pir::IrContext::Instance()->GetRegisteredOpInfo(target_op_name);
    if (op_info) {
      std::vector<pir::Type> op_item_inner_output_types;
      for (size_t i = 0; i < op->num_results(); ++i) {
        op_item_inner_output_types.push_back(op->result_type(i));
      }
      auto attributes = op->attributes();
      auto yaml_interface =
          op_info.GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>();
      paddle::dialect::OpRunTimeInfo runtime_info =
          std::get<3>(yaml_interface->get_op_info_(target_op_name));
      for (auto &attr : runtime_info.extra_args_default_value) {
        attributes[attr.first] = attr.second;
      }
      if (attributes.find("is_test") != attributes.end()) {
        attributes["is_test"] = rewriter.bool_attr(true);
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

class PatternCreator {
 public:
  explicit PatternCreator(pir::IrContext *context) : context(context) {}

  template <typename Op>
  void CreatePatterns(pir::RewritePatternSet &patternSet) {
    auto pattern = std::make_unique<OneDNNPlacementPattern<Op>>(
        context, benefit++, std::vector<std::string>{});
    patternSet.Add(std::move(pattern));
  }

 private:
  pir::IrContext *context;
  int benefit = 1;
};

class OneDNNPlacementPass : public pir::PatternRewritePass {
 public:
  OneDNNPlacementPass() : pir::PatternRewritePass("onednn_placement_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);

    PatternCreator patternCreator(context);
    patternCreator.CreatePatterns<paddle::dialect::MatmulOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::AbsOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::Abs_Op>(ps);
    patternCreator.CreatePatterns<paddle::dialect::BilinearInterpOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::ClipOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::Clip_Op>(ps);
    patternCreator.CreatePatterns<paddle::dialect::ConcatOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::Conv2dOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::Conv3dOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::DepthwiseConv2dOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::EluOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::Elu_Op>(ps);
    patternCreator.CreatePatterns<paddle::dialect::ExpOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::Exp_Op>(ps);
    patternCreator.CreatePatterns<paddle::dialect::FlattenOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::Flatten_Op>(ps);
    patternCreator.CreatePatterns<paddle::dialect::GeluOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::LayerNormOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::LeakyReluOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::LeakyRelu_Op>(ps);
    patternCreator.CreatePatterns<paddle::dialect::LogSoftmaxOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::NearestInterpOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::Pad3dOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::PreluOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::PriorBoxOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::ReluOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::Relu_Op>(ps);
    patternCreator.CreatePatterns<paddle::dialect::Relu6Op>(ps);
    patternCreator.CreatePatterns<paddle::dialect::RoundOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::Round_Op>(ps);
    patternCreator.CreatePatterns<paddle::dialect::ScaleOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::ScaleSrOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::Scale_Op>(ps);
    patternCreator.CreatePatterns<paddle::dialect::ScaleSr_Op>(ps);
    patternCreator.CreatePatterns<paddle::dialect::Sgd_Op>(ps);
    patternCreator.CreatePatterns<paddle::dialect::SgdDenseParamSparseGrad_Op>(
        ps);
    patternCreator.CreatePatterns<paddle::dialect::SgdSparseParamSparseGrad_Op>(
        ps);
    patternCreator.CreatePatterns<paddle::dialect::ShapeOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::ShapeSrOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::SigmoidOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::Sigmoid_Op>(ps);
    patternCreator.CreatePatterns<paddle::dialect::SoftplusOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::SqrtOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::SqrtSrOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::Sqrt_Op>(ps);
    patternCreator.CreatePatterns<paddle::dialect::SqrtSr_Op>(ps);
    patternCreator.CreatePatterns<paddle::dialect::SqueezeOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::Squeeze_Op>(ps);
    patternCreator.CreatePatterns<paddle::dialect::StackOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::TanhOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::Tanh_Op>(ps);
    patternCreator.CreatePatterns<paddle::dialect::AbsGradOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::ClipGradOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::ClipGrad_Op>(ps);
    patternCreator.CreatePatterns<paddle::dialect::ConcatGradOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::Conv2dGradOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::Conv3dGradOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::DepthwiseConv2dGradOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::EluGradOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::EluGrad_Op>(ps);
    patternCreator.CreatePatterns<paddle::dialect::ExpGradOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::ExpGrad_Op>(ps);
    patternCreator.CreatePatterns<paddle::dialect::ExpandGradOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::FlattenGradOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::FlattenGrad_Op>(ps);
    patternCreator.CreatePatterns<paddle::dialect::LeakyReluGradOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::LeakyReluGrad_Op>(ps);
    patternCreator.CreatePatterns<paddle::dialect::PreluGradOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::Relu6GradOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::Relu6Grad_Op>(ps);
    patternCreator.CreatePatterns<paddle::dialect::ReluGrad_Op>(ps);
    patternCreator.CreatePatterns<paddle::dialect::SigmoidGradOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::SigmoidGrad_Op>(ps);
    patternCreator.CreatePatterns<paddle::dialect::SqrtGradOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::SqrtGrad_Op>(ps);
    patternCreator.CreatePatterns<paddle::dialect::SqueezeGradOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::SqueezeGrad_Op>(ps);
    patternCreator.CreatePatterns<paddle::dialect::TanhGradOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::TanhGrad_Op>(ps);
    patternCreator.CreatePatterns<paddle::dialect::FusionGruOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::AddNOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::Cast_Op>(ps);
    patternCreator.CreatePatterns<paddle::dialect::Conv2dTransposeOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::Conv2dTransposeBiasOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::DivideOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::Divide_Op>(ps);
    patternCreator.CreatePatterns<paddle::dialect::GaussianOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::HardswishOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::LrnOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::MatmulWithFlattenOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::MaxOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::MeanOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::MinOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::MishOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::MultiplyOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::MultiplySrOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::Multiply_Op>(ps);
    patternCreator.CreatePatterns<paddle::dialect::MultiplySr_Op>(ps);
    patternCreator.CreatePatterns<paddle::dialect::PadOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::Pool2dOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::Reshape_Op>(ps);
    patternCreator.CreatePatterns<paddle::dialect::SliceOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::Softmax_Op>(ps);
    patternCreator.CreatePatterns<paddle::dialect::SplitOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::SplitWithNumOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::SubtractOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::Subtract_Op>(ps);
    patternCreator.CreatePatterns<paddle::dialect::SumOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::SwishOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::Transpose_Op>(ps);
    patternCreator.CreatePatterns<paddle::dialect::AddDoubleGradOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::AddDoubleGrad_Op>(ps);
    patternCreator.CreatePatterns<paddle::dialect::AddGrad_Op>(ps);
    patternCreator.CreatePatterns<paddle::dialect::AddTripleGradOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::AddTripleGrad_Op>(ps);
    patternCreator.CreatePatterns<paddle::dialect::BatchNormGradOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::DivideGradOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::HardswishGradOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::HardswishGrad_Op>(ps);
    patternCreator.CreatePatterns<paddle::dialect::LrnGradOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::MatmulWithFlattenGradOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::MeanGradOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::MishGradOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::MishGrad_Op>(ps);
    patternCreator.CreatePatterns<paddle::dialect::MultiplyGradOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::Pool2dGradOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::ReshapeGradOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::ReshapeGrad_Op>(ps);
    patternCreator.CreatePatterns<paddle::dialect::SliceGradOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::SoftmaxGradOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::SubtractGradOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::SubtractGrad_Op>(ps);
    patternCreator.CreatePatterns<paddle::dialect::SumGradOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::SwishGradOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::SwishGrad_Op>(ps);
    patternCreator.CreatePatterns<paddle::dialect::TransposeGradOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::GeluGradOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::ReluGradOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::AddOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::Add_Op>(ps);
    patternCreator.CreatePatterns<paddle::dialect::BatchNormOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::BatchNorm_Op>(ps);
    patternCreator.CreatePatterns<paddle::dialect::CastOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::FullOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::MatmulOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::ReshapeOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::SoftmaxOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::TransposeOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::AddGradOp>(ps);
    patternCreator.CreatePatterns<paddle::dialect::MatmulGradOp>(ps);

    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateOneDNNPlacementPass() {
  return std::make_unique<OneDNNPlacementPass>();
}

}  // namespace pir

REGISTER_IR_PASS(onednn_placement_pass, OneDNNPlacementPass);
