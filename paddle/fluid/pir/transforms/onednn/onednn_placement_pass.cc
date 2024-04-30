// REGISTER_IR_PASS(onednn_placement_pass, OneDNNPlacementPass);
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

#include "paddle/fluid/pir/dialect/kernel/ir/kernel_op.h"
#include "paddle/fluid/pir/dialect/operator/interface/op_yaml_info.h"
#include "paddle/fluid/pir/dialect/operator/ir/onednn_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/dialect/operator/trait/onednn.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_parser.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_util.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/core/value.h"

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
    target_op_name.replace(0, 5, "onednn_op");

    auto op_info =
        pir::IrContext::Instance()->GetRegisteredOpInfo(target_op_name);
    if (op_info) {
      std::vector<pir::Type> op_item_inner_output_types;
      if (op->num_results() > 0) {
        for (size_t i = 0; i < op->num_results(); ++i) {
          op_item_inner_output_types.push_back(op->result_type(i));
        }
      }
      auto attributes = op->attributes();
      auto yaml_interface =
          op_info.GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>();
      paddle::dialect::OpRunTimeInfo runtime_info =
          std::get<3>(yaml_interface->get_op_info_(target_op_name));
      for (auto &attr : runtime_info.extra_args_default_value) {
        attributes[attr.first] = attr.second;
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
  void createPatterns(pir::RewritePatternSet &patternSet) {
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
  OneDNNPlacementPass() : pir::PatternRewritePass("onednn_placement_pass", 3) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);

    PatternCreator patternCreator(context);
    patternCreator.createPatterns<paddle::dialect::MatmulOp>(ps);
    patternCreator.createPatterns<paddle::dialect::AbsOp>(ps);
    patternCreator.createPatterns<paddle::dialect::Abs_Op>(ps);
    patternCreator.createPatterns<paddle::dialect::BilinearInterpOp>(ps);
    patternCreator.createPatterns<paddle::dialect::ClipOp>(ps);
    patternCreator.createPatterns<paddle::dialect::Clip_Op>(ps);
    patternCreator.createPatterns<paddle::dialect::ConcatOp>(ps);
    patternCreator.createPatterns<paddle::dialect::Conv2dOp>(ps);
    patternCreator.createPatterns<paddle::dialect::Conv3dOp>(ps);
    patternCreator.createPatterns<paddle::dialect::DepthwiseConv2dOp>(ps);
    patternCreator.createPatterns<paddle::dialect::EluOp>(ps);
    patternCreator.createPatterns<paddle::dialect::Elu_Op>(ps);
    patternCreator.createPatterns<paddle::dialect::ExpOp>(ps);
    patternCreator.createPatterns<paddle::dialect::Exp_Op>(ps);
    patternCreator.createPatterns<paddle::dialect::FlattenOp>(ps);
    patternCreator.createPatterns<paddle::dialect::Flatten_Op>(ps);
    patternCreator.createPatterns<paddle::dialect::GeluOp>(ps);
    patternCreator.createPatterns<paddle::dialect::LayerNormOp>(ps);
    patternCreator.createPatterns<paddle::dialect::LeakyReluOp>(ps);
    patternCreator.createPatterns<paddle::dialect::LeakyRelu_Op>(ps);
    patternCreator.createPatterns<paddle::dialect::LogSoftmaxOp>(ps);
    patternCreator.createPatterns<paddle::dialect::NearestInterpOp>(ps);
    patternCreator.createPatterns<paddle::dialect::Pad3dOp>(ps);
    patternCreator.createPatterns<paddle::dialect::PreluOp>(ps);
    patternCreator.createPatterns<paddle::dialect::PriorBoxOp>(ps);
    patternCreator.createPatterns<paddle::dialect::ReluOp>(ps);
    patternCreator.createPatterns<paddle::dialect::Relu_Op>(ps);
    patternCreator.createPatterns<paddle::dialect::Relu6Op>(ps);
    patternCreator.createPatterns<paddle::dialect::RoundOp>(ps);
    patternCreator.createPatterns<paddle::dialect::Round_Op>(ps);
    patternCreator.createPatterns<paddle::dialect::ScaleOp>(ps);
    patternCreator.createPatterns<paddle::dialect::ScaleSrOp>(ps);
    patternCreator.createPatterns<paddle::dialect::Scale_Op>(ps);
    patternCreator.createPatterns<paddle::dialect::ScaleSr_Op>(ps);
    patternCreator.createPatterns<paddle::dialect::Sgd_Op>(ps);
    patternCreator.createPatterns<paddle::dialect::SgdDenseParamSparseGrad_Op>(
        ps);
    patternCreator.createPatterns<paddle::dialect::SgdSparseParamSparseGrad_Op>(
        ps);
    patternCreator.createPatterns<paddle::dialect::ShapeOp>(ps);
    patternCreator.createPatterns<paddle::dialect::ShapeSrOp>(ps);
    patternCreator.createPatterns<paddle::dialect::SigmoidOp>(ps);
    patternCreator.createPatterns<paddle::dialect::Sigmoid_Op>(ps);
    patternCreator.createPatterns<paddle::dialect::SoftplusOp>(ps);
    patternCreator.createPatterns<paddle::dialect::SqrtOp>(ps);
    patternCreator.createPatterns<paddle::dialect::SqrtSrOp>(ps);
    patternCreator.createPatterns<paddle::dialect::Sqrt_Op>(ps);
    patternCreator.createPatterns<paddle::dialect::SqrtSr_Op>(ps);
    patternCreator.createPatterns<paddle::dialect::SqueezeOp>(ps);
    patternCreator.createPatterns<paddle::dialect::Squeeze_Op>(ps);
    patternCreator.createPatterns<paddle::dialect::StackOp>(ps);
    patternCreator.createPatterns<paddle::dialect::TanhOp>(ps);
    patternCreator.createPatterns<paddle::dialect::Tanh_Op>(ps);
    patternCreator.createPatterns<paddle::dialect::AbsGradOp>(ps);
    patternCreator.createPatterns<paddle::dialect::ClipGradOp>(ps);
    patternCreator.createPatterns<paddle::dialect::ClipGrad_Op>(ps);
    patternCreator.createPatterns<paddle::dialect::ConcatGradOp>(ps);
    patternCreator.createPatterns<paddle::dialect::Conv2dGradOp>(ps);
    patternCreator.createPatterns<paddle::dialect::Conv3dGradOp>(ps);
    patternCreator.createPatterns<paddle::dialect::DepthwiseConv2dGradOp>(ps);
    patternCreator.createPatterns<paddle::dialect::EluGradOp>(ps);
    patternCreator.createPatterns<paddle::dialect::EluGrad_Op>(ps);
    patternCreator.createPatterns<paddle::dialect::ExpGradOp>(ps);
    patternCreator.createPatterns<paddle::dialect::ExpGrad_Op>(ps);
    patternCreator.createPatterns<paddle::dialect::ExpandGradOp>(ps);
    patternCreator.createPatterns<paddle::dialect::FlattenGradOp>(ps);
    patternCreator.createPatterns<paddle::dialect::FlattenGrad_Op>(ps);
    patternCreator.createPatterns<paddle::dialect::LeakyReluGradOp>(ps);
    patternCreator.createPatterns<paddle::dialect::LeakyReluGrad_Op>(ps);
    patternCreator.createPatterns<paddle::dialect::PreluGradOp>(ps);
    patternCreator.createPatterns<paddle::dialect::Relu6GradOp>(ps);
    patternCreator.createPatterns<paddle::dialect::Relu6Grad_Op>(ps);
    patternCreator.createPatterns<paddle::dialect::ReluGrad_Op>(ps);
    patternCreator.createPatterns<paddle::dialect::SigmoidGradOp>(ps);
    patternCreator.createPatterns<paddle::dialect::SigmoidGrad_Op>(ps);
    patternCreator.createPatterns<paddle::dialect::SqrtGradOp>(ps);
    patternCreator.createPatterns<paddle::dialect::SqrtGrad_Op>(ps);
    patternCreator.createPatterns<paddle::dialect::SqueezeGradOp>(ps);
    patternCreator.createPatterns<paddle::dialect::SqueezeGrad_Op>(ps);
    patternCreator.createPatterns<paddle::dialect::TanhGradOp>(ps);
    patternCreator.createPatterns<paddle::dialect::TanhGrad_Op>(ps);
    patternCreator.createPatterns<paddle::dialect::FcOp>(ps);
    patternCreator.createPatterns<paddle::dialect::FusionGruOp>(ps);
    patternCreator.createPatterns<paddle::dialect::AddNOp>(ps);
    patternCreator.createPatterns<paddle::dialect::Cast_Op>(ps);
    patternCreator.createPatterns<paddle::dialect::Conv2dTransposeOp>(ps);
    patternCreator.createPatterns<paddle::dialect::Conv2dTransposeBiasOp>(ps);
    patternCreator.createPatterns<paddle::dialect::DivideOp>(ps);
    patternCreator.createPatterns<paddle::dialect::Divide_Op>(ps);
    patternCreator.createPatterns<paddle::dialect::GaussianOp>(ps);
    patternCreator.createPatterns<paddle::dialect::HardswishOp>(ps);
    patternCreator.createPatterns<paddle::dialect::LrnOp>(ps);
    patternCreator.createPatterns<paddle::dialect::MatmulWithFlattenOp>(ps);
    patternCreator.createPatterns<paddle::dialect::MaxOp>(ps);
    patternCreator.createPatterns<paddle::dialect::MeanOp>(ps);
    patternCreator.createPatterns<paddle::dialect::MinOp>(ps);
    patternCreator.createPatterns<paddle::dialect::MishOp>(ps);
    patternCreator.createPatterns<paddle::dialect::MultiplyOp>(ps);
    patternCreator.createPatterns<paddle::dialect::MultiplySrOp>(ps);
    patternCreator.createPatterns<paddle::dialect::Multiply_Op>(ps);
    patternCreator.createPatterns<paddle::dialect::MultiplySr_Op>(ps);
    patternCreator.createPatterns<paddle::dialect::PadOp>(ps);
    patternCreator.createPatterns<paddle::dialect::Pool2dOp>(ps);
    patternCreator.createPatterns<paddle::dialect::Reshape_Op>(ps);
    patternCreator.createPatterns<paddle::dialect::SliceOp>(ps);
    patternCreator.createPatterns<paddle::dialect::Softmax_Op>(ps);
    patternCreator.createPatterns<paddle::dialect::SplitOp>(ps);
    patternCreator.createPatterns<paddle::dialect::SplitWithNumOp>(ps);
    patternCreator.createPatterns<paddle::dialect::SubtractOp>(ps);
    patternCreator.createPatterns<paddle::dialect::Subtract_Op>(ps);
    patternCreator.createPatterns<paddle::dialect::SumOp>(ps);
    patternCreator.createPatterns<paddle::dialect::SwishOp>(ps);
    patternCreator.createPatterns<paddle::dialect::Transpose_Op>(ps);
    patternCreator.createPatterns<paddle::dialect::AddDoubleGradOp>(ps);
    patternCreator.createPatterns<paddle::dialect::AddDoubleGrad_Op>(ps);
    patternCreator.createPatterns<paddle::dialect::AddGrad_Op>(ps);
    patternCreator.createPatterns<paddle::dialect::AddTripleGradOp>(ps);
    patternCreator.createPatterns<paddle::dialect::AddTripleGrad_Op>(ps);
    patternCreator.createPatterns<paddle::dialect::BatchNormGradOp>(ps);
    patternCreator.createPatterns<paddle::dialect::DivideGradOp>(ps);
    patternCreator.createPatterns<paddle::dialect::HardswishGradOp>(ps);
    patternCreator.createPatterns<paddle::dialect::HardswishGrad_Op>(ps);
    patternCreator.createPatterns<paddle::dialect::LrnGradOp>(ps);
    patternCreator.createPatterns<paddle::dialect::MatmulWithFlattenGradOp>(ps);
    patternCreator.createPatterns<paddle::dialect::MeanGradOp>(ps);
    patternCreator.createPatterns<paddle::dialect::MishGradOp>(ps);
    patternCreator.createPatterns<paddle::dialect::MishGrad_Op>(ps);
    patternCreator.createPatterns<paddle::dialect::MultiplyGradOp>(ps);
    patternCreator.createPatterns<paddle::dialect::Pool2dGradOp>(ps);
    patternCreator.createPatterns<paddle::dialect::ReshapeGradOp>(ps);
    patternCreator.createPatterns<paddle::dialect::ReshapeGrad_Op>(ps);
    patternCreator.createPatterns<paddle::dialect::SliceGradOp>(ps);
    patternCreator.createPatterns<paddle::dialect::SoftmaxGradOp>(ps);
    patternCreator.createPatterns<paddle::dialect::SubtractGradOp>(ps);
    patternCreator.createPatterns<paddle::dialect::SubtractGrad_Op>(ps);
    patternCreator.createPatterns<paddle::dialect::SumGradOp>(ps);
    patternCreator.createPatterns<paddle::dialect::SwishGradOp>(ps);
    patternCreator.createPatterns<paddle::dialect::SwishGrad_Op>(ps);
    patternCreator.createPatterns<paddle::dialect::TransposeGradOp>(ps);
    patternCreator.createPatterns<paddle::dialect::GeluGradOp>(ps);
    patternCreator.createPatterns<paddle::dialect::ReluGradOp>(ps);
    patternCreator.createPatterns<paddle::dialect::AddOp>(ps);
    patternCreator.createPatterns<paddle::dialect::Add_Op>(ps);
    patternCreator.createPatterns<paddle::dialect::BatchNormOp>(ps);
    patternCreator.createPatterns<paddle::dialect::BatchNorm_Op>(ps);
    patternCreator.createPatterns<paddle::dialect::CastOp>(ps);
    patternCreator.createPatterns<paddle::dialect::FullOp>(ps);
    patternCreator.createPatterns<paddle::dialect::MatmulOp>(ps);
    patternCreator.createPatterns<paddle::dialect::ReshapeOp>(ps);
    patternCreator.createPatterns<paddle::dialect::SoftmaxOp>(ps);
    patternCreator.createPatterns<paddle::dialect::TransposeOp>(ps);
    patternCreator.createPatterns<paddle::dialect::AddGradOp>(ps);
    patternCreator.createPatterns<paddle::dialect::MatmulGradOp>(ps);

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
