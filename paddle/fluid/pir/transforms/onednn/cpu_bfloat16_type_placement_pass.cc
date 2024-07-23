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

#include "paddle/fluid/pir/transforms/onednn/cpu_bfloat16_type_placement_pass.h"

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
class CpuBfloat16Pattern1 : public pir::OpRewritePattern<OpType> {
 public:
  using pir::OpRewritePattern<OpType>::OpRewritePattern;
  bool MatchAndRewrite(
      OpType op,
      pir::PatternRewriter &rewriter) const override {  // NOLINT
    auto op_attr = op->attributes();
    VLOG(0) << "CpuBfloat16Pattern1===";
    std::string target_op_name = op->name();
    if (target_op_name != "onednn_op.quantize") {
      if (op_attr.find("mkldnn_data_type") != op_attr.end() &&
          op_attr.find("mkldnn_data_type")->second != "bfloat16") {
        std::cout << op_attr.find("mkldnn_data_type")->second << std::endl;
        std::cout << "No mkldnn_data_type:" << target_op_name << std::endl;
        // return false;
      }
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
  void CreatePlaceOutType(pir::RewritePatternSet &patternSet) {
    auto pattern = std::make_unique<CpuBfloat16Pattern1<Op>>(
        context, benefit++, std::vector<std::string>{});
    patternSet.Add(std::move(pattern));
  }

  void ClearBenefit() { benefit = 1; }

 private:
  pir::IrContext *context;
  int benefit = 1;
};

class OneDNNBf16Pass : public pir::PatternRewritePass {
 public:
  OneDNNBf16Pass() : pir::PatternRewritePass("cpu_bfloat16_pass", 3) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    PatternCreator patternCreator(context);

    patternCreator.CreatePlaceOutType<paddle::onednn::dialect::QuantizeOp>(ps);

    patternCreator
        .CreatePlaceOutType<paddle::onednn::dialect::BilinearInterpOp>(ps);
    patternCreator.CreatePlaceOutType<paddle::onednn::dialect::CastOp>(ps);
    patternCreator.CreatePlaceOutType<paddle::onednn::dialect::Cast_Op>(ps);
    patternCreator.CreatePlaceOutType<paddle::onednn::dialect::ClipOp>(ps);
    patternCreator.CreatePlaceOutType<paddle::onednn::dialect::Clip_Op>(ps);
    patternCreator.CreatePlaceOutType<paddle::onednn::dialect::ConcatOp>(ps);
    patternCreator.CreatePlaceOutType<paddle::onednn::dialect::Conv2dOp>(ps);
    patternCreator
        .CreatePlaceOutType<paddle::onednn::dialect::Conv2dTransposeOp>(ps);
    patternCreator.CreatePlaceOutType<paddle::onednn::dialect::AddOp>(ps);
    patternCreator.CreatePlaceOutType<paddle::onednn::dialect::Add_Op>(ps);
    patternCreator.CreatePlaceOutType<paddle::onednn::dialect::MultiplyOp>(ps);
    patternCreator.CreatePlaceOutType<paddle::onednn::dialect::Multiply_Op>(ps);
    patternCreator.CreatePlaceOutType<paddle::onednn::dialect::FcOp>(ps);
    patternCreator.CreatePlaceOutType<paddle::onednn::dialect::FusionGruOp>(ps);
    patternCreator.CreatePlaceOutType<paddle::onednn::dialect::GeluOp>(ps);
    patternCreator.CreatePlaceOutType<paddle::onednn::dialect::LayerNormOp>(ps);
    patternCreator.CreatePlaceOutType<paddle::onednn::dialect::MatmulOp>(ps);
    patternCreator.CreatePlaceOutType<paddle::onednn::dialect::Pool2dOp>(ps);
    patternCreator.CreatePlaceOutType<paddle::onednn::dialect::PreluOp>(ps);
    patternCreator.CreatePlaceOutType<paddle::onednn::dialect::ReluOp>(ps);
    patternCreator.CreatePlaceOutType<paddle::onednn::dialect::Relu_Op>(ps);
    patternCreator.CreatePlaceOutType<paddle::onednn::dialect::Reshape_Op>(ps);
    patternCreator.CreatePlaceOutType<paddle::onednn::dialect::ReshapeOp>(ps);
    patternCreator.CreatePlaceOutType<paddle::onednn::dialect::ScaleOp>(ps);
    patternCreator.CreatePlaceOutType<paddle::onednn::dialect::Scale_Op>(ps);
    patternCreator.CreatePlaceOutType<paddle::onednn::dialect::SigmoidOp>(ps);
    patternCreator.CreatePlaceOutType<paddle::onednn::dialect::Sigmoid_Op>(ps);
    patternCreator.CreatePlaceOutType<paddle::onednn::dialect::SliceOp>(ps);
    patternCreator.CreatePlaceOutType<paddle::onednn::dialect::SoftmaxOp>(ps);
    patternCreator.CreatePlaceOutType<paddle::onednn::dialect::Softmax_Op>(ps);
    patternCreator.CreatePlaceOutType<paddle::onednn::dialect::SplitOp>(ps);
    patternCreator.CreatePlaceOutType<paddle::onednn::dialect::SqueezeOp>(ps);
    patternCreator.CreatePlaceOutType<paddle::onednn::dialect::Squeeze_Op>(ps);
    patternCreator.CreatePlaceOutType<paddle::onednn::dialect::SumOp>(ps);
    patternCreator.CreatePlaceOutType<paddle::onednn::dialect::TransposeOp>(ps);
    patternCreator.CreatePlaceOutType<paddle::onednn::dialect::Transpose_Op>(
        ps);

    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateCpuBf16Pass() {
  return std::make_unique<OneDNNBf16Pass>();
}
}  // namespace pir

REGISTER_IR_PASS(cpu_bfloat16_type_placement_pass, OneDNNBf16Pass);
