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

namespace {
template <class IrType1, class IrType2>
static pir::Type create_type(pir::Type type,
                             const phi::Place& place,
                             pir::Type out_dtype,
                             pir::IrContext* ctx) {
  auto input_type = type.dyn_cast<IrType1>();
  return IrType2::get(ctx,
                      place,
                      out_dtype,
                      input_type.dims(),
                      input_type.data_layout(),
                      input_type.lod(),
                      input_type.offset());
}

class CpuBfloat16TypePattern : public pir::RewritePattern {
 public:
  explicit CpuBfloat16TypePattern(pir::IrContext* context)
      : pir::RewritePattern(MatchAnyOpTypeTag(),
                            1 /*benefit*/,
                            context,
                            {} /*generated_names*/) {}

  bool Match(pir::Operation* op) const override {  // NOLINT
    if (!op->isa<paddle::onednn::dialect::QuantizeOp>() &&
        !op->isa<paddle::onednn::dialect::BilinearInterpOp>() &&
        !op->isa<paddle::onednn::dialect::CastOp>() &&
        !op->isa<paddle::onednn::dialect::Cast_Op>() &&
        !op->isa<paddle::onednn::dialect::ClipOp>() &&
        !op->isa<paddle::onednn::dialect::Clip_Op>() &&
        !op->isa<paddle::onednn::dialect::ConcatOp>() &&
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
        !op->isa<paddle::onednn::dialect::SplitOp>() &&
        !op->isa<paddle::onednn::dialect::SqueezeOp>() &&
        !op->isa<paddle::onednn::dialect::Squeeze_Op>() &&
        !op->isa<paddle::onednn::dialect::SumOp>() &&
        !op->isa<paddle::onednn::dialect::TransposeOp>() &&
        !op->isa<paddle::onednn::dialect::Transpose_Op>() &&
        !op->isa<paddle::onednn::dialect::FusedConv2dOp>() &&
        !op->isa<paddle::onednn::dialect::FusedMatmulOp>()) {
      return false;
    }
    auto op_attr = op->attributes();
    std::string target_op_name = op->name();
    if (target_op_name != "onednn_op.quantize") {
      if (op_attr.find("mkldnn_data_type") != op_attr.end()) {
        auto mkldnn_data_type = op_attr.at("mkldnn_data_type")
                                    .dyn_cast<pir::StrAttribute>()
                                    .AsString();
        if (mkldnn_data_type != "bfloat16") {
          return false;
        }
      }
    }
    return true;
  }

  void Rewrite(pir::Operation* op,
               pir::PatternRewriter& rewriter) const override {  // NOLINT
    auto op_info = pir::IrContext::Instance()->GetRegisteredOpInfo(op->name());
    pir::IrContext* ctx = pir::IrContext::Instance();
    if (op_info) {
      std::vector<pir::Type> op_item_inner_output_types;
      for (size_t i = 0; i < op->num_results(); ++i) {
        pir::Type type = op->result_type(i);
        pir::Type new_type =
            create_type<pir::DenseTensorType,
                        paddle::dialect::AllocatedDenseTensorType>(
                type, phi::CPUPlace(), pir::BFloat16Type::get(ctx), ctx);
        // set bf16 op tensor output type to bf16.
        op_item_inner_output_types.push_back(new_type);
      }
      auto attributes = op->attributes();

      pir::Operation* op_item_inner = rewriter.Build(op->operands_source(),
                                                     attributes,
                                                     op_item_inner_output_types,
                                                     op_info);
      rewriter.ReplaceOp(op, op_item_inner->results());
    }
  }
};

class OneDNNBf16TypePass : public pir::PatternRewritePass {
 public:
  OneDNNBf16TypePass()
      : pir::PatternRewritePass("cpu_bfloat16_type_placement_pass", 3) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);
    ps.Add<CpuBfloat16TypePattern>(context);
    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateCpuBf16Pass() {
  return std::make_unique<OneDNNBf16TypePass>();
}
}  // namespace pir

REGISTER_IR_PASS(cpu_bfloat16_type_placement_pass, OneDNNBf16TypePass);
