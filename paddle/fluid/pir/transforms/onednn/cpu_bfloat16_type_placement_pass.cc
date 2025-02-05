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
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"

namespace {
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
    if (op_info) {
      for (size_t i = 0; i < op->num_results(); ++i) {
        pir::Type type = op->result_type(i);
        if (type.isa<paddle::dialect::DenseTensorType>()) {
          auto dense_type = type.dyn_cast<paddle::dialect::DenseTensorType>();
          auto new_type = paddle::dialect::DenseTensorType::get(
              rewriter.ir_context(),
              paddle::dialect::TransToIrDataType(phi::DataType::BFLOAT16,
                                                 rewriter.ir_context()),
              dense_type.dims(),
              dense_type.data_layout(),
              dense_type.lod(),
              dense_type.offset());
          // set bf16 op tensor output type to bf16.
          op->result(i).set_type(new_type);
        } else if (type.isa<pir::VectorType>()) {
          auto vec_type = type.dyn_cast<pir::VectorType>();
          auto output_num = vec_type.size();
          std::vector<pir::Type> results_type(output_num);
          for (size_t idx = 0; idx < output_num; ++idx) {
            auto dense_type =
                vec_type[idx].dyn_cast<paddle::dialect::DenseTensorType>();
            auto new_type = paddle::dialect::DenseTensorType::get(
                rewriter.ir_context(),
                paddle::dialect::TransToIrDataType(phi::DataType::BFLOAT16,
                                                   rewriter.ir_context()),
                dense_type.dims(),
                dense_type.data_layout(),
                dense_type.lod(),
                dense_type.offset());
            results_type[idx] = new_type;
          }
          auto new_vec_type =
              pir::VectorType::get(rewriter.ir_context(), results_type);
          op->result(i).set_type(new_vec_type);

        } else {
          PADDLE_THROW(common::errors::Unimplemented(
              "result type is not DenseTensorType or VectorType, please close "
              "MKLDNNBf16"));
        }
      }
    }
  }
};

class OneDNNBf16TypePass : public pir::PatternRewritePass {
 public:
  OneDNNBf16TypePass()
      : pir::PatternRewritePass("cpu_bfloat16_type_placement_pass", 2) {}

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
