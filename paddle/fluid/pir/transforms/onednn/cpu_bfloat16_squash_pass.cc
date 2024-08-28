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

#include "paddle/fluid/pir/transforms/onednn/cpu_bfloat16_squash_pass.h"

#include "paddle/fluid/pir/dialect/kernel/ir/kernel_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/onednn_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

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

// Currently quantize_squash_pass is only used by bf16_quantize_pass, which only
// add quantize/dequantize with scale=1.0f & shift=0.0f, hence only deal with
// such situation for simplicity
class DequantQuantBf16SquashPattern
    : public pir::OpRewritePattern<paddle::onednn::dialect::DequantizeOp> {
 public:
  using pir::OpRewritePattern<
      paddle::onednn::dialect::DequantizeOp>::OpRewritePattern;
  bool MatchAndRewrite(
      paddle::onednn::dialect::DequantizeOp op,
      pir::PatternRewriter &rewriter) const override {  // NOLINT
    // The next op should be quant op.
    if (!op.output().HasOneUse()) return false;
    paddle::onednn::dialect::QuantizeOp quant_op =
        pir::GetUseOpsForOutput(op, 0)[0]
            .first->dyn_cast<paddle::onednn::dialect::QuantizeOp>();
    if (!quant_op) return false;
    auto *pre_op = pir::GetDefiningOpForInput(op, 0);
    if (!pre_op) return false;
    auto quant_attributes = quant_op->attributes();
    auto dequant_attributes = op->attributes();
    auto q_scale =
        quant_attributes.at("scale").dyn_cast<pir::FloatAttribute>().data();
    auto dq_scale =
        dequant_attributes.at("scale").dyn_cast<pir::FloatAttribute>().data();
    auto q_shift =
        quant_attributes.at("shift").dyn_cast<pir::FloatAttribute>().data();
    auto dq_shift =
        dequant_attributes.at("shift").dyn_cast<pir::FloatAttribute>().data();

    uint32_t idx = pre_op->num_results();
    for (uint32_t i = 0; i < pre_op->num_results(); i++) {
      if (pre_op->result(i) == op.input()) {
        idx = i;
        break;
      }
    }
    if (idx == pre_op->num_results()) return false;
    if (q_scale != 1.0f || q_shift != 0.0f) return false;

    if (q_scale == dq_scale && q_shift == dq_shift) {
      rewriter.ReplaceAllUsesWith(quant_op.output(), pre_op->result(idx));

      rewriter.EraseOp(quant_op);
      rewriter.EraseOp(op);
    } else {
      return false;
    }
    return true;
  }
};

class DequantQuantBf16MultiUserPattern
    : public pir::OpRewritePattern<paddle::onednn::dialect::DequantizeOp> {
 public:
  using pir::OpRewritePattern<
      paddle::onednn::dialect::DequantizeOp>::OpRewritePattern;
  bool MatchAndRewrite(
      paddle::onednn::dialect::DequantizeOp op,
      pir::PatternRewriter &rewriter) const override {  // NOLINT
    // The user_ops should include quant op.
    auto user_ops = pir::GetUseOpsForOutput(op, 0);
    if (user_ops.size() <= 1) return false;
    // indicate pre_op for future process
    auto *pre_op = pir::GetDefiningOpForInput(op, 0);
    if (!pre_op) return false;
    // check if all user_ops are quantize
    std::vector<bool> all_q(user_ops.size(), true);
    uint32_t idx = 0;
    for (auto [user_op, _] : user_ops) {
      paddle::onednn::dialect::QuantizeOp new_op =
          user_op->dyn_cast<paddle::onednn::dialect::QuantizeOp>();
      if (!new_op) {
        all_q[idx] = false;
      }
      idx++;
    }
    // indicate which output of pre_op is used by dq
    idx = pre_op->num_results();
    for (uint32_t i = 0; i < pre_op->num_results(); i++) {
      if (pre_op->result(i) == op.input()) {
        idx = i;
        break;
      }
    }
    if (idx == pre_op->num_results()) return false;

    auto dequant_attributes = op->attributes();
    auto dq_scale =
        dequant_attributes.at("scale").dyn_cast<pir::FloatAttribute>().data();
    auto dq_shift =
        dequant_attributes.at("shift").dyn_cast<pir::FloatAttribute>().data();

    if (dq_scale != 1.0f || dq_shift != 0.0f) return false;

    bool did_process = false;
    if (std::find(all_q.begin(), all_q.end(), false) == all_q.end()) {
      bool delete_flag = true;
      for (auto [user_op, _] : user_ops) {
        paddle::onednn::dialect::QuantizeOp new_op =
            user_op->dyn_cast<paddle::onednn::dialect::QuantizeOp>();
        auto quant_attributes = new_op->attributes();
        auto q_scale =
            quant_attributes.at("scale").dyn_cast<pir::FloatAttribute>().data();
        auto q_shift =
            quant_attributes.at("shift").dyn_cast<pir::FloatAttribute>().data();

        if (q_scale == dq_scale && q_shift == dq_shift) {
          rewriter.ReplaceAllUsesWith(new_op.output(), pre_op->result(idx));
          rewriter.EraseOp(new_op);
          did_process = true;
        } else {
          delete_flag = false;
        }
      }
      if (delete_flag) rewriter.EraseOp(op);
    } else {
      for (uint32_t i = 0; i < all_q.size(); i++) {
        if (all_q[i]) {
          paddle::onednn::dialect::QuantizeOp new_op =
              user_ops[i]
                  .first->dyn_cast<paddle::onednn::dialect::QuantizeOp>();
          auto quant_attributes = new_op->attributes();
          auto q_scale = quant_attributes.at("scale")
                             .dyn_cast<pir::FloatAttribute>()
                             .data();
          auto q_shift = quant_attributes.at("shift")
                             .dyn_cast<pir::FloatAttribute>()
                             .data();

          if (q_scale == dq_scale && q_shift == dq_shift) {
            rewriter.ReplaceAllUsesWith(new_op.output(), pre_op->result(idx));
            rewriter.EraseOp(new_op);
            did_process = true;
          }
        }
      }
    }
    return did_process;
  }
};

class QuantConvBf16SquashPattern
    : public pir::OpRewritePattern<paddle::onednn::dialect::QuantizeOp> {
 public:
  using pir::OpRewritePattern<
      paddle::onednn::dialect::QuantizeOp>::OpRewritePattern;
  bool MatchAndRewrite(
      paddle::onednn::dialect::QuantizeOp op,
      pir::PatternRewriter &rewriter) const override {  // NOLINT
    if (!op.output().HasOneUse()) return false;
    paddle::onednn::dialect::Conv2dOp next_op =
        pir::GetUseOpsForOutput(op, 0)[0]
            .first->dyn_cast<paddle::onednn::dialect::Conv2dOp>();

    if (!next_op) return false;

    auto quant_attributes = op->attributes();
    auto op_attributes = next_op->attributes();
    auto q_scale =
        quant_attributes.at("scale").dyn_cast<pir::FloatAttribute>().data();
    auto q_shift =
        quant_attributes.at("shift").dyn_cast<pir::FloatAttribute>().data();

    if (q_scale != 1.0f || q_shift != 0.0f) return false;
    if (next_op.input() != op.output()) return false;

    op_attributes["force_fp32_output"] = rewriter.bool_attr(false);
    op_attributes["fuse_residual_connection"] = rewriter.bool_attr(false);
    op_attributes["fuse_activation"] = rewriter.str_attr("");
    op_attributes["fuse_alpha"] = rewriter.float_attr(0.0f);
    op_attributes["fuse_beta"] = rewriter.float_attr(0.0f);
    op_attributes["scale_in"] = rewriter.float_attr(1.0f);
    op_attributes["scale_out"] = rewriter.float_attr(1.0f);
    op_attributes["scale_in_eltwise"] = rewriter.float_attr(1.0f);
    op_attributes["scale_weights"] =
        rewriter.array_attr({rewriter.float_attr(1.0f)});

    pir::IrContext *ctx = pir::IrContext::Instance();
    auto op_info = ctx->GetRegisteredOpInfo(
        paddle::onednn::dialect::FusedConv2dOp::name());
    if (!op_info) return false;

    std::vector<pir::Type> op_item_inner_output_types;
    for (size_t i = 0; i < next_op->num_results(); ++i) {
      pir::Type type = next_op->result_type(i);
      pir::Type new_type =
          create_type<pir::DenseTensorType,
                      paddle::dialect::AllocatedDenseTensorType>(
              type, phi::CPUPlace(), pir::BFloat16Type::get(ctx), ctx);
      // set bf16 op tensor output type to bf16.
      op_item_inner_output_types.push_back(new_type);
    }

    paddle::onednn::dialect::FusedConv2dOp new_conv2d_op =
        rewriter
            .Build(
                std::vector<pir::Value>{
                    op.input(), next_op.filter(), pir::Value{}, pir::Value{}},
                op_attributes,
                op_item_inner_output_types,
                op_info)
            ->dyn_cast<paddle::onednn::dialect::FusedConv2dOp>();

    rewriter.ReplaceAllUsesWith(next_op.out(), new_conv2d_op.output());

    rewriter.EraseOp(next_op);
    rewriter.EraseOp(op);

    return true;
  }
};

class QuantFusedConvBf16SquashPattern
    : public pir::OpRewritePattern<paddle::onednn::dialect::QuantizeOp> {
 public:
  using pir::OpRewritePattern<
      paddle::onednn::dialect::QuantizeOp>::OpRewritePattern;
  bool MatchAndRewrite(
      paddle::onednn::dialect::QuantizeOp op,
      pir::PatternRewriter &rewriter) const override {  // NOLINT
    if (!op.output().HasOneUse()) return false;
    paddle::onednn::dialect::FusedConv2dOp next_op =
        pir::GetUseOpsForOutput(op, 0)[0]
            .first->dyn_cast<paddle::onednn::dialect::FusedConv2dOp>();

    if (!next_op) return false;

    auto quant_attributes = op->attributes();
    auto op_attributes = next_op->attributes();
    auto q_scale =
        quant_attributes.at("scale").dyn_cast<pir::FloatAttribute>().data();
    auto q_shift =
        quant_attributes.at("shift").dyn_cast<pir::FloatAttribute>().data();

    if (q_scale != 1.0f || q_shift != 0.0f) return false;
    if (next_op.input() != op.output()) return false;

    auto op_info = pir::IrContext::Instance()->GetRegisteredOpInfo(
        paddle::onednn::dialect::FusedConv2dOp::name());
    if (!op_info) return false;

    pir::IrContext *ctx = pir::IrContext::Instance();
    std::vector<pir::Type> op_item_inner_output_types;
    for (size_t i = 0; i < next_op->num_results(); ++i) {
      pir::Type type = next_op->result_type(i);
      pir::Type new_type =
          create_type<pir::DenseTensorType,
                      paddle::dialect::AllocatedDenseTensorType>(
              type, phi::CPUPlace(), pir::BFloat16Type::get(ctx), ctx);
      // set bf16 op tensor output type to bf16.
      op_item_inner_output_types.push_back(new_type);
    }

    paddle::onednn::dialect::FusedConv2dOp new_conv2d_op =
        rewriter
            .Build(std::vector<pir::Value>{op.input(),
                                           next_op.filter(),
                                           next_op.bias(),
                                           next_op.residual_param()},
                   op_attributes,
                   op_item_inner_output_types,
                   op_info)
            ->dyn_cast<paddle::onednn::dialect::FusedConv2dOp>();

    rewriter.ReplaceAllUsesWith(next_op.output(), new_conv2d_op.output());
    rewriter.ReplaceAllUsesWith(op.output(), op.input());

    rewriter.EraseOp(op);
    rewriter.EraseOp(next_op);

    return true;
  }
};

class OpDequantBf16SquashPattern
    : public pir::OpRewritePattern<paddle::onednn::dialect::DequantizeOp> {
 public:
  using pir::OpRewritePattern<
      paddle::onednn::dialect::DequantizeOp>::OpRewritePattern;
  bool MatchAndRewrite(
      paddle::onednn::dialect::DequantizeOp op,
      pir::PatternRewriter &rewriter) const override {  // NOLINT
    auto *pre_op = pir::GetDefiningOpForInput(op, 0);
    if (!pre_op) return false;
    auto pre_op_name = pre_op->name();
    if (pre_op_name.find("onednn") == std::string::npos) return false;
    auto op_info = pir::IrContext::Instance()->GetRegisteredOpInfo(pre_op_name);
    if (!op_info) return false;

    auto op_attributes = pre_op->attributes();
    auto dequant_attributes = op->attributes();
    auto dq_scale =
        dequant_attributes.at("scale").dyn_cast<pir::FloatAttribute>().data();
    auto dq_shift =
        dequant_attributes.at("shift").dyn_cast<pir::FloatAttribute>().data();

    if (op_attributes.find("mkldnn_data_type") == op_attributes.end()) {
      return false;
    }
    auto onednn_dtype = op_attributes.at("mkldnn_data_type")
                            .dyn_cast<pir::StrAttribute>()
                            .AsString();

    if (!op.input().HasOneUse()) return false;
    if ((op_attributes.find("fuse_residual_connection") !=
         op_attributes.end()) &&
        (op_attributes.at("fuse_residual_connection")
             .dyn_cast<pir::BoolAttribute>()
             .data() == true)) {
      return false;
    }
    if (onednn_dtype != "bfloat16") return false;
    if (op_attributes.find("force_fp32_output") == op_attributes.end()) {
      return false;
    }
    if (dq_scale != 1.0f || dq_shift != 0.0f) return false;

    uint32_t idx = pre_op->num_results();
    for (uint32_t i = 0; i < pre_op->num_results(); i++) {
      if (pre_op->result(i) == op.input()) {
        idx = i;
        break;
      }
    }
    if (idx == pre_op->num_results()) return false;

    op_attributes["force_fp32_output"] = rewriter.bool_attr(true);

    std::vector<pir::Type> op_item_inner_output_types;
    for (size_t i = 0; i < pre_op->num_results(); ++i) {
      if (i == idx) {
        op_item_inner_output_types.push_back(op->result_type(0));
      } else {
        op_item_inner_output_types.push_back(pre_op->result_type(i));
      }
    }

    pir::Operation *new_op = rewriter.Build(pre_op->operands_source(),
                                            op_attributes,
                                            op_item_inner_output_types,
                                            op_info);

    rewriter.ReplaceOp(pre_op, new_op->results());
    rewriter.ReplaceAllUsesWith(op.output(), new_op->result(idx));
    rewriter.EraseOp(op);

    return true;
  }
};

class CPUBf16QuantizeSquashPass : public pir::PatternRewritePass {
 public:
  CPUBf16QuantizeSquashPass()
      : pir::PatternRewritePass("cpu_bf16_quantize_squash_pass", 3) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    uint32_t benfit = 100;

    auto q_dq_onednn_pattern = std::make_unique<DequantQuantBf16SquashPattern>(
        context, benfit--, std::vector<std::string>{});
    ps.Add(std::move(q_dq_onednn_pattern));

    auto q_dq_multi_onednn_pattern =
        std::make_unique<DequantQuantBf16MultiUserPattern>(
            context, benfit--, std::vector<std::string>{});
    ps.Add(std::move(q_dq_multi_onednn_pattern));

    auto q_conv_onednn_pattern = std::make_unique<QuantConvBf16SquashPattern>(
        context,
        benfit--,
        std::vector<std::string>{
            paddle::onednn::dialect::FusedConv2dOp::name(),
        });
    ps.Add(std::move(q_conv_onednn_pattern));

    auto q_fusedconv_onednn_pattern =
        std::make_unique<QuantFusedConvBf16SquashPattern>(
            context,
            benfit--,
            std::vector<std::string>{
                paddle::onednn::dialect::FusedConv2dOp::name(),
            });
    ps.Add(std::move(q_fusedconv_onednn_pattern));

    auto op_dq_onednn_pattern = std::make_unique<OpDequantBf16SquashPattern>(
        context, benfit--, std::vector<std::string>{});
    ps.Add(std::move(op_dq_onednn_pattern));

    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateCPUBf16QuantizeSquashPass() {
  return std::make_unique<CPUBf16QuantizeSquashPass>();
}

}  // namespace pir

REGISTER_IR_PASS(cpu_bf16_quantize_squash_pass, CPUBf16QuantizeSquashPass);
