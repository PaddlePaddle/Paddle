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

#include "paddle/fluid/pir/transforms/onednn/cpu_bfloat16_pass_pattern.h"

#include "paddle/fluid/pir/dialect/operator/ir/onednn_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"

#include "paddle/fluid/framework/new_executor/pir_adaptor/pir_adaptor_util.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/pir/dialect/operator/interface/op_yaml_info.h"
#include "paddle/fluid/pir/dialect/operator/ir/onednn_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_parser.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_util.h"
#include "paddle/fluid/pir/utils/general_functions.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"
#include "paddle/pir/include/pattern_rewrite/pattern_match.h"

namespace {
class CpuBfloat16Pattern : public paddle::drr::DrrPatternBase {
 private:
  std::string bfloat16_ops_;
  uint32_t benefit_;
  mutable std::map<int, int> quantize_in_list_;
  // mutable std::unordered_map<std::string, paddle::drr::Attribute> op_attrs_;

 public:
  CpuBfloat16Pattern(const std::string &bfloat16_ops, uint32_t benefit)
      : bfloat16_ops_(bfloat16_ops), benefit_(benefit) {}

  std::string name() const override {
    return bfloat16_ops_ + "CpuBfloat16Pattern";
  }

  uint32_t benefit() const override { return benefit_; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    std::unordered_map<std::string, paddle::drr::Attribute> op_attrs;
    if (bfloat16_ops_ == "onednn_op.add" || bfloat16_ops_ == "onednn_op.add_") {
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
      op_attrs.emplace("use_quantizer", pat.Attr("use_quantizer"));
      op_attrs.emplace("scale_x", pat.Attr("scale_x"));
      op_attrs.emplace("scale_y", pat.Attr("scale_y"));
      op_attrs.emplace("scale_out", pat.Attr("scale_out"));
    } else if (bfloat16_ops_ == "onednn_op.concat") {
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
      op_attrs.emplace("use_quantizer", pat.Attr("use_quantizer"));

    } else if (bfloat16_ops_ == "onednn_op.conv2d") {
      op_attrs.emplace("strides", pat.Attr("strides"));
      op_attrs.emplace("paddings", pat.Attr("paddings"));
      op_attrs.emplace("padding_algorithm", pat.Attr("padding_algorithm"));
      op_attrs.emplace("dilations", pat.Attr("dilations"));
      op_attrs.emplace("groups", pat.Attr("groups"));
      op_attrs.emplace("data_format", pat.Attr("data_format"));
      op_attrs.emplace("is_test", pat.Attr("is_test"));
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
    } else if (bfloat16_ops_ == "onednn_op.matmul") {
      op_attrs.emplace("transpose_x", pat.Attr("transpose_x"));
      op_attrs.emplace("transpose_y", pat.Attr("transpose_y"));
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));

    } else if (bfloat16_ops_ == "onednn_op.pool2d") {
      op_attrs.emplace("kernel_size", pat.Attr("kernel_size"));
      op_attrs.emplace("strides", pat.Attr("strides"));
      op_attrs.emplace("paddings", pat.Attr("paddings"));
      op_attrs.emplace("ceil_mode", pat.Attr("ceil_mode"));
      op_attrs.emplace("exclusive", pat.Attr("exclusive"));
      op_attrs.emplace("data_format", pat.Attr("data_format"));
      op_attrs.emplace("ceil_mode", pat.Attr("ceil_mode"));
      op_attrs.emplace("exclusive", pat.Attr("exclusive"));
      op_attrs.emplace("data_format", pat.Attr("data_format"));
      op_attrs.emplace("pooling_type", pat.Attr("pooling_type"));
      op_attrs.emplace("global_pooling", pat.Attr("global_pooling"));
      op_attrs.emplace("adaptive", pat.Attr("adaptive"));
      op_attrs.emplace("padding_algorithm", pat.Attr("padding_algorithm"));
      op_attrs.emplace("use_quantizer", pat.Attr("use_quantizer"));
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
      op_attrs.emplace("is_test", pat.Attr("is_test"));
    }

    const auto &op = pat.Op(bfloat16_ops_, op_attrs);
    op({&pat.Tensor("quantize_0"), &pat.Tensor("quantize_1")},
       {&pat.Tensor("out")});

    const std::vector<std::string> permitted_input_names = {
        "x", "y", "input", "residual_param"};

    std::vector<std::string> permitted_output_names = {"xshape"};
    pat.AddConstraint([this](const paddle::drr::MatchContext &match_ctx) {
      auto mkldnn_data_type = match_ctx.Attr<std::string>("mkldnn_data_type");
      if (mkldnn_data_type != "bfloat16") {
        return false;
      }

      bool need_quant = false;
      pir::Operation *input_op = match_ctx.Tensor("out").defining_op();
      uint32_t num_operands = input_op->num_operands();
      for (uint32_t i = 0; i < num_operands; i++) {
        auto *pre_op = pir::GetDefiningOpForInput(input_op, i);
        if (!pre_op->isa<paddle::onednn::dialect::QuantizeOp>()) {
          need_quant = true;
        }
      }
      return need_quant;
    });
    // std::map<int, int> quantize_in_list;
    paddle::drr::ResultPattern res = pat.ResultPattern();

    pat.AddPostProcess(
        [this](const paddle::drr::MatchContext &match_ctx) mutable {
          pir::Operation *op = match_ctx.Tensor("out").defining_op();
          const std::vector<std::string> permitted_input_names = {
              "x", "y", "input", "residual_param"};
          auto op_info =
              pir::IrContext::Instance()->GetRegisteredOpInfo(bfloat16_ops_);
          paddle::dialect::OpYamlInfoParser yaml_parser(
              op_info.GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>()
                  ->get_op_info_(bfloat16_ops_),
              paddle::dialect::IsLegacyOp(bfloat16_ops_));

          auto inputs_name = yaml_parser.InputNames();
          for (auto &input_name : inputs_name) {
            auto it = std::find(permitted_input_names.begin(),
                                permitted_input_names.end(),
                                input_name);
            if (it != permitted_input_names.end()) {
              auto index = yaml_parser.InputName2Id().at(input_name);
              auto *pre_op = pir::GetDefiningOpForInput(op, index);
              if (pre_op->isa<paddle::onednn::dialect::QuantizeOp>()) {
                quantize_in_list_.emplace(index, 0);
              } else {
                quantize_in_list_.emplace(index, 1);
              }
            }
          }
        });
    quantize_in_list_.emplace(0, 1);
    quantize_in_list_.emplace(1, 1);
    std::cout << "This is quantize_in_list_??" << quantize_in_list_.size()
              << std::endl;
    for (auto [index, value] : quantize_in_list_) {
      if (value == 1) {
        const auto &quantize_op =
            res.Op("onednn_op.quantize",
                   {{
                       {"scale", res.Float32Attr(1.f)},
                       {"shift", res.Float32Attr(0.0f)},
                       {"bfloat16", res.BoolAttr(true)},
                       {"is_negative_input", res.BoolAttr(false)},
                       {"output_format", res.StrAttr("NCHW")},
                   }});
        std::cout << "This is index:" << index << std::endl;
        quantize_op({&res.Tensor("quantize_" + std::to_string(index))},
                    {&res.Tensor("quantize_out_" + std::to_string(index))});

      } else {
        std::cout << "This is value??" << value << std::endl;
      }
    }

    const auto &res_op = res.Op(bfloat16_ops_, op_attrs);
    res_op({&res.Tensor("quantize_out_0"), &res.Tensor("quantize_out_1")},
           {&res.Tensor("out")});
  }
};

class CpuBfloat16DequantPattern : public paddle::drr::DrrPatternBase {
 private:
  std::string bfloat16_ops_;
  uint32_t benefit_;
  mutable std::map<int, int> dequantize_in_list_;

 public:
  CpuBfloat16DequantPattern(const std::string &bfloat16_ops, uint32_t benefit)
      : bfloat16_ops_(bfloat16_ops), benefit_(benefit) {}

  std::string name() const override {
    return bfloat16_ops_ + "CpuBfloat16DequantPattern";
  }

  uint32_t benefit() const override { return benefit_; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    std::unordered_map<std::string, paddle::drr::Attribute> op_attrs;
    if (bfloat16_ops_ == "onednn_op.add" || bfloat16_ops_ == "onednn_op.add_") {
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
      op_attrs.emplace("use_quantizer", pat.Attr("use_quantizer"));
      op_attrs.emplace("scale_x", pat.Attr("scale_x"));
      op_attrs.emplace("scale_y", pat.Attr("scale_y"));
      op_attrs.emplace("scale_out", pat.Attr("scale_out"));
    } else if (bfloat16_ops_ == "onednn_op.concat") {
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
      op_attrs.emplace("use_quantizer", pat.Attr("use_quantizer"));

    } else if (bfloat16_ops_ == "onednn_op.conv2d") {
      op_attrs.emplace("strides", pat.Attr("strides"));
      op_attrs.emplace("paddings", pat.Attr("paddings"));
      op_attrs.emplace("padding_algorithm", pat.Attr("padding_algorithm"));
      op_attrs.emplace("dilations", pat.Attr("dilations"));
      op_attrs.emplace("groups", pat.Attr("groups"));
      op_attrs.emplace("data_format", pat.Attr("data_format"));
      op_attrs.emplace("is_test", pat.Attr("is_test"));
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
    } else if (bfloat16_ops_ == "onednn_op.matmul") {
      op_attrs.emplace("transpose_x", pat.Attr("transpose_x"));
      op_attrs.emplace("transpose_y", pat.Attr("transpose_y"));
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));

    } else if (bfloat16_ops_ == "onednn_op.pool2d") {
      op_attrs.emplace("kernel_size", pat.Attr("kernel_size"));
      op_attrs.emplace("strides", pat.Attr("strides"));
      op_attrs.emplace("paddings", pat.Attr("paddings"));
      op_attrs.emplace("ceil_mode", pat.Attr("ceil_mode"));
      op_attrs.emplace("exclusive", pat.Attr("exclusive"));
      op_attrs.emplace("data_format", pat.Attr("data_format"));
      op_attrs.emplace("ceil_mode", pat.Attr("ceil_mode"));
      op_attrs.emplace("exclusive", pat.Attr("exclusive"));
      op_attrs.emplace("data_format", pat.Attr("data_format"));
      op_attrs.emplace("pooling_type", pat.Attr("pooling_type"));
      op_attrs.emplace("global_pooling", pat.Attr("global_pooling"));
      op_attrs.emplace("adaptive", pat.Attr("adaptive"));
      op_attrs.emplace("padding_algorithm", pat.Attr("padding_algorithm"));
      op_attrs.emplace("use_quantizer", pat.Attr("use_quantizer"));
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
      op_attrs.emplace("is_test", pat.Attr("is_test"));
    }

    const auto &op = pat.Op(bfloat16_ops_, op_attrs);
    op({&pat.Tensor("x"), &pat.Tensor("y")}, {&pat.Tensor("out")});

    paddle::drr::ResultPattern res = pat.ResultPattern();

    const std::vector<std::string> permitted_input_names = {
        "x", "y", "input", "residual_param"};

    std::vector<std::string> permitted_output_names = {"xshape"};
    pat.AddConstraint([this](const paddle::drr::MatchContext &match_ctx) {
      bool need_dequant = false;
      pir::Operation *input_op = match_ctx.Tensor("out").defining_op();
      auto next_ops = pir::GetUseOpsForOutput(input_op, 0);
      for (auto [next_op, _] : next_ops) {
        if (!next_op->isa<paddle::onednn::dialect::DequantizeOp>()) {
          need_dequant = true;
        }
      }
      return need_dequant;
    });

    pat.AddPostProcess(
        [this](const paddle::drr::MatchContext &match_ctx) mutable {
          pir::Operation *op = match_ctx.Tensor("out").defining_op();
          std::vector<std::string> nopermitted_output_names = {"xshape"};
          auto op_info =
              pir::IrContext::Instance()->GetRegisteredOpInfo(bfloat16_ops_);
          paddle::dialect::OpYamlInfoParser yaml_parser(
              op_info.GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>()
                  ->get_op_info_(bfloat16_ops_),
              paddle::dialect::IsLegacyOp(bfloat16_ops_));

          auto outputs_name = yaml_parser.OutputNames();
          for (auto &output_name : outputs_name) {
            std::cout << "222222222222222:" << output_name << std::endl;
            auto it = std::find(nopermitted_output_names.begin(),
                                nopermitted_output_names.end(),
                                output_name);
            if (it == nopermitted_output_names.end()) {
              auto index = yaml_parser.OutputName2Id().at(output_name);
              auto next_ops = pir::GetUseOpsForOutput(op, index);
              for (auto [next_op, in] : next_ops) {
                if (in == 0) {
                  if (next_op->isa<paddle::onednn::dialect::DequantizeOp>()) {
                    dequantize_in_list_.emplace(index, 0);
                  } else {
                    dequantize_in_list_.emplace(index, 1);
                  }
                }
              }
            }
          }
        });
    // dequantize_in_list_.emplace(0, 1);

    // for(auto [index, value] : dequantize_in_list_){
    //   // if(value == 1){
    //       const auto &dequantize_op =
    //           res.Op("onednn_op.dequantize",
    //               {{
    //                   {"scale", res.Float32Attr(1.f)},
    //                   {"shift", res.Float32Attr(0.0f)},
    //               }});

    // dequantize_op({&res.Tensor("dequantize_" + std::to_string(index))},
    //                       {&res.Tensor("out")});

    //   }
    // }
    const auto &res_op = res.Op(bfloat16_ops_, op_attrs);
    res_op({&res.Tensor("x"), &res.Tensor("y")}, {&res.Tensor("dequantize_1")});
    const auto &dequantize_op = res.Op("onednn_op.dequantize",
                                       {{
                                           {"scale", res.Float32Attr(1.f)},
                                           {"shift", res.Float32Attr(0.0f)},
                                       }});

    dequantize_op({&res.Tensor("dequantize_1")}, {&res.Tensor("out")});
  }
};

class CpuBfloat16Pass : public pir::PatternRewritePass {
 public:
  CpuBfloat16Pass() : pir::PatternRewritePass("cpu_bfloat16_pass", 3) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    const std::vector<std::string> bfloat16_ops{
        paddle::onednn::dialect::AddOp::name(),
        paddle::onednn::dialect::Add_Op::name(),
        paddle::onednn::dialect::MultiplyOp::name(),
        paddle::onednn::dialect::Multiply_Op::name(),
        paddle::onednn::dialect::ConcatOp::name(),
        paddle::onednn::dialect::Conv2dOp::name(),
        paddle::onednn::dialect::MatmulOp::name(),
        paddle::onednn::dialect::Pool2dOp::name(),

    };

    int benefit_idx = 1;
    for (auto op : bfloat16_ops) {
      ps.Add(paddle::drr::Create<CpuBfloat16Pattern>(context, op, benefit_idx));
      benefit_idx++;
    }
    benefit_idx = 1;
    for (auto op : bfloat16_ops) {
      ps.Add(paddle::drr::Create<CpuBfloat16DequantPattern>(
          context, op, benefit_idx));
      benefit_idx++;
    }

    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateCpuBf16PatternPass() {
  return std::make_unique<CpuBfloat16Pass>();
}
}  // namespace pir

REGISTER_IR_PASS(cpu_bfloat16_pass, CpuBfloat16Pass);
