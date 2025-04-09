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

#include "paddle/fluid/pir/transforms/onednn/cpu_bfloat16_pass.h"

#include "paddle/fluid/pir/dialect/operator/ir/onednn_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"

#include "paddle/fluid/framework/new_executor/pir_adaptor/pir_adaptor_util.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/pir/dialect/operator/interface/op_yaml_info.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_parser.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_util.h"
#include "paddle/fluid/pir/utils/general_functions.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {
class CpuBfloat16Pattern : public paddle::drr::DrrPatternBase {
 private:
  std::string bfloat16_ops_;
  uint32_t benefit_;
  uint32_t index_;

 public:
  CpuBfloat16Pattern(const std::string &bfloat16_ops,
                     uint32_t benefit,
                     uint32_t index)
      : bfloat16_ops_(bfloat16_ops), benefit_(benefit), index_(index) {}

  std::string name() const override {
    return bfloat16_ops_ + "CpuBfloat16Pattern";
  }

  uint32_t benefit() const override { return benefit_; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    std::unordered_map<std::string, paddle::drr::Attribute> op_attrs;
    bool data_format = false;
    if (bfloat16_ops_ == "onednn_op.conv2d") {
      op_attrs.emplace("strides", pat.Attr("strides"));
      op_attrs.emplace("paddings", pat.Attr("paddings"));
      op_attrs.emplace("padding_algorithm", pat.Attr("padding_algorithm"));
      op_attrs.emplace("dilations", pat.Attr("dilations"));
      op_attrs.emplace("groups", pat.Attr("groups"));
      op_attrs.emplace("data_format", pat.Attr("data_format"));
      op_attrs.emplace("is_test", pat.Attr("is_test"));
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
      op_attrs.emplace("force_fp32_output", pat.Attr("force_fp32_output"));
      data_format = true;
    } else if (bfloat16_ops_ == "onednn_op.matmul") {
      op_attrs.emplace("transpose_x", pat.Attr("transpose_x"));
      op_attrs.emplace("transpose_y", pat.Attr("transpose_y"));
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
      op_attrs.emplace("force_fp32_output", pat.Attr("force_fp32_output"));

    } else if (bfloat16_ops_ == "onednn_op.pool2d") {
      // op_attrs.emplace("kernel_size", pat.Attr("kernel_size"));
      op_attrs.emplace("strides", pat.Attr("strides"));
      op_attrs.emplace("paddings", pat.Attr("paddings"));
      op_attrs.emplace("ceil_mode", pat.Attr("ceil_mode"));
      op_attrs.emplace("exclusive", pat.Attr("exclusive"));
      op_attrs.emplace("data_format", pat.Attr("data_format"));
      op_attrs.emplace("ceil_mode", pat.Attr("ceil_mode"));
      op_attrs.emplace("exclusive", pat.Attr("exclusive"));
      op_attrs.emplace("pooling_type", pat.Attr("pooling_type"));
      op_attrs.emplace("global_pooling", pat.Attr("global_pooling"));
      op_attrs.emplace("adaptive", pat.Attr("adaptive"));
      op_attrs.emplace("padding_algorithm", pat.Attr("padding_algorithm"));
      op_attrs.emplace("use_quantizer", pat.Attr("use_quantizer"));
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
      op_attrs.emplace("is_test", pat.Attr("is_test"));
      data_format = true;
    } else if (bfloat16_ops_ == "onednn_op.prelu") {
      op_attrs.emplace("data_format", pat.Attr("data_format"));
      op_attrs.emplace("mode", pat.Attr("mode"));
      op_attrs.emplace("is_test", pat.Attr("is_test"));
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
      data_format = true;
    } else if (bfloat16_ops_ == "onednn_op.sum") {
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
      op_attrs.emplace("keepdim", pat.Attr("keepdim"));
      op_attrs.emplace("dtype", pat.Attr("dtype"));

    } else if (bfloat16_ops_ == "onednn_op.reshape_" ||
               bfloat16_ops_ == "onednn_op.reshape") {
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
      op_attrs.emplace("use_quantizer", pat.Attr("use_quantizer"));

    } else if (bfloat16_ops_ == "onednn_op.scale" ||
               bfloat16_ops_ == "onednn_op.scale_") {
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
      op_attrs.emplace("bias", pat.Attr("bias"));
      op_attrs.emplace("bias_after_scale", pat.Attr("bias_after_scale"));
    } else if (bfloat16_ops_ == "onednn_op.add" ||
               bfloat16_ops_ == "onednn_op.add_") {
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
      op_attrs.emplace("use_quantizer", pat.Attr("use_quantizer"));
      op_attrs.emplace("scale_x", pat.Attr("scale_x"));
      op_attrs.emplace("scale_y", pat.Attr("scale_y"));
      op_attrs.emplace("scale_out", pat.Attr("scale_out"));

    } else if (bfloat16_ops_ == "onednn_op.multiply" ||
               bfloat16_ops_ == "onednn_op.multiply_") {
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
    } else if (bfloat16_ops_ == "onednn_op.squeeze" ||
               bfloat16_ops_ == "onednn_op.squeeze_") {
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
    }

    const auto &op = pat.Op(bfloat16_ops_, op_attrs);
    op({&pat.Tensor("quantize_0"), &pat.Tensor("quantize_1")},
       {&pat.Tensor("out")});

    pat.AddConstraint([this](const paddle::drr::MatchContext &match_ctx) {
      auto mkldnn_data_type = match_ctx.Attr<std::string>("mkldnn_data_type");
      if (mkldnn_data_type != "bfloat16") {
        return false;
      }
      const std::vector<std::string> permitted_input_names = {
          "x", "y", "input", "residual_param"};
      auto op_info =
          pir::IrContext::Instance()->GetRegisteredOpInfo(bfloat16_ops_);
      paddle::dialect::OpYamlInfoParser yaml_parser(
          op_info.GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>()
              ->get_op_info_(bfloat16_ops_),
          paddle::dialect::IsLegacyOp(bfloat16_ops_));
      auto &name2id = yaml_parser.InputName2Id();
      auto find_item = std::find_if(
          name2id.begin(),
          name2id.end(),
          [this](const std::map<std::string, uint32_t>::value_type item) {
            return item.second == index_;
          });

      std::string input_name;
      if (find_item != name2id.end()) {
        input_name = (*find_item).first;
      }

      auto it = std::find(permitted_input_names.begin(),
                          permitted_input_names.end(),
                          input_name);
      if (it == permitted_input_names.end()) {
        return false;
      }

      pir::Operation *input_op = match_ctx.Tensor("out").defining_op();
      uint32_t num_operands = input_op->num_operands();
      if (index_ >= num_operands || !input_op->operand_source(index_)) {
        return false;
      }
      auto *pre_op = pir::GetDefiningOpForInput(input_op, index_);
      if (!pre_op) {
        return false;
      }
      if (!pre_op->isa<paddle::onednn::dialect::QuantizeOp>()) {
        return true;
      }

      return false;
    });
    paddle::drr::ResultPattern res = pat.ResultPattern();

    const auto &quantize_op = res.Op(
        "onednn_op.quantize",
        {{
            {"scale", res.Float32Attr(1.f)},
            {"shift", res.Float32Attr(0.0f)},
            {"bfloat16", res.BoolAttr(true)},
            {"is_negative_input", res.BoolAttr(false)},
            {"output_format",
             data_format ? pat.Attr("data_format") : res.StrAttr("NCHW")},
        }});
    quantize_op({&res.Tensor("quantize_" + std::to_string(index_))},
                {&res.Tensor("quantize_out_" + std::to_string(index_))});

    const auto &res_op = res.Op(bfloat16_ops_, op_attrs);
    if (index_ == 0) {
      res_op({&res.Tensor("quantize_out_0"), &res.Tensor("quantize_1")},
             {&res.Tensor("out")});

    } else if (index_ == 1) {
      res_op({&res.Tensor("quantize_0"), &res.Tensor("quantize_out_1")},
             {&res.Tensor("out")});
    }
  }
};

class CpuBfloat16DequantPattern : public paddle::drr::DrrPatternBase {
 private:
  std::string bfloat16_ops_;
  uint32_t benefit_;
  uint32_t index_;

 public:
  CpuBfloat16DequantPattern(const std::string &bfloat16_ops, uint32_t benefit)
      : bfloat16_ops_(bfloat16_ops), benefit_(benefit), index_(0) {}

  std::string name() const override {
    return bfloat16_ops_ + "CpuBfloat16DequantPattern";
  }

  uint32_t benefit() const override { return benefit_; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    std::unordered_map<std::string, paddle::drr::Attribute> op_attrs;
    if (bfloat16_ops_ == "onednn_op.conv2d") {
      op_attrs.emplace("strides", pat.Attr("strides"));
      op_attrs.emplace("paddings", pat.Attr("paddings"));
      op_attrs.emplace("padding_algorithm", pat.Attr("padding_algorithm"));
      op_attrs.emplace("dilations", pat.Attr("dilations"));
      op_attrs.emplace("groups", pat.Attr("groups"));
      op_attrs.emplace("data_format", pat.Attr("data_format"));
      op_attrs.emplace("is_test", pat.Attr("is_test"));
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
      op_attrs.emplace("force_fp32_output", pat.Attr("force_fp32_output"));
    } else if (bfloat16_ops_ == "onednn_op.matmul") {
      op_attrs.emplace("transpose_x", pat.Attr("transpose_x"));
      op_attrs.emplace("transpose_y", pat.Attr("transpose_y"));
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
      op_attrs.emplace("force_fp32_output", pat.Attr("force_fp32_output"));

    } else if (bfloat16_ops_ == "onednn_op.pool2d") {
      // op_attrs.emplace("kernel_size", pat.Attr("kernel_size"));
      op_attrs.emplace("strides", pat.Attr("strides"));
      op_attrs.emplace("paddings", pat.Attr("paddings"));
      op_attrs.emplace("ceil_mode", pat.Attr("ceil_mode"));
      op_attrs.emplace("exclusive", pat.Attr("exclusive"));
      op_attrs.emplace("data_format", pat.Attr("data_format"));
      op_attrs.emplace("ceil_mode", pat.Attr("ceil_mode"));
      op_attrs.emplace("exclusive", pat.Attr("exclusive"));
      op_attrs.emplace("pooling_type", pat.Attr("pooling_type"));
      op_attrs.emplace("global_pooling", pat.Attr("global_pooling"));
      op_attrs.emplace("adaptive", pat.Attr("adaptive"));
      op_attrs.emplace("padding_algorithm", pat.Attr("padding_algorithm"));
      op_attrs.emplace("use_quantizer", pat.Attr("use_quantizer"));
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
      op_attrs.emplace("is_test", pat.Attr("is_test"));

    } else if (bfloat16_ops_ == "onednn_op.prelu") {
      op_attrs.emplace("data_format", pat.Attr("data_format"));
      op_attrs.emplace("mode", pat.Attr("mode"));
      op_attrs.emplace("is_test", pat.Attr("is_test"));
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));

    } else if (bfloat16_ops_ == "onednn_op.sum") {
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
      op_attrs.emplace("keepdim", pat.Attr("keepdim"));
      op_attrs.emplace("dtype", pat.Attr("dtype"));

    } else if (bfloat16_ops_ == "onednn_op.reshape_" ||
               bfloat16_ops_ == "onednn_op.reshape") {
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
      op_attrs.emplace("use_quantizer", pat.Attr("use_quantizer"));

    } else if (bfloat16_ops_ == "onednn_op.scale" ||
               bfloat16_ops_ == "onednn_op.scale_") {
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
      op_attrs.emplace("bias", pat.Attr("bias"));
      op_attrs.emplace("bias_after_scale", pat.Attr("bias_after_scale"));
    } else if (bfloat16_ops_ == "onednn_op.add" ||
               bfloat16_ops_ == "onednn_op.add_") {
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
      op_attrs.emplace("use_quantizer", pat.Attr("use_quantizer"));
      op_attrs.emplace("scale_x", pat.Attr("scale_x"));
      op_attrs.emplace("scale_y", pat.Attr("scale_y"));
      op_attrs.emplace("scale_out", pat.Attr("scale_out"));

    } else if (bfloat16_ops_ == "onednn_op.multiply" ||
               bfloat16_ops_ == "onednn_op.multiply_") {
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
    } else if (bfloat16_ops_ == "onednn_op.squeeze" ||
               bfloat16_ops_ == "onednn_op.squeeze_") {
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
    }

    const auto &op = pat.Op(bfloat16_ops_, op_attrs);
    op({&pat.Tensor("x"), &pat.Tensor("y")}, {&pat.Tensor("out")});

    paddle::drr::ResultPattern res = pat.ResultPattern();

    pat.AddConstraint([this](const paddle::drr::MatchContext &match_ctx) {
      bool need_dequant = false;
      pir::Operation *input_op = match_ctx.Tensor("out").defining_op();

      auto mkldnn_data_type = match_ctx.Attr<std::string>("mkldnn_data_type");
      if (mkldnn_data_type != "bfloat16") {
        return false;
      }

      std::vector<std::string> no_permitted_output_names = {"xshape"};
      auto op_info =
          pir::IrContext::Instance()->GetRegisteredOpInfo(bfloat16_ops_);
      paddle::dialect::OpYamlInfoParser yaml_parser(
          op_info.GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>()
              ->get_op_info_(bfloat16_ops_),
          paddle::dialect::IsLegacyOp(bfloat16_ops_));

      auto &name2id = yaml_parser.OutputName2Id();
      auto find_item = std::find_if(
          name2id.begin(),
          name2id.end(),
          [this](const std::map<std::string, uint32_t>::value_type item) {
            return item.second == index_;
          });

      std::string output_name;
      if (find_item != name2id.end()) {
        output_name = (*find_item).first;
      }

      auto it = std::find(no_permitted_output_names.begin(),
                          no_permitted_output_names.end(),
                          output_name);
      if (it != no_permitted_output_names.end()) {
        return false;
      }

      auto next_ops = pir::GetUseOpsForOutput(input_op, 0);
      for (auto [next_op, _] : next_ops) {
        if (!next_op->isa<paddle::onednn::dialect::DequantizeOp>()) {
          need_dequant = true;
        }
      }
      return need_dequant;
    });

    const auto &res_op = res.Op(bfloat16_ops_, op_attrs);
    res_op({&res.Tensor("x"), &res.Tensor("y")}, {&res.Tensor("dequantize")});
    const auto &dequantize_op = res.Op("onednn_op.dequantize",
                                       {{
                                           {"scale", res.Float32Attr(1.f)},
                                           {"shift", res.Float32Attr(0.0f)},
                                       }});

    dequantize_op({&res.Tensor("dequantize")}, {&res.Tensor("out")});
  }
};

class CpuBfloat16PatternOne_one : public paddle::drr::DrrPatternBase {
 private:
  std::string bfloat16_ops_;
  uint32_t benefit_;
  uint32_t index_;

 public:
  CpuBfloat16PatternOne_one(const std::string &bfloat16_ops, uint32_t benefit)
      : bfloat16_ops_(bfloat16_ops), benefit_(benefit), index_(0) {}

  std::string name() const override {
    return bfloat16_ops_ + "CpuBfloat16PatternOne_one";
  }

  uint32_t benefit() const override { return benefit_; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    std::unordered_map<std::string, paddle::drr::Attribute> op_attrs;
    bool data_format = false;
    if (bfloat16_ops_ == "onednn_op.gelu") {
      op_attrs.emplace("approximate", pat.Attr("approximate"));
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
    } else if (bfloat16_ops_ == "onednn_op.softmax" ||
               bfloat16_ops_ == "onednn_op.softmax_") {
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
      op_attrs.emplace("axis", pat.Attr("axis"));
      op_attrs.emplace("data_format", pat.Attr("data_format"));
      op_attrs.emplace("is_test", pat.Attr("is_test"));
      data_format = true;
    } else if (bfloat16_ops_ == "onednn_op.transpose" ||
               bfloat16_ops_ == "onednn_op.transpose_") {
      op_attrs.emplace("perm", pat.Attr("perm"));
      op_attrs.emplace("data_format", pat.Attr("data_format"));
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
      data_format = true;
    } else if (bfloat16_ops_ == "onednn_op.relu" ||
               bfloat16_ops_ == "onednn_op.relu_") {
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));

    } else if (bfloat16_ops_ == "onednn_op.sigmoid" ||
               bfloat16_ops_ == "onednn_op.sigmoid_") {
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
    }

    const auto &op = pat.Op(bfloat16_ops_, op_attrs);
    op({&pat.Tensor("quantize_0")}, {&pat.Tensor("out")});

    pat.AddConstraint([this](const paddle::drr::MatchContext &match_ctx) {
      auto mkldnn_data_type = match_ctx.Attr<std::string>("mkldnn_data_type");
      if (mkldnn_data_type != "bfloat16") {
        return false;
      }
      const std::vector<std::string> permitted_input_names = {
          "x", "y", "input", "residual_param"};
      auto op_info =
          pir::IrContext::Instance()->GetRegisteredOpInfo(bfloat16_ops_);
      paddle::dialect::OpYamlInfoParser yaml_parser(
          op_info.GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>()
              ->get_op_info_(bfloat16_ops_),
          paddle::dialect::IsLegacyOp(bfloat16_ops_));
      auto &name2id = yaml_parser.InputName2Id();
      auto find_item = std::find_if(
          name2id.begin(),
          name2id.end(),
          [this](const std::map<std::string, uint32_t>::value_type item) {
            return item.second == index_;
          });

      std::string input_name;
      if (find_item != name2id.end()) {
        input_name = (*find_item).first;
      }

      auto it = std::find(permitted_input_names.begin(),
                          permitted_input_names.end(),
                          input_name);
      if (it == permitted_input_names.end()) {
        return false;
      }

      pir::Operation *input_op = match_ctx.Tensor("out").defining_op();
      uint32_t num_operands = input_op->num_operands();
      if (index_ >= num_operands || !input_op->operand_source(index_)) {
        return false;
      }

      auto *pre_op = pir::GetDefiningOpForInput(input_op, index_);
      if (!pre_op) {
        return false;
      }
      if (!pre_op->isa<paddle::onednn::dialect::QuantizeOp>()) {
        return true;
      }

      return false;
    });
    paddle::drr::ResultPattern res = pat.ResultPattern();

    const auto &quantize_op = res.Op(
        "onednn_op.quantize",
        {{
            {"scale", res.Float32Attr(1.f)},
            {"shift", res.Float32Attr(0.0f)},
            {"bfloat16", res.BoolAttr(true)},
            {"is_negative_input", res.BoolAttr(false)},
            {"output_format",
             data_format ? pat.Attr("data_format") : res.StrAttr("NCHW")},
        }});
    quantize_op({&res.Tensor("quantize_0")}, {&res.Tensor("quantize_out_0")});

    const auto &res_op = res.Op(bfloat16_ops_, op_attrs);
    res_op({&res.Tensor("quantize_out_0")}, {&res.Tensor("out")});
  }
};

class CpuBfloat16DequantPatternOne_one : public paddle::drr::DrrPatternBase {
 private:
  std::string bfloat16_ops_;
  uint32_t benefit_;
  uint32_t index_;

 public:
  CpuBfloat16DequantPatternOne_one(const std::string &bfloat16_ops,
                                   uint32_t benefit)
      : bfloat16_ops_(bfloat16_ops), benefit_(benefit), index_(0) {}

  std::string name() const override {
    return bfloat16_ops_ + "CpuBfloat16DequantPatternOne_one";
  }

  uint32_t benefit() const override { return benefit_; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    std::unordered_map<std::string, paddle::drr::Attribute> op_attrs;
    if (bfloat16_ops_ == "onednn_op.gelu") {
      op_attrs.emplace("approximate", pat.Attr("approximate"));
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
    } else if (bfloat16_ops_ == "onednn_op.softmax" ||
               bfloat16_ops_ == "onednn_op.softmax_") {
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
      op_attrs.emplace("axis", pat.Attr("axis"));
      op_attrs.emplace("data_format", pat.Attr("data_format"));
      op_attrs.emplace("is_test", pat.Attr("is_test"));

    } else if (bfloat16_ops_ == "onednn_op.transpose" ||
               bfloat16_ops_ == "onednn_op.transpose_") {
      op_attrs.emplace("perm", pat.Attr("perm"));
      op_attrs.emplace("data_format", pat.Attr("data_format"));
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
    } else if (bfloat16_ops_ == "onednn_op.relu" ||
               bfloat16_ops_ == "onednn_op.relu_") {
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));

    } else if (bfloat16_ops_ == "onednn_op.sigmoid" ||
               bfloat16_ops_ == "onednn_op.sigmoid_") {
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
    }

    const auto &op = pat.Op(bfloat16_ops_, op_attrs);
    op({&pat.Tensor("x")}, {&pat.Tensor("out")});

    paddle::drr::ResultPattern res = pat.ResultPattern();

    pat.AddConstraint([this](const paddle::drr::MatchContext &match_ctx) {
      bool need_dequant = false;
      pir::Operation *input_op = match_ctx.Tensor("out").defining_op();

      auto mkldnn_data_type = match_ctx.Attr<std::string>("mkldnn_data_type");
      if (mkldnn_data_type != "bfloat16") {
        return false;
      }

      std::vector<std::string> no_permitted_output_names = {"xshape"};
      auto op_info =
          pir::IrContext::Instance()->GetRegisteredOpInfo(bfloat16_ops_);
      paddle::dialect::OpYamlInfoParser yaml_parser(
          op_info.GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>()
              ->get_op_info_(bfloat16_ops_),
          paddle::dialect::IsLegacyOp(bfloat16_ops_));

      auto &name2id = yaml_parser.OutputName2Id();
      auto find_item = std::find_if(
          name2id.begin(),
          name2id.end(),
          [this](const std::map<std::string, uint32_t>::value_type item) {
            return item.second == index_;
          });

      std::string output_name;
      if (find_item != name2id.end()) {
        output_name = (*find_item).first;
      }

      auto it = std::find(no_permitted_output_names.begin(),
                          no_permitted_output_names.end(),
                          output_name);
      if (it != no_permitted_output_names.end()) {
        return false;
      }

      auto next_ops = pir::GetUseOpsForOutput(input_op, 0);
      for (auto [next_op, _] : next_ops) {
        if (!next_op->isa<paddle::onednn::dialect::DequantizeOp>()) {
          need_dequant = true;
        }
      }
      return need_dequant;
    });

    const auto &res_op = res.Op(bfloat16_ops_, op_attrs);
    res_op({&res.Tensor("x")}, {&res.Tensor("dequantize")});
    const auto &dequantize_op = res.Op("onednn_op.dequantize",
                                       {{
                                           {"scale", res.Float32Attr(1.f)},
                                           {"shift", res.Float32Attr(0.0f)},
                                       }});

    dequantize_op({&res.Tensor("dequantize")}, {&res.Tensor("out")});
  }
};

class CpuBfloat16Pattern2_2 : public paddle::drr::DrrPatternBase {
 private:
  std::string bfloat16_ops_;
  uint32_t benefit_;
  uint32_t index_;

 public:
  CpuBfloat16Pattern2_2(const std::string &bfloat16_ops,
                        uint32_t benefit,
                        uint32_t index)
      : bfloat16_ops_(bfloat16_ops), benefit_(benefit), index_(index) {}

  std::string name() const override {
    return bfloat16_ops_ + "CpuBfloat16Pattern2_2";
  }

  uint32_t benefit() const override { return benefit_; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    std::unordered_map<std::string, paddle::drr::Attribute> op_attrs;
    if (bfloat16_ops_ == "onednn_op.squeeze" ||
        bfloat16_ops_ == "onednn_op.squeeze_") {
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
    }
    const auto &op = pat.Op(bfloat16_ops_, op_attrs);

    op({&pat.Tensor("quantize_0"), &pat.Tensor("quantize_1")},
       {&pat.Tensor("out_0"), &pat.Tensor("out_1")});

    pat.AddConstraint([this](const paddle::drr::MatchContext &match_ctx) {
      auto mkldnn_data_type = match_ctx.Attr<std::string>("mkldnn_data_type");
      if (mkldnn_data_type != "bfloat16") {
        return false;
      }
      const std::vector<std::string> permitted_input_names = {
          "x", "y", "input", "residual_param"};
      auto op_info =
          pir::IrContext::Instance()->GetRegisteredOpInfo(bfloat16_ops_);
      paddle::dialect::OpYamlInfoParser yaml_parser(
          op_info.GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>()
              ->get_op_info_(bfloat16_ops_),
          paddle::dialect::IsLegacyOp(bfloat16_ops_));
      auto &name2id = yaml_parser.InputName2Id();
      auto find_item = std::find_if(
          name2id.begin(),
          name2id.end(),
          [this](const std::map<std::string, uint32_t>::value_type item) {
            return item.second == index_;
          });

      std::string input_name;
      if (find_item != name2id.end()) {
        input_name = (*find_item).first;
      }

      auto it = std::find(permitted_input_names.begin(),
                          permitted_input_names.end(),
                          input_name);
      if (it == permitted_input_names.end()) {
        return false;
      }

      pir::Operation *input_op = match_ctx.Tensor("out_0").defining_op();
      uint32_t num_operands = input_op->num_operands();
      if (index_ >= num_operands || !input_op->operand_source(index_)) {
        return false;
      }
      auto *pre_op = pir::GetDefiningOpForInput(input_op, index_);
      if (!pre_op->isa<paddle::onednn::dialect::QuantizeOp>()) {
        return true;
      }

      return false;
    });
    paddle::drr::ResultPattern res = pat.ResultPattern();

    const auto &quantize_op =
        res.Op("onednn_op.quantize",
               {{
                   {"scale", res.Float32Attr(1.f)},
                   {"shift", res.Float32Attr(0.0f)},
                   {"bfloat16", res.BoolAttr(true)},
                   {"is_negative_input", res.BoolAttr(false)},
                   {"output_format", res.StrAttr("NCHW")},
               }});
    quantize_op({&res.Tensor("quantize_" + std::to_string(index_))},
                {&res.Tensor("quantize_out_" + std::to_string(index_))});

    const auto &res_op = res.Op(bfloat16_ops_, op_attrs);
    if (index_ == 0) {
      res_op({&res.Tensor("quantize_out_0"), &res.Tensor("quantize_1")},
             {{&res.Tensor("out_0"), &res.Tensor("out_1")}});

    } else {
      res_op({&res.Tensor("quantize_0"), &res.Tensor("quantize_out_1")},
             {{&res.Tensor("out_0"), &res.Tensor("out_1")}});
    }
  }
};

class CpuBfloat16DequantPattern2_2 : public paddle::drr::DrrPatternBase {
 private:
  std::string bfloat16_ops_;
  uint32_t benefit_;
  uint32_t index_;

 public:
  CpuBfloat16DequantPattern2_2(const std::string &bfloat16_ops,
                               uint32_t benefit,
                               uint32_t index)
      : bfloat16_ops_(bfloat16_ops), benefit_(benefit), index_(index) {}

  std::string name() const override {
    return bfloat16_ops_ + "CpuBfloat16DequantPattern2_2";
  }

  uint32_t benefit() const override { return benefit_; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    std::unordered_map<std::string, paddle::drr::Attribute> op_attrs;
    if (bfloat16_ops_ == "onednn_op.squeeze" ||
        bfloat16_ops_ == "onednn_op.squeeze_") {
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
    }
    const auto &op = pat.Op(bfloat16_ops_, op_attrs);
    op({&pat.Tensor("x"), &pat.Tensor("y")},
       {&pat.Tensor("out_0"), &pat.Tensor("out_1")});

    paddle::drr::ResultPattern res = pat.ResultPattern();

    pat.AddConstraint([this](const paddle::drr::MatchContext &match_ctx) {
      bool need_dequant = false;
      pir::Operation *input_op = match_ctx.Tensor("out_0").defining_op();

      auto mkldnn_data_type = match_ctx.Attr<std::string>("mkldnn_data_type");
      if (mkldnn_data_type != "bfloat16") {
        return false;
      }

      std::vector<std::string> no_permitted_output_names = {"xshape"};
      auto op_info =
          pir::IrContext::Instance()->GetRegisteredOpInfo(bfloat16_ops_);
      paddle::dialect::OpYamlInfoParser yaml_parser(
          op_info.GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>()
              ->get_op_info_(bfloat16_ops_),
          paddle::dialect::IsLegacyOp(bfloat16_ops_));

      auto &name2id = yaml_parser.OutputName2Id();
      auto find_item = std::find_if(
          name2id.begin(),
          name2id.end(),
          [this](const std::map<std::string, uint32_t>::value_type item) {
            return item.second == index_;
          });

      std::string output_name;
      if (find_item != name2id.end()) {
        output_name = (*find_item).first;
      }

      auto it = std::find(no_permitted_output_names.begin(),
                          no_permitted_output_names.end(),
                          output_name);
      if (it != no_permitted_output_names.end()) {
        return false;
      }

      auto next_ops = pir::GetUseOpsForOutput(input_op, index_);
      for (auto [next_op, _] : next_ops) {
        if (!next_op->isa<paddle::onednn::dialect::DequantizeOp>()) {
          need_dequant = true;
        }
      }
      return need_dequant;
    });

    const auto &res_op = res.Op(bfloat16_ops_, op_attrs);
    if (index_ == 0) {
      res_op({&res.Tensor("x"), &res.Tensor("y")},
             {&res.Tensor("dequantize_0"), &res.Tensor("out_1")});
      const auto &dequantize_op = res.Op("onednn_op.dequantize",
                                         {{
                                             {"scale", res.Float32Attr(1.f)},
                                             {"shift", res.Float32Attr(0.0f)},
                                         }});

      dequantize_op({&res.Tensor("dequantize_0")}, {&res.Tensor("out_0")});
    } else {
      res_op({&res.Tensor("x"), &res.Tensor("y")},
             {&res.Tensor("out_0"), &res.Tensor("dequantize_1")});
      const auto &dequantize_op = res.Op("onednn_op.dequantize",
                                         {{
                                             {"scale", res.Float32Attr(1.f)},
                                             {"shift", res.Float32Attr(0.0f)},
                                         }});

      dequantize_op({&res.Tensor("dequantize_1")}, {&res.Tensor("out_1")});
    }
  }
};

class CpuBfloat16PatternThree_one : public paddle::drr::DrrPatternBase {
 private:
  std::string bfloat16_ops_;
  uint32_t benefit_;
  uint32_t index_;

 public:
  CpuBfloat16PatternThree_one(const std::string &bfloat16_ops,
                              uint32_t benefit,
                              uint32_t index)
      : bfloat16_ops_(bfloat16_ops), benefit_(benefit), index_(index) {}

  std::string name() const override {
    return bfloat16_ops_ + "CpuBfloat16PatternThree_one";
  }

  uint32_t benefit() const override { return benefit_; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    std::unordered_map<std::string, paddle::drr::Attribute> op_attrs;
    bool data_format = false;
    if (bfloat16_ops_ == "onednn_op.fc") {
      op_attrs.emplace("in_num_col_dims", pat.Attr("in_num_col_dims"));
      op_attrs.emplace("activation_type", pat.Attr("activation_type"));
      op_attrs.emplace("padding_weights", pat.Attr("padding_weights"));
      op_attrs.emplace("use_quantizer", pat.Attr("use_quantizer"));
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
      op_attrs.emplace("scale_in", pat.Attr("scale_in"));
      op_attrs.emplace("scale_weights", pat.Attr("scale_weights"));
      op_attrs.emplace("scale_out", pat.Attr("scale_out"));
      op_attrs.emplace("force_fp32_output", pat.Attr("force_fp32_output"));
      op_attrs.emplace("fuse_activation", pat.Attr("fuse_activation"));
      op_attrs.emplace("fuse_alpha", pat.Attr("fuse_alpha"));
      op_attrs.emplace("fuse_beta", pat.Attr("fuse_beta"));
      op_attrs.emplace("fused_output_scale", pat.Attr("fused_output_scale"));
      op_attrs.emplace("fused_reshape2_shape",
                       pat.Attr("fused_reshape2_shape"));
    } else if (bfloat16_ops_ == "onednn_op.slice") {
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
      op_attrs.emplace("decrease_axis", pat.Attr("decrease_axis"));
      op_attrs.emplace("infer_flags", pat.Attr("infer_flags"));
      op_attrs.emplace("axes", pat.Attr("axes"));

    } else if (bfloat16_ops_ == "onednn_op.split") {
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
    } else if (bfloat16_ops_ == "onednn_op.fused_matmul") {
      op_attrs.emplace("trans_x", pat.Attr("trans_x"));
      op_attrs.emplace("trans_y", pat.Attr("trans_y"));
      op_attrs.emplace("matmul_alpha", pat.Attr("matmul_alpha"));
      op_attrs.emplace("fused_output_scale", pat.Attr("fused_output_scale"));
      op_attrs.emplace("fused_reshape_x", pat.Attr("fused_reshape_x"));
      op_attrs.emplace("fused_transpose_x", pat.Attr("fused_transpose_x"));
      op_attrs.emplace("fused_reshape_y", pat.Attr("fused_reshape_y"));
      op_attrs.emplace("fused_transpose_y", pat.Attr("fused_transpose_y"));
      op_attrs.emplace("fused_reshape_out", pat.Attr("fused_reshape_out"));
      op_attrs.emplace("fused_transpose_out", pat.Attr("fused_transpose_out"));
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
      op_attrs.emplace("scale_x", pat.Attr("scale_x"));
      op_attrs.emplace("scale_y", pat.Attr("scale_y"));
      op_attrs.emplace("scale_in_eltwise", pat.Attr("scale_in_eltwise"));
      op_attrs.emplace("scale_out", pat.Attr("scale_out"));
      op_attrs.emplace("force_fp32_output", pat.Attr("force_fp32_output"));
      op_attrs.emplace("fuse_activation", pat.Attr("fuse_activation"));
      op_attrs.emplace("fuse_alpha", pat.Attr("fuse_alpha"));
      op_attrs.emplace("fuse_beta", pat.Attr("fuse_beta"));
    } else if (bfloat16_ops_ == "onednn_op.clip" ||
               bfloat16_ops_ == "onednn_op.clip_") {
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
    } else if (bfloat16_ops_ == "onednn_op.conv2d_transpose") {
      op_attrs.emplace("is_test", pat.Attr("is_test"));
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
      op_attrs.emplace("data_format", pat.Attr("data_format"));
      op_attrs.emplace("dilations", pat.Attr("dilations"));
      op_attrs.emplace("groups", pat.Attr("groups"));
      op_attrs.emplace("padding_algorithm", pat.Attr("padding_algorithm"));
      op_attrs.emplace("output_padding", pat.Attr("output_padding"));
      op_attrs.emplace("paddings", pat.Attr("paddings"));
      op_attrs.emplace("strides", pat.Attr("strides"));
      op_attrs.emplace("force_fp32_output", pat.Attr("force_fp32_output"));
      data_format = true;
    }

    const auto &op = pat.Op(bfloat16_ops_, op_attrs);
    op({&pat.Tensor("quantize_0"),
        &pat.Tensor("quantize_1"),
        &pat.Tensor("quantize_2")},
       {&pat.Tensor("out")});

    pat.AddConstraint([this](const paddle::drr::MatchContext &match_ctx) {
      auto mkldnn_data_type = match_ctx.Attr<std::string>("mkldnn_data_type");
      if (mkldnn_data_type != "bfloat16") {
        return false;
      }
      // For fused_matmul, it name residual_data as residual_param
      const std::vector<std::string> permitted_input_names = {
          "x", "y", "input", "residual_param", "residual_data"};
      auto op_info =
          pir::IrContext::Instance()->GetRegisteredOpInfo(bfloat16_ops_);
      paddle::dialect::OpYamlInfoParser yaml_parser(
          op_info.GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>()
              ->get_op_info_(bfloat16_ops_),
          paddle::dialect::IsLegacyOp(bfloat16_ops_));
      auto &name2id = yaml_parser.InputName2Id();
      auto find_item = std::find_if(
          name2id.begin(),
          name2id.end(),
          [this](const std::map<std::string, uint32_t>::value_type item) {
            return item.second == index_;
          });

      std::string input_name;
      if (find_item != name2id.end()) {
        input_name = (*find_item).first;
      }

      auto it = std::find(permitted_input_names.begin(),
                          permitted_input_names.end(),
                          input_name);
      if (it == permitted_input_names.end()) {
        return false;
      }

      pir::Operation *input_op = match_ctx.Tensor("out").defining_op();
      uint32_t num_operands = input_op->num_operands();
      if (index_ >= num_operands || !input_op->operand_source(index_)) {
        return false;
      }
      auto *pre_op = pir::GetDefiningOpForInput(input_op, index_);
      if (!pre_op) {
        return false;
      }
      if (!pre_op->isa<paddle::onednn::dialect::QuantizeOp>()) {
        return true;
      }

      return false;
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();

    const auto &quantize_op = res.Op(
        "onednn_op.quantize",
        {{
            {"scale", res.Float32Attr(1.f)},
            {"shift", res.Float32Attr(0.0f)},
            {"bfloat16", res.BoolAttr(true)},
            {"is_negative_input", res.BoolAttr(false)},
            {"output_format",
             data_format ? pat.Attr("data_format") : res.StrAttr("NCHW")},
        }});
    quantize_op({&res.Tensor("quantize_" + std::to_string(index_))},
                {&res.Tensor("quantize_out_" + std::to_string(index_))});

    const auto &res_op = res.Op(bfloat16_ops_, op_attrs);
    if (index_ == 0) {
      res_op({&res.Tensor("quantize_out_0"),
              &res.Tensor("quantize_1"),
              &res.Tensor("quantize_2")},
             {&res.Tensor("out")});

    } else if (index_ == 1) {
      res_op({&res.Tensor("quantize_0"),
              &res.Tensor("quantize_out_1"),
              &res.Tensor("quantize_2")},
             {&res.Tensor("out")});

    } else if (index_ == 2) {
      res_op({&res.Tensor("quantize_0"),
              &res.Tensor("quantize_1"),
              &res.Tensor("quantize_out_2")},
             {&res.Tensor("out")});
    }
  }
};

class CpuBfloat16DequantPatternThree_one : public paddle::drr::DrrPatternBase {
 private:
  std::string bfloat16_ops_;
  uint32_t benefit_;
  uint32_t index_;

 public:
  CpuBfloat16DequantPatternThree_one(const std::string &bfloat16_ops,
                                     uint32_t benefit)
      : bfloat16_ops_(bfloat16_ops), benefit_(benefit), index_(0) {}

  std::string name() const override {
    return bfloat16_ops_ + "CpuBfloat16DequantPatternThree_one";
  }

  uint32_t benefit() const override { return benefit_; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    std::unordered_map<std::string, paddle::drr::Attribute> op_attrs;
    if (bfloat16_ops_ == "onednn_op.fc") {
      op_attrs.emplace("in_num_col_dims", pat.Attr("in_num_col_dims"));
      op_attrs.emplace("activation_type", pat.Attr("activation_type"));
      op_attrs.emplace("padding_weights", pat.Attr("padding_weights"));
      op_attrs.emplace("use_quantizer", pat.Attr("use_quantizer"));
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
      op_attrs.emplace("scale_in", pat.Attr("scale_in"));
      op_attrs.emplace("scale_weights", pat.Attr("scale_weights"));
      op_attrs.emplace("scale_out", pat.Attr("scale_out"));
      op_attrs.emplace("force_fp32_output", pat.Attr("force_fp32_output"));
      op_attrs.emplace("fuse_activation", pat.Attr("fuse_activation"));
      op_attrs.emplace("fuse_alpha", pat.Attr("fuse_alpha"));
      op_attrs.emplace("fuse_beta", pat.Attr("fuse_beta"));
      op_attrs.emplace("fused_output_scale", pat.Attr("fused_output_scale"));
      op_attrs.emplace("fused_reshape2_shape",
                       pat.Attr("fused_reshape2_shape"));
    } else if (bfloat16_ops_ == "onednn_op.slice") {
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
      op_attrs.emplace("decrease_axis", pat.Attr("decrease_axis"));
      op_attrs.emplace("infer_flags", pat.Attr("infer_flags"));
      op_attrs.emplace("axes", pat.Attr("axes"));

    } else if (bfloat16_ops_ == "onednn_op.split") {
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
    } else if (bfloat16_ops_ == "onednn_op.fused_matmul") {
      op_attrs.emplace("trans_x", pat.Attr("trans_x"));
      op_attrs.emplace("trans_y", pat.Attr("trans_y"));
      op_attrs.emplace("matmul_alpha", pat.Attr("matmul_alpha"));
      op_attrs.emplace("fused_output_scale", pat.Attr("fused_output_scale"));
      op_attrs.emplace("fused_reshape_x", pat.Attr("fused_reshape_x"));
      op_attrs.emplace("fused_transpose_x", pat.Attr("fused_transpose_x"));
      op_attrs.emplace("fused_reshape_y", pat.Attr("fused_reshape_y"));
      op_attrs.emplace("fused_transpose_y", pat.Attr("fused_transpose_y"));
      op_attrs.emplace("fused_reshape_out", pat.Attr("fused_reshape_out"));
      op_attrs.emplace("fused_transpose_out", pat.Attr("fused_transpose_out"));
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
      op_attrs.emplace("scale_x", pat.Attr("scale_x"));
      op_attrs.emplace("scale_y", pat.Attr("scale_y"));
      op_attrs.emplace("scale_in_eltwise", pat.Attr("scale_in_eltwise"));
      op_attrs.emplace("scale_out", pat.Attr("scale_out"));
      op_attrs.emplace("force_fp32_output", pat.Attr("force_fp32_output"));
      op_attrs.emplace("fuse_activation", pat.Attr("fuse_activation"));
      op_attrs.emplace("fuse_alpha", pat.Attr("fuse_alpha"));
      op_attrs.emplace("fuse_beta", pat.Attr("fuse_beta"));
    } else if (bfloat16_ops_ == "onednn_op.clip" ||
               bfloat16_ops_ == "onednn_op.clip_") {
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
    } else if (bfloat16_ops_ == "onednn_op.conv2d_transpose") {
      op_attrs.emplace("is_test", pat.Attr("is_test"));
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
      op_attrs.emplace("data_format", pat.Attr("data_format"));
      op_attrs.emplace("dilations", pat.Attr("dilations"));
      op_attrs.emplace("groups", pat.Attr("groups"));
      op_attrs.emplace("padding_algorithm", pat.Attr("padding_algorithm"));
      op_attrs.emplace("output_padding", pat.Attr("output_padding"));
      op_attrs.emplace("paddings", pat.Attr("paddings"));
      op_attrs.emplace("strides", pat.Attr("strides"));
      op_attrs.emplace("force_fp32_output", pat.Attr("force_fp32_output"));
    }

    const auto &op = pat.Op(bfloat16_ops_, op_attrs);
    op({&pat.Tensor("x"), &pat.Tensor("y"), &pat.Tensor("z")},
       {&pat.Tensor("out")});

    paddle::drr::ResultPattern res = pat.ResultPattern();

    pat.AddConstraint([this](const paddle::drr::MatchContext &match_ctx) {
      bool need_dequant = false;
      pir::Operation *input_op = match_ctx.Tensor("out").defining_op();

      auto mkldnn_data_type = match_ctx.Attr<std::string>("mkldnn_data_type");
      if (mkldnn_data_type != "bfloat16") {
        return false;
      }

      std::vector<std::string> no_permitted_output_names = {"xshape"};
      auto op_info =
          pir::IrContext::Instance()->GetRegisteredOpInfo(bfloat16_ops_);
      paddle::dialect::OpYamlInfoParser yaml_parser(
          op_info.GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>()
              ->get_op_info_(bfloat16_ops_),
          paddle::dialect::IsLegacyOp(bfloat16_ops_));

      auto &name2id = yaml_parser.OutputName2Id();
      auto find_item = std::find_if(
          name2id.begin(),
          name2id.end(),
          [this](const std::map<std::string, uint32_t>::value_type item) {
            return item.second == index_;
          });

      std::string output_name;
      if (find_item != name2id.end()) {
        output_name = (*find_item).first;
      }

      auto it = std::find(no_permitted_output_names.begin(),
                          no_permitted_output_names.end(),
                          output_name);
      if (it != no_permitted_output_names.end()) {
        return false;
      }

      auto next_ops = pir::GetUseOpsForOutput(input_op, 0);
      for (auto [next_op, _] : next_ops) {
        if (!next_op->isa<paddle::onednn::dialect::DequantizeOp>()) {
          need_dequant = true;
        }
      }
      return need_dequant;
    });

    const auto &res_op = res.Op(bfloat16_ops_, op_attrs);
    res_op({&res.Tensor("x"), &res.Tensor("y"), &res.Tensor("z")},
           {&res.Tensor("dequantize")});
    const auto &dequantize_op = res.Op("onednn_op.dequantize",
                                       {{
                                           {"scale", res.Float32Attr(1.f)},
                                           {"shift", res.Float32Attr(0.0f)},
                                       }});

    dequantize_op({&res.Tensor("dequantize")}, {&res.Tensor("out")});
  }
};

class CpuBfloat16FusionGruPattern : public paddle::drr::DrrPatternBase {
 private:
  std::string bfloat16_ops_;
  uint32_t benefit_;
  uint32_t index_;

 public:
  CpuBfloat16FusionGruPattern(const std::string &bfloat16_ops,
                              uint32_t benefit,
                              uint32_t index)
      : bfloat16_ops_(bfloat16_ops), benefit_(benefit), index_(index) {}

  std::string name() const override {
    return "CpuBfloat16FusionGruPattern" + std::to_string(index_);
  }

  uint32_t benefit() const override { return benefit_; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    std::unordered_map<std::string, paddle::drr::Attribute> op_attrs;
    op_attrs.emplace("scale_weights", pat.Attr("scale_weights"));
    op_attrs.emplace("shift_data", pat.Attr("shift_data"));
    op_attrs.emplace("scale_data", pat.Attr("scale_data"));
    op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
    op_attrs.emplace("force_fp32_output", pat.Attr("force_fp32_output"));
    op_attrs.emplace("origin_mode", pat.Attr("origin_mode"));
    op_attrs.emplace("use_seq", pat.Attr("use_seq"));
    op_attrs.emplace("is_reverse", pat.Attr("is_reverse"));
    op_attrs.emplace("gate_activation", pat.Attr("gate_activation"));
    op_attrs.emplace("activation", pat.Attr("activation"));

    const auto &op = pat.Op(bfloat16_ops_, op_attrs);
    op({&pat.Tensor("quantize_0"),
        &pat.Tensor("quantize_1"),
        &pat.Tensor("quantize_2"),
        &pat.Tensor("quantize_3"),
        &pat.Tensor("quantize_4")},
       {&pat.Tensor("out_0"),
        &pat.Tensor("out_1"),
        &pat.Tensor("out_2"),
        &pat.Tensor("out_3"),
        &pat.Tensor("out_4")});

    pat.AddConstraint([this](const paddle::drr::MatchContext &match_ctx) {
      auto mkldnn_data_type = match_ctx.Attr<std::string>("mkldnn_data_type");
      if (mkldnn_data_type != "bfloat16") {
        return false;
      }
      const std::vector<std::string> permitted_input_names = {
          "x", "y", "input", "residual_param"};
      auto op_info =
          pir::IrContext::Instance()->GetRegisteredOpInfo(bfloat16_ops_);
      paddle::dialect::OpYamlInfoParser yaml_parser(
          op_info.GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>()
              ->get_op_info_(bfloat16_ops_),
          paddle::dialect::IsLegacyOp(bfloat16_ops_));
      auto &name2id = yaml_parser.InputName2Id();
      auto find_item = std::find_if(
          name2id.begin(),
          name2id.end(),
          [this](const std::map<std::string, uint32_t>::value_type item) {
            return item.second == index_;
          });

      std::string input_name;
      if (find_item != name2id.end()) {
        input_name = (*find_item).first;
      }

      auto it = std::find(permitted_input_names.begin(),
                          permitted_input_names.end(),
                          input_name);
      if (it == permitted_input_names.end()) {
        return false;
      }

      pir::Operation *input_op = match_ctx.Tensor("out_0").defining_op();
      uint32_t num_operands = input_op->num_operands();
      if (index_ >= num_operands || !input_op->operand_source(index_)) {
        return false;
      }
      auto *pre_op = pir::GetDefiningOpForInput(input_op, index_);
      if (!pre_op) {
        return false;
      }
      if (!pre_op->isa<paddle::onednn::dialect::QuantizeOp>()) {
        return true;
      }

      return false;
    });
    paddle::drr::ResultPattern res = pat.ResultPattern();

    const auto &quantize_op =
        res.Op("onednn_op.quantize",
               {{
                   {"scale", res.Float32Attr(1.f)},
                   {"shift", res.Float32Attr(0.0f)},
                   {"bfloat16", res.BoolAttr(true)},
                   {"is_negative_input", res.BoolAttr(false)},
                   {"output_format", res.StrAttr("NCHW")},
               }});
    quantize_op({&res.Tensor("quantize_" + std::to_string(index_))},
                {&res.Tensor("quantize_out_" + std::to_string(index_))});

    const auto &res_op = res.Op(bfloat16_ops_, op_attrs);
    if (index_ == 0) {
      res_op({&res.Tensor("quantize_out_0"),
              &res.Tensor("quantize_1"),
              &res.Tensor("quantize_2"),
              &res.Tensor("quantize_3"),
              &res.Tensor("quantize_4")},
             {{&res.Tensor("out_0"),
               &res.Tensor("out_1"),
               &res.Tensor("out_2"),
               &res.Tensor("out_3"),
               &res.Tensor("out_4")}});

    } else if (index_ == 1) {
      res_op({&res.Tensor("quantize_0"),
              &res.Tensor("quantize_out_1"),
              &res.Tensor("quantize_2"),
              &res.Tensor("quantize_3"),
              &res.Tensor("quantize_4")},
             {{&res.Tensor("out_0"),
               &res.Tensor("out_1"),
               &res.Tensor("out_2"),
               &res.Tensor("out_3"),
               &res.Tensor("out_4")}});

    } else if (index_ == 2) {
      res_op({&res.Tensor("quantize_0"),
              &res.Tensor("quantize_1"),
              &res.Tensor("quantize_out_2"),
              &res.Tensor("quantize_3"),
              &res.Tensor("quantize_4")},
             {{&res.Tensor("out_0"),
               &res.Tensor("out_1"),
               &res.Tensor("out_2"),
               &res.Tensor("out_3"),
               &res.Tensor("out_4")}});

    } else if (index_ == 3) {
      res_op({&res.Tensor("quantize_0"),
              &res.Tensor("quantize_1"),
              &res.Tensor("quantize_2"),
              &res.Tensor("quantize_out_3"),
              &res.Tensor("quantize_4")},
             {{&res.Tensor("out_0"),
               &res.Tensor("out_1"),
               &res.Tensor("out_2"),
               &res.Tensor("out_3"),
               &res.Tensor("out_4")}});

    } else if (index_ == 4) {
      res_op({&res.Tensor("quantize_0"),
              &res.Tensor("quantize_1"),
              &res.Tensor("quantize_2"),
              &res.Tensor("quantize_3"),
              &res.Tensor("quantize_out_4")},
             {{&res.Tensor("out_0"),
               &res.Tensor("out_1"),
               &res.Tensor("out_2"),
               &res.Tensor("out_3"),
               &res.Tensor("out_4")}});
    }
  }
};

class CpuBfloat16FusionGruDequantPattern : public paddle::drr::DrrPatternBase {
 private:
  std::string bfloat16_ops_;
  uint32_t benefit_;
  uint32_t index_;

 public:
  CpuBfloat16FusionGruDequantPattern(const std::string &bfloat16_ops,
                                     uint32_t benefit,
                                     uint32_t index)
      : bfloat16_ops_(bfloat16_ops), benefit_(benefit), index_(index) {}

  std::string name() const override {
    return "CpuBfloat16FusionGruDequantPattern" + std::to_string(index_);
  }

  uint32_t benefit() const override { return benefit_; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    std::unordered_map<std::string, paddle::drr::Attribute> op_attrs;
    op_attrs.emplace("scale_weights", pat.Attr("scale_weights"));
    op_attrs.emplace("shift_data", pat.Attr("shift_data"));
    op_attrs.emplace("scale_data", pat.Attr("scale_data"));
    op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
    op_attrs.emplace("force_fp32_output", pat.Attr("force_fp32_output"));
    op_attrs.emplace("origin_mode", pat.Attr("origin_mode"));
    op_attrs.emplace("use_seq", pat.Attr("use_seq"));
    op_attrs.emplace("is_reverse", pat.Attr("is_reverse"));
    op_attrs.emplace("gate_activation", pat.Attr("gate_activation"));
    op_attrs.emplace("activation", pat.Attr("activation"));

    const auto &op = pat.Op(bfloat16_ops_, op_attrs);
    op({&pat.Tensor("quantize_0"),
        &pat.Tensor("quantize_1"),
        &pat.Tensor("quantize_2"),
        &pat.Tensor("quantize_3"),
        &pat.Tensor("quantize_4")},
       {&pat.Tensor("out_0"),
        &pat.Tensor("out_1"),
        &pat.Tensor("out_2"),
        &pat.Tensor("out_3"),
        &pat.Tensor("out_4")});

    paddle::drr::ResultPattern res = pat.ResultPattern();

    pat.AddConstraint([this](const paddle::drr::MatchContext &match_ctx) {
      bool need_dequant = false;
      pir::Operation *input_op = match_ctx.Tensor("out_0").defining_op();

      auto mkldnn_data_type = match_ctx.Attr<std::string>("mkldnn_data_type");
      if (mkldnn_data_type != "bfloat16") {
        return false;
      }

      std::vector<std::string> no_permitted_output_names = {"xshape"};
      auto op_info =
          pir::IrContext::Instance()->GetRegisteredOpInfo(bfloat16_ops_);
      paddle::dialect::OpYamlInfoParser yaml_parser(
          op_info.GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>()
              ->get_op_info_(bfloat16_ops_),
          paddle::dialect::IsLegacyOp(bfloat16_ops_));

      auto &name2id = yaml_parser.OutputName2Id();
      auto find_item = std::find_if(
          name2id.begin(),
          name2id.end(),
          [this](const std::map<std::string, uint32_t>::value_type item) {
            return item.second == index_;
          });

      std::string output_name;
      if (find_item != name2id.end()) {
        output_name = (*find_item).first;
      }

      auto it = std::find(no_permitted_output_names.begin(),
                          no_permitted_output_names.end(),
                          output_name);
      if (it != no_permitted_output_names.end()) {
        return false;
      }

      auto next_ops = pir::GetUseOpsForOutput(input_op, index_);
      for (auto [next_op, _] : next_ops) {
        if (!next_op->isa<paddle::onednn::dialect::DequantizeOp>()) {
          need_dequant = true;
        }
      }
      return need_dequant;
    });

    const auto &res_op = res.Op(bfloat16_ops_, op_attrs);
    if (index_ == 0) {
      res_op({&res.Tensor("quantize_0"),
              &res.Tensor("quantize_1"),
              &res.Tensor("quantize_2"),
              &res.Tensor("quantize_3"),
              &res.Tensor("quantize_4")},
             {&res.Tensor("dequantize_0"),
              &res.Tensor("out_1"),
              &res.Tensor("out_2"),
              &res.Tensor("out_3"),
              &res.Tensor("out_4")});
      const auto &dequantize_op = res.Op("onednn_op.dequantize",
                                         {{
                                             {"scale", res.Float32Attr(1.f)},
                                             {"shift", res.Float32Attr(0.0f)},
                                         }});

      dequantize_op({&res.Tensor("dequantize_0")}, {&res.Tensor("out_0")});
    } else if (index_ == 1) {
      res_op({&res.Tensor("quantize_0"),
              &res.Tensor("quantize_1"),
              &res.Tensor("quantize_2"),
              &res.Tensor("quantize_3"),
              &res.Tensor("quantize_4")},
             {&res.Tensor("out_0"),
              &res.Tensor("dequantize_1"),
              &res.Tensor("out_2"),
              &res.Tensor("out_3"),
              &res.Tensor("out_4")});
      const auto &dequantize_op = res.Op("onednn_op.dequantize",
                                         {{
                                             {"scale", res.Float32Attr(1.f)},
                                             {"shift", res.Float32Attr(0.0f)},
                                         }});

      dequantize_op({&res.Tensor("dequantize_1")}, {&res.Tensor("out_1")});

    } else if (index_ == 2) {
      res_op({&res.Tensor("quantize_0"),
              &res.Tensor("quantize_1"),
              &res.Tensor("quantize_2"),
              &res.Tensor("quantize_3"),
              &res.Tensor("quantize_4")},
             {&res.Tensor("out_0"),
              &res.Tensor("out_1"),
              &res.Tensor("dequantize_2"),
              &res.Tensor("out_3"),
              &res.Tensor("out_4")});
      const auto &dequantize_op = res.Op("onednn_op.dequantize",
                                         {{
                                             {"scale", res.Float32Attr(1.f)},
                                             {"shift", res.Float32Attr(0.0f)},
                                         }});

      dequantize_op({&res.Tensor("dequantize_2")}, {&res.Tensor("out_2")});

    } else if (index_ == 3) {
      res_op({&res.Tensor("quantize_0"),
              &res.Tensor("quantize_1"),
              &res.Tensor("quantize_2"),
              &res.Tensor("quantize_3"),
              &res.Tensor("quantize_4")},
             {&res.Tensor("out_0"),
              &res.Tensor("out_1"),
              &res.Tensor("out_2"),
              &res.Tensor("dequantize_3"),
              &res.Tensor("out_4")});
      const auto &dequantize_op = res.Op("onednn_op.dequantize",
                                         {{
                                             {"scale", res.Float32Attr(1.f)},
                                             {"shift", res.Float32Attr(0.0f)},
                                         }});

      dequantize_op({&res.Tensor("dequantize_3")}, {&res.Tensor("out_3")});

    } else if (index_ == 4) {
      res_op({&res.Tensor("quantize_0"),
              &res.Tensor("quantize_1"),
              &res.Tensor("quantize_2"),
              &res.Tensor("quantize_3"),
              &res.Tensor("quantize_4")},
             {&res.Tensor("out_0"),
              &res.Tensor("out_1"),
              &res.Tensor("out_2"),
              &res.Tensor("out_3"),
              &res.Tensor("dequantize_4")});
      const auto &dequantize_op = res.Op("onednn_op.dequantize",
                                         {{
                                             {"scale", res.Float32Attr(1.f)},
                                             {"shift", res.Float32Attr(0.0f)},
                                         }});

      dequantize_op({&res.Tensor("dequantize_4")}, {&res.Tensor("out_4")});
    }
  }
};

class CpuBfloat16LayerNormOpPattern : public paddle::drr::DrrPatternBase {
 private:
  std::string bfloat16_ops_;
  uint32_t benefit_;
  uint32_t index_;

 public:
  CpuBfloat16LayerNormOpPattern(const std::string &bfloat16_ops,
                                uint32_t benefit,
                                uint32_t index)
      : bfloat16_ops_(bfloat16_ops), benefit_(benefit), index_(index) {}

  std::string name() const override {
    return "CpuBfloat16LayerNormOpPattern" + std::to_string(index_);
  }

  uint32_t benefit() const override { return benefit_; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    std::unordered_map<std::string, paddle::drr::Attribute> op_attrs;
    op_attrs.emplace("is_test", pat.Attr("is_test"));
    op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
    op_attrs.emplace("begin_norm_axis", pat.Attr("begin_norm_axis"));
    op_attrs.emplace("epsilon", pat.Attr("epsilon"));

    const auto &op = pat.Op(bfloat16_ops_, op_attrs);
    op({&pat.Tensor("quantize_0"),
        &pat.Tensor("quantize_1"),
        &pat.Tensor("quantize_2")},
       {&pat.Tensor("out_0"), &pat.Tensor("out_1"), &pat.Tensor("out_2")});

    pat.AddConstraint([this](const paddle::drr::MatchContext &match_ctx) {
      auto mkldnn_data_type = match_ctx.Attr<std::string>("mkldnn_data_type");
      if (mkldnn_data_type != "bfloat16") {
        return false;
      }
      const std::vector<std::string> permitted_input_names = {
          "x", "y", "input", "residual_param"};
      auto op_info =
          pir::IrContext::Instance()->GetRegisteredOpInfo(bfloat16_ops_);
      paddle::dialect::OpYamlInfoParser yaml_parser(
          op_info.GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>()
              ->get_op_info_(bfloat16_ops_),
          paddle::dialect::IsLegacyOp(bfloat16_ops_));
      auto &name2id = yaml_parser.InputName2Id();
      auto find_item = std::find_if(
          name2id.begin(),
          name2id.end(),
          [this](const std::map<std::string, uint32_t>::value_type item) {
            return item.second == index_;
          });

      std::string input_name;
      if (find_item != name2id.end()) {
        input_name = (*find_item).first;
      }

      auto it = std::find(permitted_input_names.begin(),
                          permitted_input_names.end(),
                          input_name);
      if (it == permitted_input_names.end()) {
        return false;
      }

      pir::Operation *input_op = match_ctx.Tensor("out_0").defining_op();
      uint32_t num_operands = input_op->num_operands();
      if (index_ >= num_operands || !input_op->operand_source(index_)) {
        return false;
      }
      auto *pre_op = pir::GetDefiningOpForInput(input_op, index_);
      if (!pre_op) {
        return false;
      }
      if (!pre_op->isa<paddle::onednn::dialect::QuantizeOp>()) {
        return true;
      }

      return false;
    });
    // std::map<int, int> quantize_in_list;
    paddle::drr::ResultPattern res = pat.ResultPattern();

    const auto &quantize_op =
        res.Op("onednn_op.quantize",
               {{
                   {"scale", res.Float32Attr(1.f)},
                   {"shift", res.Float32Attr(0.0f)},
                   {"bfloat16", res.BoolAttr(true)},
                   {"is_negative_input", res.BoolAttr(false)},
                   {"output_format", res.StrAttr("NCHW")},
               }});
    quantize_op({&res.Tensor("quantize_" + std::to_string(index_))},
                {&res.Tensor("quantize_out_" + std::to_string(index_))});

    const auto &res_op = res.Op(bfloat16_ops_, op_attrs);
    if (index_ == 0) {
      res_op(
          {&res.Tensor("quantize_out_0"),
           &res.Tensor("quantize_1"),
           &res.Tensor("quantize_2")},
          {{&res.Tensor("out_0"), &res.Tensor("out_1"), &res.Tensor("out_2")}});

    } else if (index_ == 1) {
      res_op(
          {&res.Tensor("quantize_0"),
           &res.Tensor("quantize_out_1"),
           &res.Tensor("quantize_2")},
          {{&res.Tensor("out_0"), &res.Tensor("out_1"), &res.Tensor("out_2")}});

    } else if (index_ == 2) {
      res_op(
          {&res.Tensor("quantize_0"),
           &res.Tensor("quantize_1"),
           &res.Tensor("quantize_out_2")},
          {{&res.Tensor("out_0"), &res.Tensor("out_1"), &res.Tensor("out_2")}});
    }
  }
};

class CpuBfloat16LayerNormDequantPattern : public paddle::drr::DrrPatternBase {
 private:
  std::string bfloat16_ops_;
  uint32_t benefit_;
  uint32_t index_;

 public:
  CpuBfloat16LayerNormDequantPattern(const std::string &bfloat16_ops,
                                     uint32_t benefit,
                                     uint32_t index)
      : bfloat16_ops_(bfloat16_ops), benefit_(benefit), index_(index) {}

  std::string name() const override {
    return "CpuBfloat16LayerNormDequantPattern" + std::to_string(index_);
  }

  uint32_t benefit() const override { return benefit_; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    std::unordered_map<std::string, paddle::drr::Attribute> op_attrs;
    op_attrs.emplace("is_test", pat.Attr("is_test"));
    op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
    op_attrs.emplace("begin_norm_axis", pat.Attr("begin_norm_axis"));
    op_attrs.emplace("epsilon", pat.Attr("epsilon"));

    const auto &op = pat.Op(bfloat16_ops_, op_attrs);
    op({&pat.Tensor("quantize_0"),
        &pat.Tensor("quantize_1"),
        &pat.Tensor("quantize_2")},
       {&pat.Tensor("out_0"), &pat.Tensor("out_1"), &pat.Tensor("out_2")});

    paddle::drr::ResultPattern res = pat.ResultPattern();

    pat.AddConstraint([this](const paddle::drr::MatchContext &match_ctx) {
      bool need_dequant = false;
      pir::Operation *input_op = match_ctx.Tensor("out_0").defining_op();

      auto mkldnn_data_type = match_ctx.Attr<std::string>("mkldnn_data_type");
      if (mkldnn_data_type != "bfloat16") {
        return false;
      }

      std::vector<std::string> no_permitted_output_names = {"xshape"};
      if (bfloat16_ops_ == "onednn_op.layer_norm") {
        no_permitted_output_names.push_back("mean");
        no_permitted_output_names.push_back("variance");
      }
      auto op_info =
          pir::IrContext::Instance()->GetRegisteredOpInfo(bfloat16_ops_);
      paddle::dialect::OpYamlInfoParser yaml_parser(
          op_info.GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>()
              ->get_op_info_(bfloat16_ops_),
          paddle::dialect::IsLegacyOp(bfloat16_ops_));

      auto &name2id = yaml_parser.OutputName2Id();
      auto find_item = std::find_if(
          name2id.begin(),
          name2id.end(),
          [this](const std::map<std::string, uint32_t>::value_type item) {
            return item.second == index_;
          });

      std::string output_name;
      if (find_item != name2id.end()) {
        output_name = (*find_item).first;
      }

      auto it = std::find(no_permitted_output_names.begin(),
                          no_permitted_output_names.end(),
                          output_name);
      if (it != no_permitted_output_names.end()) {
        return false;
      }

      auto next_ops = pir::GetUseOpsForOutput(input_op, index_);
      for (auto [next_op, _] : next_ops) {
        if (!next_op->isa<paddle::onednn::dialect::DequantizeOp>()) {
          need_dequant = true;
        }
      }
      return need_dequant;
    });

    const auto &res_op = res.Op(bfloat16_ops_, op_attrs);
    if (index_ == 0) {
      res_op({&res.Tensor("quantize_0"),
              &res.Tensor("quantize_1"),
              &res.Tensor("quantize_2")},
             {&res.Tensor("dequantize_0"),
              &res.Tensor("out_1"),
              &res.Tensor("out_2")});
      const auto &dequantize_op = res.Op("onednn_op.dequantize",
                                         {{
                                             {"scale", res.Float32Attr(1.f)},
                                             {"shift", res.Float32Attr(0.0f)},
                                         }});

      dequantize_op({&res.Tensor("dequantize_0")}, {&res.Tensor("out_0")});
    } else if (index_ == 1) {
      res_op({&res.Tensor("quantize_0"),
              &res.Tensor("quantize_1"),
              &res.Tensor("quantize_2")},
             {&res.Tensor("out_0"),
              &res.Tensor("dequantize_1"),
              &res.Tensor("out_2")});
      const auto &dequantize_op = res.Op("onednn_op.dequantize",
                                         {{
                                             {"scale", res.Float32Attr(1.f)},
                                             {"shift", res.Float32Attr(0.0f)},
                                         }});

      dequantize_op({&res.Tensor("dequantize_1")}, {&res.Tensor("out_1")});

    } else if (index_ == 2) {
      res_op({&res.Tensor("quantize_0"),
              &res.Tensor("quantize_1"),
              &res.Tensor("quantize_2")},
             {&res.Tensor("out_0"),
              &res.Tensor("out_1"),
              &res.Tensor("dequantize_2")});
      const auto &dequantize_op = res.Op("onednn_op.dequantize",
                                         {{
                                             {"scale", res.Float32Attr(1.f)},
                                             {"shift", res.Float32Attr(0.0f)},
                                         }});

      dequantize_op({&res.Tensor("dequantize_2")}, {&res.Tensor("out_2")});
    }
  }
};

class CpuBfloat16PatternFour_one : public paddle::drr::DrrPatternBase {
 private:
  std::string bfloat16_ops_;
  uint32_t benefit_;
  uint32_t index_;

 public:
  CpuBfloat16PatternFour_one(const std::string &bfloat16_ops,
                             uint32_t benefit,
                             uint32_t index)
      : bfloat16_ops_(bfloat16_ops), benefit_(benefit), index_(index) {}

  std::string name() const override {
    return bfloat16_ops_ + "CpuBfloat16PatternFour_one";
  }

  uint32_t benefit() const override { return benefit_; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    std::unordered_map<std::string, paddle::drr::Attribute> op_attrs;
    if (bfloat16_ops_ == "onednn_op.bilinear_interp") {
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
      op_attrs.emplace("align_mode", pat.Attr("align_mode"));
      op_attrs.emplace("align_corners", pat.Attr("align_corners"));
      op_attrs.emplace("interp_method", pat.Attr("interp_method"));
      op_attrs.emplace("scale", pat.Attr("scale"));
      op_attrs.emplace("out_w", pat.Attr("out_w"));
      op_attrs.emplace("out_h", pat.Attr("out_h"));
      op_attrs.emplace("out_d", pat.Attr("out_d"));
      op_attrs.emplace("data_format", pat.Attr("data_format"));
    } else if (bfloat16_ops_ == "onednn_op.fused_conv2d") {
      op_attrs.emplace("strides", pat.Attr("strides"));
      op_attrs.emplace("paddings", pat.Attr("paddings"));
      op_attrs.emplace("padding_algorithm", pat.Attr("padding_algorithm"));
      op_attrs.emplace("dilations", pat.Attr("dilations"));
      op_attrs.emplace("groups", pat.Attr("groups"));
      op_attrs.emplace("data_format", pat.Attr("data_format"));
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
      op_attrs.emplace("fuse_activation", pat.Attr("fuse_activation"));
      op_attrs.emplace("fuse_residual_connection",
                       pat.Attr("fuse_residual_connection"));
      op_attrs.emplace("force_fp32_output", pat.Attr("force_fp32_output"));
      op_attrs.emplace("fuse_alpha", pat.Attr("fuse_alpha"));
      op_attrs.emplace("fuse_beta", pat.Attr("fuse_beta"));
      op_attrs.emplace("scale_in", pat.Attr("scale_in"));
      op_attrs.emplace("scale_out", pat.Attr("scale_out"));
      op_attrs.emplace("scale_in_eltwise", pat.Attr("scale_in_eltwise"));
      op_attrs.emplace("scale_weights", pat.Attr("scale_weights"));
    } else if (bfloat16_ops_ == "onednn_op.conv2d_transpose_bias") {
      op_attrs.emplace("fuse_beta", pat.Attr("fuse_beta"));
      op_attrs.emplace("fuse_alpha", pat.Attr("fuse_alpha"));
      op_attrs.emplace("fuse_activation", pat.Attr("fuse_activation"));
      op_attrs.emplace("fuse_relu", pat.Attr("fuse_relu"));
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
      op_attrs.emplace("force_fp32_output", pat.Attr("force_fp32_output"));
      op_attrs.emplace("is_test", pat.Attr("is_test"));
      op_attrs.emplace("data_format", pat.Attr("data_format"));
      op_attrs.emplace("dilations", pat.Attr("dilations"));
      op_attrs.emplace("groups", pat.Attr("groups"));
      op_attrs.emplace("padding_algorithm", pat.Attr("padding_algorithm"));
      op_attrs.emplace("output_padding", pat.Attr("output_padding"));
      op_attrs.emplace("paddings", pat.Attr("paddings"));
      op_attrs.emplace("strides", pat.Attr("strides"));
    }

    const auto &op = pat.Op(bfloat16_ops_, op_attrs);
    op({&pat.Tensor("quantize_0"),
        &pat.Tensor("quantize_1"),
        &pat.Tensor("quantize_2"),
        &pat.Tensor("quantize_3")},
       {&pat.Tensor("out")});

    pat.AddConstraint([this](const paddle::drr::MatchContext &match_ctx) {
      auto mkldnn_data_type = match_ctx.Attr<std::string>("mkldnn_data_type");
      if (mkldnn_data_type != "bfloat16") {
        return false;
      }
      const std::vector<std::string> permitted_input_names = {
          "x", "y", "input", "residual_param"};
      auto op_info =
          pir::IrContext::Instance()->GetRegisteredOpInfo(bfloat16_ops_);
      paddle::dialect::OpYamlInfoParser yaml_parser(
          op_info.GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>()
              ->get_op_info_(bfloat16_ops_),
          paddle::dialect::IsLegacyOp(bfloat16_ops_));
      auto &name2id = yaml_parser.InputName2Id();
      auto find_item = std::find_if(
          name2id.begin(),
          name2id.end(),
          [this](const std::map<std::string, uint32_t>::value_type item) {
            return item.second == index_;
          });

      std::string input_name;
      if (find_item != name2id.end()) {
        input_name = (*find_item).first;
      }

      auto it = std::find(permitted_input_names.begin(),
                          permitted_input_names.end(),
                          input_name);
      if (it == permitted_input_names.end()) {
        return false;
      }

      pir::Operation *input_op = match_ctx.Tensor("out").defining_op();
      uint32_t num_operands = input_op->num_operands();
      if (index_ >= num_operands || !input_op->operand_source(index_)) {
        return false;
      }
      auto *pre_op = pir::GetDefiningOpForInput(input_op, index_);
      if (!pre_op) {
        return false;
      }
      if (!pre_op->isa<paddle::onednn::dialect::QuantizeOp>()) {
        return true;
      }

      return false;
    });
    // std::map<int, int> quantize_in_list;
    paddle::drr::ResultPattern res = pat.ResultPattern();

    const auto &quantize_op =
        res.Op("onednn_op.quantize",
               {{
                   {"scale", res.Float32Attr(1.f)},
                   {"shift", res.Float32Attr(0.0f)},
                   {"bfloat16", res.BoolAttr(true)},
                   {"is_negative_input", res.BoolAttr(false)},
                   {"output_format", pat.Attr("data_format")},
               }});
    quantize_op({&res.Tensor("quantize_" + std::to_string(index_))},
                {&res.Tensor("quantize_out_" + std::to_string(index_))});

    const auto &res_op = res.Op(bfloat16_ops_, op_attrs);
    if (index_ == 0) {
      res_op({&res.Tensor("quantize_out_0"),
              &res.Tensor("quantize_1"),
              &res.Tensor("quantize_2"),
              &res.Tensor("quantize_3")},
             {&res.Tensor("out")});

    } else if (index_ == 1) {
      res_op({&res.Tensor("quantize_0"),
              &res.Tensor("quantize_out_1"),
              &res.Tensor("quantize_2"),
              &res.Tensor("quantize_3")},
             {&res.Tensor("out")});

    } else if (index_ == 2) {
      res_op({&res.Tensor("quantize_0"),
              &res.Tensor("quantize_1"),
              &res.Tensor("quantize_out_2"),
              &res.Tensor("quantize_3")},
             {&res.Tensor("out")});

    } else if (index_ == 3) {
      res_op({&res.Tensor("quantize_0"),
              &res.Tensor("quantize_1"),
              &res.Tensor("quantize_2"),
              &res.Tensor("quantize_out_3")},
             {&res.Tensor("out")});
    }
  }
};

class CpuBfloat16DequantPatternFour_one : public paddle::drr::DrrPatternBase {
 private:
  std::string bfloat16_ops_;
  uint32_t benefit_;
  uint32_t index_;

 public:
  CpuBfloat16DequantPatternFour_one(const std::string &bfloat16_ops,
                                    uint32_t benefit)
      : bfloat16_ops_(bfloat16_ops), benefit_(benefit), index_(0) {}

  std::string name() const override {
    return bfloat16_ops_ + "CpuBfloat16DequantPatternFour_one";
  }

  uint32_t benefit() const override { return benefit_; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    std::unordered_map<std::string, paddle::drr::Attribute> op_attrs;
    if (bfloat16_ops_ == "onednn_op.bilinear_interp") {
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
      op_attrs.emplace("align_mode", pat.Attr("align_mode"));
      op_attrs.emplace("align_corners", pat.Attr("align_corners"));
      op_attrs.emplace("interp_method", pat.Attr("interp_method"));
      op_attrs.emplace("scale", pat.Attr("scale"));
      op_attrs.emplace("out_w", pat.Attr("out_w"));
      op_attrs.emplace("out_h", pat.Attr("out_h"));
      op_attrs.emplace("out_d", pat.Attr("out_d"));
      op_attrs.emplace("data_format", pat.Attr("data_format"));
    } else if (bfloat16_ops_ == "onednn_op.fused_conv2d") {
      op_attrs.emplace("strides", pat.Attr("strides"));
      op_attrs.emplace("paddings", pat.Attr("paddings"));
      op_attrs.emplace("padding_algorithm", pat.Attr("padding_algorithm"));
      op_attrs.emplace("dilations", pat.Attr("dilations"));
      op_attrs.emplace("groups", pat.Attr("groups"));
      op_attrs.emplace("data_format", pat.Attr("data_format"));
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
      op_attrs.emplace("fuse_activation", pat.Attr("fuse_activation"));
      op_attrs.emplace("fuse_residual_connection",
                       pat.Attr("fuse_residual_connection"));
      op_attrs.emplace("force_fp32_output", pat.Attr("force_fp32_output"));
      op_attrs.emplace("fuse_alpha", pat.Attr("fuse_alpha"));
      op_attrs.emplace("fuse_beta", pat.Attr("fuse_beta"));
      op_attrs.emplace("scale_in", pat.Attr("scale_in"));
      op_attrs.emplace("scale_out", pat.Attr("scale_out"));
      op_attrs.emplace("scale_in_eltwise", pat.Attr("scale_in_eltwise"));
      op_attrs.emplace("scale_weights", pat.Attr("scale_weights"));
    } else if (bfloat16_ops_ == "onednn_op.conv2d_transpose_bias") {
      op_attrs.emplace("fuse_beta", pat.Attr("fuse_beta"));
      op_attrs.emplace("fuse_alpha", pat.Attr("fuse_alpha"));
      op_attrs.emplace("fuse_activation", pat.Attr("fuse_activation"));
      op_attrs.emplace("fuse_relu", pat.Attr("fuse_relu"));
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
      op_attrs.emplace("force_fp32_output", pat.Attr("force_fp32_output"));
      op_attrs.emplace("is_test", pat.Attr("is_test"));
      op_attrs.emplace("data_format", pat.Attr("data_format"));
      op_attrs.emplace("dilations", pat.Attr("dilations"));
      op_attrs.emplace("groups", pat.Attr("groups"));
      op_attrs.emplace("padding_algorithm", pat.Attr("padding_algorithm"));
      op_attrs.emplace("output_padding", pat.Attr("output_padding"));
      op_attrs.emplace("paddings", pat.Attr("paddings"));
      op_attrs.emplace("strides", pat.Attr("strides"));
    }

    const auto &op = pat.Op(bfloat16_ops_, op_attrs);
    op({&pat.Tensor("x"), &pat.Tensor("y"), &pat.Tensor("z"), &pat.Tensor("s")},
       {&pat.Tensor("out")});

    paddle::drr::ResultPattern res = pat.ResultPattern();

    pat.AddConstraint([this](const paddle::drr::MatchContext &match_ctx) {
      bool need_dequant = false;
      pir::Operation *input_op = match_ctx.Tensor("out").defining_op();

      auto mkldnn_data_type = match_ctx.Attr<std::string>("mkldnn_data_type");
      if (mkldnn_data_type != "bfloat16") {
        return false;
      }

      std::vector<std::string> no_permitted_output_names = {"xshape"};
      auto op_info =
          pir::IrContext::Instance()->GetRegisteredOpInfo(bfloat16_ops_);
      paddle::dialect::OpYamlInfoParser yaml_parser(
          op_info.GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>()
              ->get_op_info_(bfloat16_ops_),
          paddle::dialect::IsLegacyOp(bfloat16_ops_));

      auto &name2id = yaml_parser.OutputName2Id();
      auto find_item = std::find_if(
          name2id.begin(),
          name2id.end(),
          [this](const std::map<std::string, uint32_t>::value_type item) {
            return item.second == index_;
          });

      std::string output_name;
      if (find_item != name2id.end()) {
        output_name = (*find_item).first;
      }

      auto it = std::find(no_permitted_output_names.begin(),
                          no_permitted_output_names.end(),
                          output_name);
      if (it != no_permitted_output_names.end()) {
        return false;
      }

      auto next_ops = pir::GetUseOpsForOutput(input_op, 0);
      for (auto [next_op, _] : next_ops) {
        if (!next_op->isa<paddle::onednn::dialect::DequantizeOp>()) {
          need_dequant = true;
        }
      }
      return need_dequant;
    });

    const auto &res_op = res.Op(bfloat16_ops_, op_attrs);
    res_op({&res.Tensor("x"),
            &res.Tensor("y"),
            &res.Tensor("z"),
            &res.Tensor("s")},
           {&res.Tensor("dequantize")});
    const auto &dequantize_op = res.Op("onednn_op.dequantize",
                                       {{
                                           {"scale", res.Float32Attr(1.f)},
                                           {"shift", res.Float32Attr(0.0f)},
                                       }});

    dequantize_op({&res.Tensor("dequantize")}, {&res.Tensor("out")});
  }
};

template <typename OpType>
class CastBf16Pattern : public pir::OpRewritePattern<OpType> {
 public:
  using pir::OpRewritePattern<OpType>::OpRewritePattern;

  bool MatchAndRewrite(
      OpType op,
      pir::PatternRewriter &rewriter) const override {  // NOLINT
    std::string target_op_name = op->name();
    if (!(target_op_name == "onednn_op.cast" ||
          target_op_name == "onednn_op.cast_"))
      return false;

    auto attributes = op->attributes();
    auto dtype_attr = attributes["dtype"];
    phi::DataType dtype =
        dtype_attr.template dyn_cast<paddle::dialect::DataTypeAttribute>()
            .data();
    if (dtype == phi::DataType::FLOAT32) {
      pir::Attribute new_dtype = paddle::dialect::DataTypeAttribute::get(
          rewriter.ir_context(), phi::DataType::BFLOAT16);
      attributes["dtype"] = new_dtype;
    } else {
      return false;
    }

    OpType new_cast = rewriter.Build<OpType>(op->operand_source(0), attributes);

    std::unordered_map<std::string, pir::Attribute> dq_attributes;
    dq_attributes["scale"] = rewriter.float_attr(1.0f);
    dq_attributes["shift"] = rewriter.float_attr(0.0f);
    paddle::onednn::dialect::DequantizeOp dq_op =
        rewriter.Build<paddle::onednn::dialect::DequantizeOp>(new_cast.out(),
                                                              dq_attributes);

    rewriter.ReplaceAllUsesWith(op->result(0), dq_op.output());
    rewriter.EraseOp(op);
    return true;
  }
};

class CpuBfloat16Pass : public pir::PatternRewritePass {
 public:
  CpuBfloat16Pass() : pir::PatternRewritePass("cpu_bfloat16_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);

    // op with two inputs and one output
    const std::vector<std::string> bfloat16_ops_two_one{
        paddle::onednn::dialect::Conv2dOp::name(),
        paddle::onednn::dialect::MatmulOp::name(),
        paddle::onednn::dialect::Pool2dOp::name(),
        paddle::onednn::dialect::PreluOp::name(),
        paddle::onednn::dialect::SumOp::name(),
        paddle::onednn::dialect::Reshape_Op::name(),
        paddle::onednn::dialect::ReshapeOp::name(),
        paddle::onednn::dialect::ScaleOp::name(),
        paddle::onednn::dialect::Scale_Op::name(),
        paddle::onednn::dialect::AddOp::name(),
        paddle::onednn::dialect::Add_Op::name(),
        paddle::onednn::dialect::MultiplyOp::name(),
        paddle::onednn::dialect::Multiply_Op::name(),
        paddle::onednn::dialect::SqueezeOp::name(),
        paddle::onednn::dialect::Squeeze_Op::name(),
    };

    // op with one inputs and one output
    const std::vector<std::string> bfloat16_ops_one_one{
        paddle::onednn::dialect::GeluOp::name(),
        paddle::onednn::dialect::SoftmaxOp::name(),
        paddle::onednn::dialect::Softmax_Op::name(),
        paddle::onednn::dialect::TransposeOp::name(),
        paddle::onednn::dialect::Transpose_Op::name(),
        paddle::onednn::dialect::ReluOp::name(),
        paddle::onednn::dialect::Relu_Op::name(),
        paddle::onednn::dialect::SigmoidOp::name(),
        paddle::onednn::dialect::Sigmoid_Op::name(),
    };

    // op with three inputs and one outputs
    const std::vector<std::string> bfloat16_ops_three_one{
        paddle::onednn::dialect::FcOp::name(),
        paddle::onednn::dialect::SliceOp::name(),
        paddle::onednn::dialect::FusedMatmulOp::name(),
        paddle::onednn::dialect::ClipOp::name(),
        paddle::onednn::dialect::Clip_Op::name(),
        paddle::onednn::dialect::Conv2dTransposeOp::name(),
    };

    const std::vector<std::string> bfloat16_ops_four_one{
        paddle::onednn::dialect::BilinearInterpOp::name(),
        paddle::onednn::dialect::Conv2dTransposeBiasOp::name(),
        paddle::onednn::dialect::FusedConv2dOp::name()};

    // op with two inputs and one output
    int benefit_idx = 1;
    for (auto op : bfloat16_ops_two_one) {
      ps.Add(
          paddle::drr::Create<CpuBfloat16Pattern>(context, op, benefit_idx, 0));
      benefit_idx++;
    }
    for (auto op : bfloat16_ops_two_one) {
      ps.Add(
          paddle::drr::Create<CpuBfloat16Pattern>(context, op, benefit_idx, 1));
      benefit_idx++;
    }

    benefit_idx = 1;
    for (auto op : bfloat16_ops_two_one) {
      ps.Add(paddle::drr::Create<CpuBfloat16DequantPattern>(
          context, op, benefit_idx));
      benefit_idx++;
    }

    // op with one inputs and one output
    benefit_idx = 1;
    for (auto op : bfloat16_ops_one_one) {
      ps.Add(paddle::drr::Create<CpuBfloat16PatternOne_one>(
          context, op, benefit_idx));
      benefit_idx++;
    }

    benefit_idx = 1;
    for (auto op : bfloat16_ops_one_one) {
      ps.Add(paddle::drr::Create<CpuBfloat16DequantPatternOne_one>(
          context, op, benefit_idx));
      benefit_idx++;
    }

    // op with three inputs and one output
    benefit_idx = 1;
    for (int i = 0; i < 3; i++) {
      for (auto op : bfloat16_ops_three_one) {
        ps.Add(paddle::drr::Create<CpuBfloat16PatternThree_one>(
            context, op, benefit_idx, i));
        benefit_idx++;
      }
    }

    benefit_idx = 1;
    for (auto op : bfloat16_ops_three_one) {
      ps.Add(paddle::drr::Create<CpuBfloat16DequantPatternThree_one>(
          context, op, benefit_idx));
      benefit_idx++;
    }

    // FusionGruOp: 5 in, 5 out
    for (int i = 0; i < 5; i++) {
      ps.Add(paddle::drr::Create<CpuBfloat16FusionGruPattern>(
          context,
          paddle::onednn::dialect::FusionGruOp::name(),
          i + 1 /*benefit*/,
          i));
    }
    for (int i = 0; i < 5; i++) {
      ps.Add(paddle::drr::Create<CpuBfloat16FusionGruDequantPattern>(
          context,
          paddle::onednn::dialect::FusionGruOp::name(),
          i + 1 /*benefit*/,
          i));
    }

    // LayerNormOp: 3 in, 3 out
    for (int i = 0; i < 3; i++) {
      ps.Add(paddle::drr::Create<CpuBfloat16LayerNormOpPattern>(
          context,
          paddle::onednn::dialect::LayerNormOp::name(),
          i + 1 /*benefit*/,
          i));
    }
    for (int i = 0; i < 3; i++) {
      ps.Add(paddle::drr::Create<CpuBfloat16LayerNormDequantPattern>(
          context, paddle::onednn::dialect::LayerNormOp::name(), i + 1, i));
    }

    benefit_idx = 1;
    for (int i = 0; i < 4; i++) {
      for (auto op : bfloat16_ops_four_one) {
        ps.Add(paddle::drr::Create<CpuBfloat16PatternFour_one>(
            context, op, benefit_idx, i));
        benefit_idx++;
      }
    }

    ps.Add(paddle::drr::Create<CpuBfloat16DequantPatternFour_one>(
        context, paddle::onednn::dialect::BilinearInterpOp::name(), 1));

    ps.Add(paddle::drr::Create<CpuBfloat16DequantPatternFour_one>(
        context, paddle::onednn::dialect::Conv2dTransposeBiasOp::name(), 2));

    ps.Add(paddle::drr::Create<CpuBfloat16DequantPatternFour_one>(
        context, paddle::onednn::dialect::FusedConv2dOp::name(), 3));

    auto cast_bf16_pattern =
        std::make_unique<CastBf16Pattern<paddle::onednn::dialect::CastOp>>(
            context,
            benefit_idx++,
            std::vector<std::string>{
                paddle::onednn::dialect::DequantizeOp::name(),
            });
    ps.Add(std::move(cast_bf16_pattern));

    auto cast_bf16_pattern_2 =
        std::make_unique<CastBf16Pattern<paddle::onednn::dialect::Cast_Op>>(
            context,
            benefit_idx++,
            std::vector<std::string>{
                paddle::onednn::dialect::DequantizeOp::name(),
            });
    ps.Add(std::move(cast_bf16_pattern_2));

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
