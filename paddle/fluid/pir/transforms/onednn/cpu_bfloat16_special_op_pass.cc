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

#include "paddle/fluid/pir/transforms/onednn/cpu_bfloat16_special_op_pass.h"

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

#include <stdio.h>
#include <string.h>
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"
#include "paddle/pir/include/pattern_rewrite/pattern_match.h"

namespace {
class CpuBfloat16InplacePattern : public paddle::drr::DrrPatternBase {
 private:
  std::string bfloat16_ops_;
  uint32_t benefit_;
  uint32_t index_;

 public:
  CpuBfloat16InplacePattern(const std::string &bfloat16_ops,
                            uint32_t benefit,
                            uint32_t index)
      : bfloat16_ops_(bfloat16_ops), benefit_(benefit), index_(index) {}

  std::string name() const override {
    return bfloat16_ops_ + "CpuBfloat16InplacePattern";
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

    } else if (bfloat16_ops_ == "onednn_op.multiply" ||
               bfloat16_ops_ == "onednn_op.multiply_") {
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
    } else if (bfloat16_ops_ == "onednn_op.scale" ||
               bfloat16_ops_ == "onednn_op.scale_") {
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
      op_attrs.emplace("bias", pat.Attr("bias"));
      op_attrs.emplace("bias_after_scale", pat.Attr("bias_after_scale"));
    }

    const auto &op = pat.Op(bfloat16_ops_, op_attrs);
    op({&pat.Tensor("quantize_0"), &pat.Tensor("quantize_1")},
       {&pat.Tensor("out")});

    const std::vector<std::string> permitted_input_names = {
        "x", "y", "input", "residual_param"};

    pat.AddConstraint([this](const paddle::drr::MatchContext &match_ctx) {
      auto mkldnn_data_type = match_ctx.Attr<std::string>("mkldnn_data_type");
      bool is_bfloat16 = true;
      if (mkldnn_data_type != "bfloat16") {
        // return false;
        is_bfloat16 = false;
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
      bool need_quantized = false;
      if (!pre_op->isa<paddle::onednn::dialect::QuantizeOp>()) {
        // return true;
        need_quantized = true;
      }
      bool pre_quantized = false;
      for (size_t i = 0; i < input_op->num_operands(); ++i) {
        if (i != index_) {
          auto *pre_i_op = pir::GetDefiningOpForInput(input_op, i);
          if (!pre_i_op) {
            continue;
          }
          if (pre_i_op->isa<paddle::onednn::dialect::QuantizeOp>()) {
            pre_quantized = true;
          }
        }
      }
      auto next_ops = pir::GetUseOpsForOutput(input_op, 0);
      for (auto [next_op, _] : next_ops) {
        if (next_op->isa<paddle::onednn::dialect::DequantizeOp>()) {
          pre_quantized = true;
        }
      }
      /*
        The reason to do this is some of ops, like add(add_), rele(relu_)...
        the mkldnn_data_type will auto convert to fp32, and can not
        find reason, maybe related Inplace op in PIR.
      */
      if (is_bfloat16) {
        if (need_quantized) {
          return true;
        } else {
          return false;
        }
      } else {
        if (pre_quantized && need_quantized) {
          return true;
        }
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
             {&res.Tensor("out")});

    } else if (index_ == 1) {
      res_op({&res.Tensor("quantize_0"), &res.Tensor("quantize_out_1")},
             {&res.Tensor("out")});
    }
  }
};

class CpuBfloat16DequantInplacePattern : public paddle::drr::DrrPatternBase {
 private:
  std::string bfloat16_ops_;
  uint32_t benefit_;
  uint32_t index_;

 public:
  CpuBfloat16DequantInplacePattern(const std::string &bfloat16_ops,
                                   uint32_t benefit)
      : bfloat16_ops_(bfloat16_ops), benefit_(benefit), index_(0) {}

  std::string name() const override {
    return bfloat16_ops_ + "CpuBfloat16DequantInplacePattern";
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
    } else if (bfloat16_ops_ == "onednn_op.multiply" ||
               bfloat16_ops_ == "onednn_op.multiply_") {
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
    } else if (bfloat16_ops_ == "onednn_op.scale" ||
               bfloat16_ops_ == "onednn_op.scale_") {
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
      op_attrs.emplace("bias", pat.Attr("bias"));
      op_attrs.emplace("bias_after_scale", pat.Attr("bias_after_scale"));
    }

    const auto &op = pat.Op(bfloat16_ops_, op_attrs);
    op({&pat.Tensor("x"), &pat.Tensor("y")}, {&pat.Tensor("out")});

    paddle::drr::ResultPattern res = pat.ResultPattern();

    pat.AddConstraint([this](const paddle::drr::MatchContext &match_ctx) {
      pir::Operation *input_op = match_ctx.Tensor("out").defining_op();

      auto mkldnn_data_type = match_ctx.Attr<std::string>("mkldnn_data_type");
      bool is_bfloat16 = true;
      if (mkldnn_data_type != "bfloat16") {
        // return false;
        is_bfloat16 = false;
      }

      std::vector<std::string> nopermitted_output_names = {"xshape"};
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

      auto it = std::find(nopermitted_output_names.begin(),
                          nopermitted_output_names.end(),
                          output_name);
      if (it != nopermitted_output_names.end()) {
        return false;
      }

      bool need_quantized = false;
      auto next_ops = pir::GetUseOpsForOutput(input_op, 0);
      for (auto [next_op, _] : next_ops) {
        if (!next_op->isa<paddle::onednn::dialect::DequantizeOp>()) {
          need_quantized = true;
        }
      }
      bool pre_quantized = false;
      for (size_t i = 0; i < input_op->num_operands(); ++i) {
        auto *pre_i_op = pir::GetDefiningOpForInput(input_op, i);
        if (!pre_i_op) {
          continue;
        }
        if (pre_i_op->isa<paddle::onednn::dialect::QuantizeOp>()) {
          pre_quantized = true;
        }
      }
      if (is_bfloat16) {
        if (need_quantized) {
          return true;
        } else {
          return false;
        }
      } else {
        if (pre_quantized && need_quantized) {
          return true;
        }
      }

      return false;
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

class CpuBfloat16InplacePattern1_1 : public paddle::drr::DrrPatternBase {
 private:
  std::string bfloat16_ops_;
  uint32_t benefit_;
  uint32_t index_;

 public:
  CpuBfloat16InplacePattern1_1(const std::string &bfloat16_ops,
                               uint32_t benefit)
      : bfloat16_ops_(bfloat16_ops), benefit_(benefit), index_(0) {}

  std::string name() const override {
    return bfloat16_ops_ + "CpuBfloat16InplacePattern1_1";
  }

  uint32_t benefit() const override { return benefit_; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    std::unordered_map<std::string, paddle::drr::Attribute> op_attrs;
    if (bfloat16_ops_ == "onednn_op.cast" ||
        bfloat16_ops_ == "onednn_op.cast_") {
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
      op_attrs.emplace("dtype", pat.Attr("dtype"));

    } else if (bfloat16_ops_ == "onednn_op.relu" ||
               bfloat16_ops_ == "onednn_op.relu_") {
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));

    } else if (bfloat16_ops_ == "onednn_op.sigmoid" ||
               bfloat16_ops_ == "onednn_op.sigmoid_") {
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
    }

    const auto &op = pat.Op(bfloat16_ops_, op_attrs);
    op({&pat.Tensor("quantize_0")}, {&pat.Tensor("out")});

    const std::vector<std::string> permitted_input_names = {
        "x", "y", "input", "residual_param"};

    pat.AddConstraint([this](const paddle::drr::MatchContext &match_ctx) {
      auto mkldnn_data_type = match_ctx.Attr<std::string>("mkldnn_data_type");
      bool is_bfloat16 = true;
      if (mkldnn_data_type != "bfloat16") {
        // return false;
        is_bfloat16 = false;
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
      bool need_quantized = false;
      if (!pre_op->isa<paddle::onednn::dialect::QuantizeOp>()) {
        // return true;
        need_quantized = true;
      }
      bool pre_quantized = false;
      auto next_ops = pir::GetUseOpsForOutput(input_op, 0);
      for (auto [next_op, _] : next_ops) {
        if (next_op->isa<paddle::onednn::dialect::DequantizeOp>()) {
          pre_quantized = true;
        }
      }
      /*
        The reason to do this is some of ops, like add(add_), rele(relu_)...
        the mkldnn_data_type will auto convert to fp32, and can not
        find reason, maybe related Inplace op in PIR.
      */
      if (is_bfloat16) {
        if (need_quantized) {
          return true;
        } else {
          return false;
        }
      } else {
        if (pre_quantized && need_quantized) {
          return true;
        }
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
    quantize_op({&res.Tensor("quantize_0")}, {&res.Tensor("quantize_out_0")});

    const auto &res_op =
        bfloat16_ops_ == "onednn_op.relu"
            ? res.Op(bfloat16_ops_,
                     {{"mkldnn_data_type", res.StrAttr("bfloat16")}})
            : res.Op(bfloat16_ops_, op_attrs);
    res_op({&res.Tensor("quantize_out_0")}, {&res.Tensor("out")});
  }
};

class CpuBfloat16DequantInplacePattern1_1 : public paddle::drr::DrrPatternBase {
 private:
  std::string bfloat16_ops_;
  uint32_t benefit_;
  uint32_t index_;

 public:
  CpuBfloat16DequantInplacePattern1_1(const std::string &bfloat16_ops,
                                      uint32_t benefit)
      : bfloat16_ops_(bfloat16_ops), benefit_(benefit), index_(0) {}

  std::string name() const override {
    return bfloat16_ops_ + "CpuBfloat16DequantInplacePattern1_1";
  }

  uint32_t benefit() const override { return benefit_; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    std::unordered_map<std::string, paddle::drr::Attribute> op_attrs;
    if (bfloat16_ops_ == "onednn_op.cast" ||
        bfloat16_ops_ == "onednn_op.cast_") {
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
      op_attrs.emplace("dtype", pat.Attr("dtype"));

    } else if (bfloat16_ops_ == "onednn_op.relu" ||
               bfloat16_ops_ == "onednn_op.relu_") {
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));

    } else if (bfloat16_ops_ == "onednn_op.sigmoid" ||
               bfloat16_ops_ == "onednn_op.sigmoid_") {
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
    }

    const auto &op = pat.Op(bfloat16_ops_, op_attrs);
    op({&pat.Tensor("x")}, {&pat.Tensor("out")});

    paddle::drr::ResultPattern res = pat.ResultPattern();

    pat.AddConstraint([this](const paddle::drr::MatchContext &match_ctx) {
      pir::Operation *input_op = match_ctx.Tensor("out").defining_op();

      auto mkldnn_data_type = match_ctx.Attr<std::string>("mkldnn_data_type");
      bool is_bfloat16 = true;
      if (mkldnn_data_type != "bfloat16") {
        // return false;
        is_bfloat16 = false;
      }

      std::vector<std::string> nopermitted_output_names = {"xshape"};
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

      auto it = std::find(nopermitted_output_names.begin(),
                          nopermitted_output_names.end(),
                          output_name);
      if (it != nopermitted_output_names.end()) {
        return false;
      }

      bool need_quantized = false;

      auto next_ops = pir::GetUseOpsForOutput(input_op, 0);
      for (auto [next_op, _] : next_ops) {
        if (!next_op->isa<paddle::onednn::dialect::DequantizeOp>()) {
          need_quantized = true;
        }
      }
      bool pre_quantized = false;
      for (size_t i = 0; i < input_op->num_operands(); ++i) {
        auto *pre_i_op = pir::GetDefiningOpForInput(input_op, i);
        if (!pre_i_op) {
          continue;
        }
        if (pre_i_op->isa<paddle::onednn::dialect::QuantizeOp>()) {
          pre_quantized = true;
        }
      }
      if (is_bfloat16) {
        if (need_quantized) {
          return true;
        } else {
          return false;
        }
      } else {
        if (pre_quantized && need_quantized) {
          return true;
        }
      }

      return false;
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

class CpuBfloat16InplacePattern2_2 : public paddle::drr::DrrPatternBase {
 private:
  std::string bfloat16_ops_;
  uint32_t benefit_;
  uint32_t index_;

 public:
  CpuBfloat16InplacePattern2_2(const std::string &bfloat16_ops,
                               uint32_t benefit,
                               uint32_t index)
      : bfloat16_ops_(bfloat16_ops), benefit_(benefit), index_(index) {}

  std::string name() const override {
    return bfloat16_ops_ + "CpuBfloat16InplacePattern2_2";
  }

  uint32_t benefit() const override { return benefit_; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    std::unordered_map<std::string, paddle::drr::Attribute> op_attrs;
    if (bfloat16_ops_ == "onednn_op.reshape_" ||
        bfloat16_ops_ == "onednn_op.reshape") {
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
      // op_attrs.emplace("shape", pat.Attr("shape"));
      op_attrs.emplace("use_quantizer", pat.Attr("use_quantizer"));

    } else if (bfloat16_ops_ == "onednn_op.squeeze" ||
               bfloat16_ops_ == "onednn_op.squeeze_") {
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
    }
    const auto &op = pat.Op(bfloat16_ops_, op_attrs);

    op({&pat.Tensor("quantize_0"), &pat.Tensor("quantize_1")},
       {&pat.Tensor("out_0"), &pat.Tensor("out_1")});

    const std::vector<std::string> permitted_input_names = {
        "x", "y", "input", "residual_param"};

    pat.AddConstraint([this](const paddle::drr::MatchContext &match_ctx) {
      auto mkldnn_data_type = match_ctx.Attr<std::string>("mkldnn_data_type");
      bool is_bfloat16 = true;
      if (mkldnn_data_type != "bfloat16") {
        // return false;
        is_bfloat16 = false;
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
      bool need_quantized = false;
      auto *pre_op = pir::GetDefiningOpForInput(input_op, index_);
      if (!pre_op) {
        return false;
      }
      if (!pre_op->isa<paddle::onednn::dialect::QuantizeOp>()) {
        need_quantized = true;
      }
      bool pre_quantized = false;
      for (size_t i = 0; i < input_op->num_operands(); ++i) {
        if (i != index_) {
          auto *pre_i_op = pir::GetDefiningOpForInput(input_op, i);
          if (pre_i_op->isa<paddle::onednn::dialect::QuantizeOp>()) {
            pre_quantized = true;
          }
        }
      }
      auto next_ops = pir::GetUseOpsForOutput(input_op, 0);
      for (auto [next_op, _] : next_ops) {
        if (next_op->isa<paddle::onednn::dialect::DequantizeOp>()) {
          pre_quantized = true;
        }
      }
      /*
        The reason to do this is some of ops, like add(add_), rele(relu_)...
        the mkldnn_data_type will auto convert to fp32, and can not
        find reason, maybe related Inplace op in PIR.
      */
      if (is_bfloat16) {
        if (need_quantized) {
          return true;
        } else {
          return false;
        }
      } else {
        if (pre_quantized && need_quantized) {
          return true;
        }
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
      res_op({&res.Tensor("quantize_out_0"), &res.Tensor("quantize_1")},
             {{&res.Tensor("out_0"), &res.Tensor("out_1")}});

    } else if (index_ == 1) {
      res_op({&res.Tensor("quantize_0"), &res.Tensor("quantize_out_1")},
             {{&res.Tensor("out_0"), &res.Tensor("out_1")}});
    }
  }
};

class CpuBfloat16DequantInplacePattern2_2 : public paddle::drr::DrrPatternBase {
 private:
  std::string bfloat16_ops_;
  uint32_t benefit_;
  uint32_t index_;

 public:
  CpuBfloat16DequantInplacePattern2_2(const std::string &bfloat16_ops,
                                      uint32_t benefit,
                                      uint32_t index)
      : bfloat16_ops_(bfloat16_ops), benefit_(benefit), index_(index) {}

  std::string name() const override {
    return bfloat16_ops_ + "CpuBfloat16DequantInplacePattern2_2";
  }

  uint32_t benefit() const override { return benefit_; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    std::unordered_map<std::string, paddle::drr::Attribute> op_attrs;
    if (bfloat16_ops_ == "onednn_op.reshape_" ||
        bfloat16_ops_ == "onednn_op.reshape") {
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
      // op_attrs.emplace("shape", pat.Attr("shape"));
      op_attrs.emplace("use_quantizer", pat.Attr("use_quantizer"));

    } else if (bfloat16_ops_ == "onednn_op.squeeze" ||
               bfloat16_ops_ == "onednn_op.squeeze_") {
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
    }
    const auto &op = pat.Op(bfloat16_ops_, op_attrs);
    op({&pat.Tensor("x"), &pat.Tensor("y")},
       {&pat.Tensor("out_0"), &pat.Tensor("out_1")});

    paddle::drr::ResultPattern res = pat.ResultPattern();

    pat.AddConstraint([this](const paddle::drr::MatchContext &match_ctx) {
      pir::Operation *input_op = match_ctx.Tensor("out_0").defining_op();

      auto mkldnn_data_type = match_ctx.Attr<std::string>("mkldnn_data_type");
      bool is_bfloat16 = true;
      if (mkldnn_data_type != "bfloat16") {
        // return false;
        is_bfloat16 = false;
      }

      std::vector<std::string> nopermitted_output_names = {"xshape"};
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

      auto it = std::find(nopermitted_output_names.begin(),
                          nopermitted_output_names.end(),
                          output_name);
      if (it != nopermitted_output_names.end()) {
        return false;
      }

      bool need_quantized = false;
      auto next_ops = pir::GetUseOpsForOutput(input_op, index_);
      for (auto [next_op, _] : next_ops) {
        if (!next_op->isa<paddle::onednn::dialect::DequantizeOp>()) {
          need_quantized = true;
        }
      }
      bool pre_quantized = false;
      for (size_t i = 0; i < input_op->num_operands(); ++i) {
        auto *pre_i_op = pir::GetDefiningOpForInput(input_op, i);
        if (!pre_i_op) {
          continue;
        }
        if (pre_i_op->isa<paddle::onednn::dialect::QuantizeOp>()) {
          pre_quantized = true;
        }
      }
      if (is_bfloat16) {
        if (need_quantized) {
          return true;
        } else {
          return false;
        }
      } else {
        if (pre_quantized && need_quantized) {
          return true;
        }
      }

      return false;
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
    } else if (index_ == 1) {
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

class CpuBfloat16InplacePattern3_1 : public paddle::drr::DrrPatternBase {
 private:
  std::string bfloat16_ops_;
  uint32_t benefit_;
  uint32_t index_;

 public:
  CpuBfloat16InplacePattern3_1(const std::string &bfloat16_ops,
                               uint32_t benefit,
                               uint32_t index)
      : bfloat16_ops_(bfloat16_ops), benefit_(benefit), index_(index) {}

  std::string name() const override {
    return bfloat16_ops_ + "CpuBfloat16InplacePattern3_1";
  }

  uint32_t benefit() const override { return benefit_; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    std::unordered_map<std::string, paddle::drr::Attribute> op_attrs;
    if (bfloat16_ops_ == "onednn_op.clip" ||
        bfloat16_ops_ == "onednn_op.clip_") {
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
      // op_attrs.emplace("min", pat.Attr("min"));
      // op_attrs.emplace("max", pat.Attr("max"));
    }
    const auto &op = pat.Op(bfloat16_ops_, op_attrs);
    op({&pat.Tensor("quantize_0"),
        &pat.Tensor("quantize_1"),
        &pat.Tensor("quantize_2")},
       {&pat.Tensor("out")});

    const std::vector<std::string> permitted_input_names = {
        "x", "y", "input", "residual_param"};

    pat.AddConstraint([this](const paddle::drr::MatchContext &match_ctx) {
      auto mkldnn_data_type = match_ctx.Attr<std::string>("mkldnn_data_type");
      bool is_bfloat16 = true;
      if (mkldnn_data_type != "bfloat16") {
        // return false;
        is_bfloat16 = false;
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
      bool need_quantized = false;
      auto *pre_op = pir::GetDefiningOpForInput(input_op, index_);
      if (!pre_op->isa<paddle::onednn::dialect::QuantizeOp>()) {
        need_quantized = true;
      }
      bool pre_quantized = false;
      for (size_t i = 0; i < input_op->num_operands(); ++i) {
        if (i != index_) {
          auto *pre_i_op = pir::GetDefiningOpForInput(input_op, i);
          if (!pre_i_op) {
            continue;
          }
          if (pre_i_op->isa<paddle::onednn::dialect::QuantizeOp>()) {
            pre_quantized = true;
          }
        }
      }
      auto next_ops = pir::GetUseOpsForOutput(input_op, 0);
      for (auto [next_op, _] : next_ops) {
        if (next_op->isa<paddle::onednn::dialect::DequantizeOp>()) {
          pre_quantized = true;
        }
      }
      /*
        The reason to do this is some of ops, like add(add_), rele(relu_)...
        the mkldnn_data_type will auto convert to fp32, and can not
        find reason, maybe related Inplace op in PIR.
      */
      if (is_bfloat16) {
        if (need_quantized) {
          return true;
        } else {
          return false;
        }
      } else {
        if (pre_quantized && need_quantized) {
          return true;
        }
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
      res_op({&res.Tensor("quantize_out_0"),
              &res.Tensor("quantize_1"),
              &res.Tensor("quantize_2")},
             {{&res.Tensor("out")}});

    } else if (index_ == 1) {
      res_op({&res.Tensor("quantize_0"),
              &res.Tensor("quantize_out_1"),
              &res.Tensor("quantize_2")},
             {{&res.Tensor("out")}});

    } else if (index_ == 2) {
      res_op({&res.Tensor("quantize_0"),
              &res.Tensor("quantize_1"),
              &res.Tensor("quantize_out_2")},
             {{&res.Tensor("out")}});
    }
  }
};

class CpuBfloat16DequantInplacePattern3_1 : public paddle::drr::DrrPatternBase {
 private:
  std::string bfloat16_ops_;
  uint32_t benefit_;
  uint32_t index_;

 public:
  CpuBfloat16DequantInplacePattern3_1(const std::string &bfloat16_ops,
                                      uint32_t benefit)
      : bfloat16_ops_(bfloat16_ops), benefit_(benefit), index_(0) {}

  std::string name() const override {
    return bfloat16_ops_ + "CpuBfloat16DequantInplacePattern3_1";
  }

  uint32_t benefit() const override { return benefit_; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    std::unordered_map<std::string, paddle::drr::Attribute> op_attrs;
    if (bfloat16_ops_ == "onednn_op.clip" ||
        bfloat16_ops_ == "onednn_op.clip_") {
      op_attrs.emplace("mkldnn_data_type", pat.Attr("mkldnn_data_type"));
    }

    const auto &op = pat.Op(bfloat16_ops_, op_attrs);
    op({&pat.Tensor("x"), &pat.Tensor("y"), &pat.Tensor("z")},
       {&pat.Tensor("out")});

    paddle::drr::ResultPattern res = pat.ResultPattern();

    pat.AddConstraint([this](const paddle::drr::MatchContext &match_ctx) {
      pir::Operation *input_op = match_ctx.Tensor("out").defining_op();

      auto mkldnn_data_type = match_ctx.Attr<std::string>("mkldnn_data_type");
      bool is_bfloat16 = true;
      if (mkldnn_data_type != "bfloat16") {
        // return false;
        is_bfloat16 = false;
      }

      std::vector<std::string> nopermitted_output_names = {"xshape"};
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

      auto it = std::find(nopermitted_output_names.begin(),
                          nopermitted_output_names.end(),
                          output_name);
      if (it != nopermitted_output_names.end()) {
        return false;
      }

      bool need_quantized = false;
      auto next_ops = pir::GetUseOpsForOutput(input_op, 0);
      for (auto [next_op, _] : next_ops) {
        if (!next_op->isa<paddle::onednn::dialect::DequantizeOp>()) {
          need_quantized = true;
        }
      }
      bool pre_quantized = false;
      for (size_t i = 0; i < input_op->num_operands(); ++i) {
        auto *pre_i_op = pir::GetDefiningOpForInput(input_op, i);
        if (!pre_i_op) {
          continue;
        }
        if (pre_i_op->isa<paddle::onednn::dialect::QuantizeOp>()) {
          pre_quantized = true;
        }
      }
      if (is_bfloat16) {
        if (need_quantized) {
          return true;
        } else {
          return false;
        }
      } else {
        if (pre_quantized && need_quantized) {
          return true;
        }
      }

      return false;
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

class CpuBfloat16SpecialPass : public pir::PatternRewritePass {
 public:
  CpuBfloat16SpecialPass()
      : pir::PatternRewritePass("cpu_bfloat16_special_op_pass", 3) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    const std::vector<std::string> bfloat16_ops_two_one{
        // op with two inputs and one output
        paddle::onednn::dialect::AddOp::name(),
        paddle::onednn::dialect::Add_Op::name(),
        paddle::onednn::dialect::MultiplyOp::name(),
        paddle::onednn::dialect::Multiply_Op::name(),
        paddle::onednn::dialect::ScaleOp::name(),
        paddle::onednn::dialect::Scale_Op::name(),

    };

    const std::vector<std::string> bfloat16_ops_one_one{
        // op with one inputs and one output
        paddle::onednn::dialect::CastOp::name(),
        paddle::onednn::dialect::Cast_Op::name(),
        paddle::onednn::dialect::ReluOp::name(),
        paddle::onednn::dialect::Relu_Op::name(),
        paddle::onednn::dialect::SigmoidOp::name(),
        paddle::onednn::dialect::Sigmoid_Op::name(),
        paddle::onednn::dialect::SoftmaxOp::name(),
        paddle::onednn::dialect::Softmax_Op::name(),
        paddle::onednn::dialect::TransposeOp::name(),
        paddle::onednn::dialect::Transpose_Op::name(),

    };

    const std::vector<std::string> bfloat16_ops_two_two{
        // op with two inputs and two output
        paddle::onednn::dialect::Reshape_Op::name(),
        paddle::onednn::dialect::ReshapeOp::name(),
        paddle::onednn::dialect::SqueezeOp::name(),
        paddle::onednn::dialect::Squeeze_Op::name(),
    };

    const std::vector<std::string> bfloat16_ops_three_one{
        // op with three inputs and one output
        paddle::onednn::dialect::ClipOp::name(),
        paddle::onednn::dialect::Clip_Op::name(),
    };

    // op with two inputs and one output
    int benefit_idx = 1;
    for (auto op : bfloat16_ops_two_one) {
      ps.Add(paddle::drr::Create<CpuBfloat16InplacePattern>(
          context, op, benefit_idx, 0));
      benefit_idx++;
    }
    for (auto op : bfloat16_ops_two_one) {
      ps.Add(paddle::drr::Create<CpuBfloat16InplacePattern>(
          context, op, benefit_idx, 1));
      benefit_idx++;
    }

    benefit_idx = 1;
    for (auto op : bfloat16_ops_two_one) {
      ps.Add(paddle::drr::Create<CpuBfloat16DequantInplacePattern>(
          context, op, benefit_idx));
      benefit_idx++;
    }

    // op with one inputs and one output
    benefit_idx = 1;
    for (auto op : bfloat16_ops_one_one) {
      ps.Add(paddle::drr::Create<CpuBfloat16InplacePattern1_1>(
          context, op, benefit_idx));
      benefit_idx++;
    }
    benefit_idx = 1;
    for (auto op : bfloat16_ops_one_one) {
      ps.Add(paddle::drr::Create<CpuBfloat16DequantInplacePattern1_1>(
          context, op, benefit_idx));
      benefit_idx++;
    }

    // op with two inputs and two output
    benefit_idx = 1;
    for (auto op : bfloat16_ops_two_two) {
      ps.Add(paddle::drr::Create<CpuBfloat16InplacePattern2_2>(
          context, op, benefit_idx, 0));
      benefit_idx++;
    }

    // shape or aixs not in permitied list, not use quant op before it
    // benefit_idx = 1;
    // for (auto op : bfloat16_ops_two_two) {
    //   if( op != paddle::onednn::dialect::ReshapeOp::name() ||
    //     op != paddle::onednn::dialect::Reshape_Op::name())
    //   ps.Add(paddle::drr::Create<CpuBfloat16InplacePattern2_2>(
    //       context, op, benefit_idx, 1));
    //   benefit_idx++;
    // }
    benefit_idx = 1;
    for (auto op : bfloat16_ops_two_two) {
      ps.Add(paddle::drr::Create<CpuBfloat16DequantInplacePattern2_2>(
          context, op, benefit_idx, 0));
      benefit_idx++;
    }
    benefit_idx = 1;
    for (auto op : bfloat16_ops_two_two) {
      ps.Add(paddle::drr::Create<CpuBfloat16DequantInplacePattern2_2>(
          context, op, benefit_idx, 1));
      benefit_idx++;
    }

    // op with three inputs and one output
    benefit_idx = 1;
    for (auto op : bfloat16_ops_three_one) {
      ps.Add(paddle::drr::Create<CpuBfloat16InplacePattern3_1>(
          context, op, benefit_idx, 0));
      benefit_idx++;
    }
    benefit_idx = 1;
    for (auto op : bfloat16_ops_three_one) {
      ps.Add(paddle::drr::Create<CpuBfloat16InplacePattern3_1>(
          context, op, benefit_idx, 1));
      benefit_idx++;
    }
    benefit_idx = 1;
    for (auto op : bfloat16_ops_three_one) {
      ps.Add(paddle::drr::Create<CpuBfloat16InplacePattern3_1>(
          context, op, benefit_idx, 2));
      benefit_idx++;
    }
    benefit_idx = 1;
    for (auto op : bfloat16_ops_three_one) {
      ps.Add(paddle::drr::Create<CpuBfloat16DequantInplacePattern3_1>(
          context, op, benefit_idx));
      benefit_idx++;
    }

    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateCpuBf16SpecialPatternPass() {
  return std::make_unique<CpuBfloat16SpecialPass>();
}
}  // namespace pir

REGISTER_IR_PASS(cpu_bfloat16_special_op_pass, CpuBfloat16SpecialPass);
