// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/cinn/hlir/dialect/operator/transforms/pd_to_cinn_pass.h"

#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/op_with_group_merge_util.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/pir/core/builtin_dialect.h"
#include "paddle/pir/core/builtin_op.h"
#include "paddle/pir/pass/pass.h"
#include "paddle/pir/pass/pass_manager.h"
#include "paddle/pir/pattern_rewrite/pattern_rewrite_driver.h"

namespace cinn {
namespace dialect {
namespace ir {
using CompatibleInfo = cinn::hlir::framework::pir::CompatibleInfo;

class SumOpPattern : public paddle::drr::DrrPatternBase {
 public:
  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    // Source Pattern
    paddle::drr::SourcePattern pattern = ctx->SourcePattern();
    const auto &full_int_array =
        pattern.Op(paddle::dialect::FullIntArrayOp::name(),
                   {{"value", pattern.Attr("axis_info")},
                    {"dtype", pattern.Attr("dtype_2")},
                    {"place", pattern.Attr("place_2")}});

    const auto &sum = pattern.Op(paddle::dialect::SumOp::name(),
                                 {{"dtype", pattern.Attr("dtype")},
                                  {"keepdim", pattern.Attr("keep_dim")}});
    pattern.Tensor("ret") = sum(pattern.Tensor("arg0"), full_int_array());

    // Result patterns
    paddle::drr::ResultPattern res = pattern.ResultPattern();
    const auto &cinn_reduce_sum =
        res.Op(cinn::dialect::ReduceSumOp::name(),
               {{"dim", pattern.Attr("axis_info")},
                {"keep_dim", pattern.Attr("keep_dim")}});
    res.Tensor("ret") = cinn_reduce_sum(res.Tensor("arg0"));
  }

  std::string name() const override { return "SumOpPattern"; }
};

class MaxOpPattern : public paddle::drr::DrrPatternBase {
 public:
  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    // Source Pattern
    paddle::drr::SourcePattern pattern = ctx->SourcePattern();
    const auto &full_int_array =
        pattern.Op(paddle::dialect::FullIntArrayOp::name(),
                   {{"value", pattern.Attr("axis_info")},
                    {"dtype", pattern.Attr("dtype_2")},
                    {"place", pattern.Attr("place_2")}});

    const auto &pd_max = pattern.Op(paddle::dialect::MaxOp::name(),
                                    {{"keepdim", pattern.Attr("keep_dim")}});
    pattern.Tensor("ret") = pd_max(pattern.Tensor("arg0"), full_int_array());

    // Result patterns
    paddle::drr::ResultPattern res = pattern.ResultPattern();
    const auto &cinn_reduce_max =
        res.Op(cinn::dialect::ReduceMaxOp::name(),
               {{"dim", pattern.Attr("axis_info")},
                {"keep_dim", pattern.Attr("keep_dim")}});
    res.Tensor("ret") = cinn_reduce_max(res.Tensor("arg0"));
  }

  std::string name() const override { return "MaxOpPattern"; }
};

class MinOpPattern : public paddle::drr::DrrPatternBase {
 public:
  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    // Source Pattern
    paddle::drr::SourcePattern pattern = ctx->SourcePattern();
    const auto &full_int_array =
        pattern.Op(paddle::dialect::FullIntArrayOp::name(),
                   {{"value", pattern.Attr("axis_info")},
                    {"dtype", pattern.Attr("dtype_2")},
                    {"place", pattern.Attr("place_2")}});

    const auto &pd_max = pattern.Op(paddle::dialect::MinOp::name(),
                                    {{"keepdim", pattern.Attr("keep_dim")}});
    pattern.Tensor("ret") = pd_max(pattern.Tensor("arg0"), full_int_array());

    // Result patterns
    paddle::drr::ResultPattern res = pattern.ResultPattern();
    const auto &cinn_reduce_max =
        res.Op(cinn::dialect::ReduceMinOp::name(),
               {{"dim", pattern.Attr("axis_info")},
                {"keep_dim", pattern.Attr("keep_dim")}});
    res.Tensor("ret") = cinn_reduce_max(res.Tensor("arg0"));
  }

  std::string name() const override { return "MinOpPattern"; }
};

class ProdOpPattern : public paddle::drr::DrrPatternBase {
 public:
  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    // Source Pattern
    paddle::drr::SourcePattern pattern = ctx->SourcePattern();
    const auto &full_int_array =
        pattern.Op(paddle::dialect::FullIntArrayOp::name(),
                   {{"value", pattern.Attr("axis_info")},
                    {"dtype", pattern.Attr("dtype_2")},
                    {"place", pattern.Attr("place_2")}});

    const auto &pd_max = pattern.Op(paddle::dialect::ProdOp::name(),
                                    {{"keep_dim", pattern.Attr("keep_dim")}});
    pattern.Tensor("ret") = pd_max(pattern.Tensor("arg0"), full_int_array());

    // Result patterns
    paddle::drr::ResultPattern res = pattern.ResultPattern();
    const auto &cinn_reduce_max =
        res.Op(cinn::dialect::ReduceProdOp::name(),
               {{"dim", pattern.Attr("axis_info")},
                {"keep_dim", pattern.Attr("keep_dim")}});
    res.Tensor("ret") = cinn_reduce_max(res.Tensor("arg0"));
  }

  std::string name() const override { return "ProdOpPattern"; }
};

class ScaleOpPattern : public pir::OpRewritePattern<paddle::dialect::ScaleOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::ScaleOp>::OpRewritePattern;

  bool Match(paddle::dialect::ScaleOp op) const override {
    bool flag = CompatibleInfo::IsSupportCinn(*op.operation());
    return flag;
  }

  void Rewrite(paddle::dialect::ScaleOp op,
               pir::PatternRewriter &rewriter) const override {
    auto scale_factor_gen_op = op->operand_source(1).defining_op();

    if (auto full_op =
            scale_factor_gen_op->dyn_cast<paddle::dialect::FullOp>()) {
      // scale is generator by full op
      // get attribute value from full op
      auto scale_value =
          full_op.attribute("value").dyn_cast<pir::FloatAttribute>().data();

      auto cinn_scale = rewriter.Build<cinn::dialect::ScaleOp>(
          op->operand_source(0),
          scale_value,
          op->attributes().at("bias").dyn_cast<pir::FloatAttribute>().data(),
          op->attributes()
              .at("bias_after_scale")
              .dyn_cast<pir::BoolAttribute>()
              .data());
      rewriter.ReplaceAllUsesWith(op.result(0), cinn_scale.result(0));
      rewriter.EraseOp(op);
    } else {
      // using mul op
      auto bias =
          op->attributes().at("bias").dyn_cast<pir::FloatAttribute>().data();

      auto mul_in = op.operand_source(0);
      if (bias != 0.0f) {
        auto full_op = rewriter.Build<paddle::dialect::FullOp>(
            std::vector<int64_t>({1}), bias, phi::DataType::FLOAT32);
        auto add_op = rewriter.Build<paddle::dialect::AddOp>(
            op.operand_source(0), full_op.result(0));
        mul_in = add_op.result(0);
      }

      auto mul_op = rewriter.Build<paddle::dialect::MultiplyOp>(
          mul_in, op->operand_source(1));

      rewriter.ReplaceAllUsesWith(op.result(0), mul_op.result(0));
      rewriter.EraseOp(op);
    }
  }
};

class ReshapeOpPattern
    : public pir::OpRewritePattern<paddle::dialect::ReshapeOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::ReshapeOp>::OpRewritePattern;

  bool Match(paddle::dialect::ReshapeOp op) const override {
    bool flag = CompatibleInfo::IsSupportCinn(*op.operation());
    auto scale_factor_gen_op = op->operand_source(1).defining_op();
    auto full_op =
        scale_factor_gen_op->dyn_cast<paddle::dialect::FullIntArrayOp>();
    return flag && full_op;
  }

  void Rewrite(paddle::dialect::ReshapeOp op,
               pir::PatternRewriter &rewriter) const override {
    auto scale_factor_gen_op = op->operand_source(1).defining_op();

    auto full_op =
        scale_factor_gen_op->dyn_cast<paddle::dialect::FullIntArrayOp>();
    // scale is generator by full op
    // get attribute value from full op

    auto out_shape_attr =
        full_op.attribute("value").dyn_cast<pir::ArrayAttribute>().AsVector();

    std::vector<int> vec_out_shape;
    if (out_shape_attr.size() > 0) {
      PADDLE_ENFORCE_EQ(out_shape_attr[0].isa<::pir::Int64Attribute>(),
                        true,
                        phi::errors::Unimplemented(
                            "the 0th elementwise MUST be ir::Int64Attribute"));
      for (size_t i = 0; i < out_shape_attr.size(); ++i) {
        vec_out_shape.push_back(
            out_shape_attr[i].dyn_cast<::pir::Int64Attribute>().data());
      }
    }

    auto cinn_reshape = rewriter.Build<cinn::dialect::ReshapeOp>(
        op->operand_source(0), vec_out_shape);
    rewriter.ReplaceAllUsesWith(op.result(0), cinn_reshape.result(0));
    rewriter.EraseOp(op);
  }
};

class Pool2dOpPattern
    : public pir::OpRewritePattern<paddle::dialect::Pool2dOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::Pool2dOp>::OpRewritePattern;

  bool Match(paddle::dialect::Pool2dOp op) const override {
    bool flag = CompatibleInfo::IsSupportCinn(*op.operation());
    auto kernel_size_gen_op = op->operand_source(1).defining_op();
    auto full_op =
        kernel_size_gen_op->dyn_cast<paddle::dialect::FullIntArrayOp>();
    return flag && full_op;
  }

  void Rewrite(paddle::dialect::Pool2dOp op,
               pir::PatternRewriter &rewriter) const override {
    auto kernel_size_gen_op = op->operand_source(1).defining_op();
    auto full_op =
        kernel_size_gen_op->dyn_cast<paddle::dialect::FullIntArrayOp>();

    auto kernel_size_attr =
        full_op.attribute("value").dyn_cast<pir::ArrayAttribute>().AsVector();

    // kernel_size is generator by full op
    // get attribute value from full op
    std::vector<pir::Attribute> kernel_size;
    for (size_t i = 0; i < static_cast<size_t>(kernel_size_attr.size()); i++) {
      pir::Attribute attr = pir::Int32Attribute::get(
          pir::IrContext::Instance(),
          kernel_size_attr[i].dyn_cast<::pir::Int64Attribute>().data());
      kernel_size.push_back(attr);
    }
    auto attrs = op->attributes();
    attrs["kernel_size"] =
        pir::ArrayAttribute::get(pir::IrContext::Instance(), kernel_size);
    attrs["stride_size"] = attrs.at("strides");
    attrs["padding_size"] = attrs.at("paddings");
    attrs["pool_type"] = attrs.at("pooling_type");
    attrs.erase("strides");
    attrs.erase("paddings");
    attrs.erase("pooling_type");

    auto cinn_reshape =
        rewriter.Build<cinn::dialect::Pool2dOp>(op->operand_source(0), attrs);
    rewriter.ReplaceAllUsesWith(op.result(0), cinn_reshape.result(0));
    rewriter.EraseOp(op);
  }
};

class IsCloseOpPattern
    : public pir::OpRewritePattern<paddle::dialect::IscloseOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::IscloseOp>::OpRewritePattern;

  bool Match(paddle::dialect::IscloseOp op) const override {
    bool flag = CompatibleInfo::IsSupportCinn(*op.operation());
    auto rtol_op = op->operand_source(2)
                       .defining_op()
                       ->dyn_cast<paddle::dialect::FullOp>();
    auto atol_op = op->operand_source(3)
                       .defining_op()
                       ->dyn_cast<paddle::dialect::FullOp>();
    return flag && rtol_op && atol_op;
  }

  void Rewrite(paddle::dialect::IscloseOp op,
               pir::PatternRewriter &rewriter) const override {
    auto rtol_op = op->operand_source(2)
                       .defining_op()
                       ->dyn_cast<paddle::dialect::FullOp>();

    auto atol_op = op->operand_source(3)
                       .defining_op()
                       ->dyn_cast<paddle::dialect::FullOp>();

    auto rtol_val =
        rtol_op.attribute("value").dyn_cast<::pir::FloatAttribute>().data();
    auto atol_val =
        atol_op.attribute("value").dyn_cast<::pir::FloatAttribute>().data();
    auto equal_nan =
        op->attribute("equal_nan").dyn_cast<::pir::BoolAttribute>().data();

    auto cinn_isclose =
        rewriter.Build<cinn::dialect::IscloseOp>(op->operand_source(0),
                                                 op->operand_source(1),
                                                 rtol_val,
                                                 atol_val,
                                                 equal_nan);
    rewriter.ReplaceAllUsesWith(op.result(0), cinn_isclose.result(0));
    rewriter.EraseOp(op);
  }
};

class SliceOpPattern : public pir::OpRewritePattern<paddle::dialect::SliceOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::SliceOp>::OpRewritePattern;

  bool Match(paddle::dialect::SliceOp op) const override {
    bool flag = CompatibleInfo::IsSupportCinn(*op.operation());
    auto start_gen_op = op->operand_source(1)
                            .defining_op()
                            ->dyn_cast<paddle::dialect::FullIntArrayOp>();

    auto end_gen_op = op->operand_source(2)
                          .defining_op()
                          ->dyn_cast<paddle::dialect::FullIntArrayOp>();
    return flag && start_gen_op && end_gen_op;
  }

  void Rewrite(paddle::dialect::SliceOp op,
               pir::PatternRewriter &rewriter) const override {
    auto start_gen_op = op->operand_source(1)
                            .defining_op()
                            ->dyn_cast<paddle::dialect::FullIntArrayOp>();

    auto end_gen_op = op->operand_source(2)
                          .defining_op()
                          ->dyn_cast<paddle::dialect::FullIntArrayOp>();
    // scale is generator by full op
    // get attribute value from full op
    auto start_vec = cinn::dialect::ir::GetVectorAttr(start_gen_op, "value");
    auto end_vec = cinn::dialect::ir::GetVectorAttr(end_gen_op, "value");
    auto axes = cinn::dialect::ir::GetVectorAttr(op, "axes");
    auto decrease_axis = cinn::dialect::ir::GetVectorAttr(op, "decrease_axis");
    auto infer_flags = cinn::dialect::ir::GetVectorAttr(op, "infer_flags");

    auto cinn_slice =
        rewriter.Build<cinn::dialect::SliceOp>(op->operand_source(0),
                                               axes,
                                               start_vec,
                                               end_vec,
                                               infer_flags,
                                               decrease_axis);
    // NOTE(Aurelius84): In SliceRawInferMeta, it not always share_lod, so
    // we need to update it maually.
    cinn_slice.result(0).set_type(op.result(0).type());
    rewriter.ReplaceAllUsesWith(op.result(0), cinn_slice.result(0));
    rewriter.EraseOp(op);
  }
};

class ConcatOpPattern
    : public pir::OpRewritePattern<paddle::dialect::ConcatOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::ConcatOp>::OpRewritePattern;

  bool Match(paddle::dialect::ConcatOp op) const override {
    bool flag = CompatibleInfo::IsSupportCinn(*op.operation());
    auto axis_gen_op = op->operand_source(1).defining_op();
    return flag && axis_gen_op->dyn_cast<paddle::dialect::FullOp>();
  }

  void Rewrite(paddle::dialect::ConcatOp op,
               pir::PatternRewriter &rewriter) const override {
    auto axis_gen_op = op->operand_source(1).defining_op();
    auto full_op = axis_gen_op->dyn_cast<paddle::dialect::FullOp>();
    int axis = static_cast<int>(
        full_op.attribute("value").dyn_cast<::pir::FloatAttribute>().data());

    auto input_ops = op->operand_source(0)
                         .defining_op()
                         ->dyn_cast<pir::CombineOp>()
                         .inputs();

    auto cinn_concat = rewriter.Build<cinn::dialect::ConcatOp>(input_ops, axis);
    rewriter.ReplaceAllUsesWith(op.result(0), cinn_concat.result(0));
    rewriter.EraseOp(op);
  }
};

class PowOpPattern : public pir::OpRewritePattern<paddle::dialect::PowOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::PowOp>::OpRewritePattern;

  bool Match(paddle::dialect::PowOp op) const override {
    bool flag = CompatibleInfo::IsSupportCinn(*op.operation());
    return flag;
  }

  void Rewrite(paddle::dialect::PowOp op,
               pir::PatternRewriter &rewriter) const override {
    auto factor = op->attribute("y").dyn_cast<pir::FloatAttribute>().data();
    auto full_op =
        rewriter.Build<paddle::dialect::FullOp>(std::vector<int64_t>({1}),
                                                factor,
                                                phi::DataType::FLOAT32,
                                                phi::CPUPlace());

    auto elementwise_pow = rewriter.Build<paddle::dialect::ElementwisePowOp>(
        op->operand_source(0), full_op->result(0));
    rewriter.ReplaceAllUsesWith(op.result(0), elementwise_pow.result(0));
    rewriter.EraseOp(op);
  }
};

class SplitOpPattern : public pir::OpRewritePattern<paddle::dialect::SplitOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::SplitOp>::OpRewritePattern;

  bool Match(paddle::dialect::SplitOp op) const override {
    bool flag = CompatibleInfo::IsSupportCinn(*op.operation());
    auto sections_gen_op = op->operand_source(1)
                               .defining_op()
                               ->dyn_cast<paddle::dialect::FullIntArrayOp>();
    auto axis_gen_op = op->operand_source(2)
                           .defining_op()
                           ->dyn_cast<paddle::dialect::FullOp>();
    return flag && sections_gen_op && axis_gen_op;
  }

  void Rewrite(paddle::dialect::SplitOp op,
               pir::PatternRewriter &rewriter) const override {
    auto sections_gen_op = op->operand_source(1)
                               .defining_op()
                               ->dyn_cast<paddle::dialect::FullIntArrayOp>();
    auto axis_gen_op = op->operand_source(2)
                           .defining_op()
                           ->dyn_cast<paddle::dialect::FullOp>();
    auto section_attr = sections_gen_op.attribute("value")
                            .dyn_cast<pir::ArrayAttribute>()
                            .AsVector();

    std::vector<int> vec_sections;
    if (section_attr.size() > 0) {
      for (size_t i = 0; i < section_attr.size(); ++i) {
        vec_sections.push_back(
            section_attr[i].dyn_cast<::pir::Int64Attribute>().data());
      }
    }
    int axis = static_cast<int>(axis_gen_op.attribute("value")
                                    .dyn_cast<::pir::FloatAttribute>()
                                    .data());

    auto input_ele = op->operand_source(0)
                         .type()
                         .dyn_cast<paddle::dialect::DenseTensorType>();
    if (axis < 0) {
      axis += input_ele.dims().size();
    }

    auto cinn_split = rewriter.Build<cinn::dialect::SplitOp>(
        op->operand_source(0), vec_sections, axis);

    auto orig_out = op.result(0);
    for (auto it = orig_out.use_begin(); it != orig_out.use_end();) {
      auto slice_op = (it++)->owner();
      CHECK(slice_op->isa<::pir::SliceOp>())
          << "Currently only support pir::slice as downstream op";
      int index = slice_op->dyn_cast<::pir::SliceOp>()
                      .attribute("index")
                      .dyn_cast<::pir::Int32Attribute>()
                      .data();
      rewriter.ReplaceAllUsesWith(slice_op->result(0),
                                  cinn_split.result(index));
      rewriter.EraseOp(slice_op);
    }
    rewriter.EraseOp(op);
  }
};

class SplitWithNumOpPattern
    : public pir::OpRewritePattern<paddle::dialect::SplitWithNumOp> {
 public:
  using pir::OpRewritePattern<
      paddle::dialect::SplitWithNumOp>::OpRewritePattern;

  bool Match(paddle::dialect::SplitWithNumOp op) const override {
    bool flag = CompatibleInfo::IsSupportCinn(*op.operation());
    auto axis_gen_op = op->operand_source(1).defining_op();
    auto full_op = axis_gen_op->dyn_cast<paddle::dialect::FullOp>();
    return flag && full_op;
  }

  void Rewrite(paddle::dialect::SplitWithNumOp op,
               pir::PatternRewriter &rewriter) const override {
    auto axis_gen_op = op->operand_source(1).defining_op();
    auto full_op = axis_gen_op->dyn_cast<paddle::dialect::FullOp>();
    int axis = static_cast<int>(
        full_op.attribute("value").dyn_cast<::pir::FloatAttribute>().data());

    auto input_ele = op->operand_source(0)
                         .type()
                         .dyn_cast<paddle::dialect::DenseTensorType>();
    if (axis < 0) {
      axis += input_ele.dims().size();
    }
    std::vector<int> sections;

    auto split_dim = input_ele.dims()[axis];

    auto split_num =
        op->attribute("num").dyn_cast<::pir::Int32Attribute>().data();
    auto part_ele = (split_dim + split_num - 1) / split_num;

    int total_split_num = 0;
    for (int i = 0; i < split_num - 1; ++i) {
      sections.push_back(part_ele);
      total_split_num += part_ele;
    }

    sections.push_back(split_dim - total_split_num);

    auto cinn_split = rewriter.Build<cinn::dialect::SplitOp>(
        op->operand_source(0), sections, axis);

    auto orig_out = op.result(0);
    for (auto it = orig_out.use_begin(); it != orig_out.use_end();) {
      auto slice_op = (it++)->owner();
      CHECK(slice_op->isa<::pir::SliceOp>());
      int index = slice_op->dyn_cast<::pir::SliceOp>()
                      .attribute("index")
                      .dyn_cast<::pir::Int32Attribute>()
                      .data();
      rewriter.ReplaceAllUsesWith(slice_op->result(0),
                                  cinn_split.result(index));
      rewriter.EraseOp(slice_op);
    }

    rewriter.EraseOp(op);
  }
};

class AddNOpPattern : public pir::OpRewritePattern<paddle::dialect::AddNOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::AddNOp>::OpRewritePattern;

  bool MatchAndRewrite(paddle::dialect::AddNOp op,
                       pir::PatternRewriter &rewriter) const override {
    auto combine_op =
        op->operand_source(0).defining_op()->dyn_cast<pir::CombineOp>();
    auto input_ops = combine_op.inputs();

    auto tmp = input_ops[0];

    for (size_t i = 1; i < input_ops.size(); ++i) {
      tmp = rewriter.Build<paddle::dialect::AddOp>(tmp, input_ops[i]).result(0);
    }

    rewriter.ReplaceAllUsesWith(op.result(0), tmp);

    rewriter.EraseOp(op);
    rewriter.EraseOp(combine_op);

    return true;
  }
};

class ExpandOpPattern
    : public pir::OpRewritePattern<paddle::dialect::ExpandOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::ExpandOp>::OpRewritePattern;

  bool Match(paddle::dialect::ExpandOp op) const override {
    bool flag = CompatibleInfo::IsSupportCinn(*op.operation());
    auto out_shape_gen_op = op->operand_source(1)
                                .defining_op()
                                ->dyn_cast<paddle::dialect::FullIntArrayOp>();
    return flag && out_shape_gen_op;
  }

  void Rewrite(paddle::dialect::ExpandOp op,
               pir::PatternRewriter &rewriter) const override {
    auto out_shape_gen_op = op->operand_source(1)
                                .defining_op()
                                ->dyn_cast<paddle::dialect::FullIntArrayOp>();

    auto section_attr = out_shape_gen_op.attribute("value")
                            .dyn_cast<pir::ArrayAttribute>()
                            .AsVector();

    std::vector<int64_t> output_shape;
    if (section_attr.size() > 0) {
      for (size_t i = 0; i < section_attr.size(); ++i) {
        output_shape.push_back(
            section_attr[i].dyn_cast<::pir::Int64Attribute>().data());
      }
    }

    auto in_dim = op.operand_source(0)
                      .type()
                      .dyn_cast<paddle::dialect::DenseTensorType>()
                      .dims();

    auto broadcast_axis =
        cinn::hlir::framework::pir::GetBroadcastAxis(in_dim, output_shape);

    auto out = rewriter
                   .Build<cinn::dialect::BroadcastOp>(
                       op.operand_source(0), broadcast_axis, output_shape)
                   .result(0);

    rewriter.ReplaceAllUsesWith(op.result(0), out);

    rewriter.EraseOp(op);
  }
};

class UniformOpPattern : public paddle::drr::DrrPatternBase {
 public:
  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    // Source Pattern
    paddle::drr::SourcePattern pattern = ctx->SourcePattern();
    const auto &full_int_array =
        pattern.Op(paddle::dialect::FullIntArrayOp::name(),
                   {{"value", pattern.Attr("axis_info")},
                    {"dtype", pattern.Attr("dtype_2")},
                    {"place", pattern.Attr("place_2")}});

    const auto &min_full = pattern.Op(paddle::dialect::FullOp::name(),
                                      {{"shape", pattern.Attr("shape1")},
                                       {"value", pattern.Attr("min_value")},
                                       {"dtype", pattern.Attr("dtype_min")},
                                       {"place", pattern.Attr("place_min")}});

    const auto &max_full = pattern.Op(paddle::dialect::FullOp::name(),
                                      {{"shape", pattern.Attr("shape2")},
                                       {"value", pattern.Attr("max_value")},
                                       {"dtype", pattern.Attr("dtype_max")},
                                       {"place", pattern.Attr("place_max")}});

    const auto &pd_uniform =
        pattern.Op(paddle::dialect::UniformOp::name(),
                   {{"dtype", pattern.Attr("uniform_dtype")},
                    {"place", pattern.Attr("uniform_place")},
                    {"seed", pattern.Attr("seed")}});
    pattern.Tensor("ret") =
        pd_uniform(full_int_array(), min_full(), max_full());
    // int64_t[] shape,  float min, float max, int seed, DataType dtype, int
    // diag_num, int diag_step, float diag_val)
    //  Result patterns
    paddle::drr::ResultPattern res = pattern.ResultPattern();
    const auto &cinn_uniform =
        res.Op(cinn::dialect::UniformRandomOp::name(),
               {{"shape", pattern.Attr("axis_info")},
                {"min", pattern.Attr("min_value")},
                {"max", pattern.Attr("max_value")},
                {"seed", pattern.Attr("seed")},
                {"dtype", pattern.Attr("uniform_dtype")},
                {"diag_num", pattern.Attr("seed")},
                {"diag_step", pattern.Attr("seed")},
                {"diag_val", pattern.Attr("min_value")}});
    res.Tensor("ret") = cinn_uniform();
  }

  std::string name() const override { return "ProdOpPattern"; }
};

PdOpToCinnOpPass::PdOpToCinnOpPass()
    : pir::PatternRewritePass("pd_to_cinn_pass", 1) {}

pir::RewritePatternSet PdOpToCinnOpPass::InitializePatterns(
    pir::IrContext *context) {
  pir::RewritePatternSet ps(context);
  ps.Add<ScaleOpPattern>(
      context);  // NOTE, scale op pattern should before AddBroadcastTo
  ps.Add(SumOpPattern().Build(context));
  ps.Add(MaxOpPattern().Build(context));
  ps.Add(MinOpPattern().Build(context));
  ps.Add(ProdOpPattern().Build(context));
  ps.Add<ReshapeOpPattern>(context);
  ps.Add<Pool2dOpPattern>(context);
  ps.Add<ConcatOpPattern>(context);
  ps.Add<SliceOpPattern>(context);
  ps.Add<PowOpPattern>(context);
  ps.Add<SplitWithNumOpPattern>(context);
  ps.Add<AddNOpPattern>(context);
  ps.Add<SplitOpPattern>(context);
  ps.Add<ExpandOpPattern>(context);
  ps.Add<IsCloseOpPattern>(context);
  // ps.Add(UniformOpPattern().Build(context));

  return ps;
}

bool PdOpToCinnOpPass::CanApplyOn(pir::Operation *op) const {
  return op->isa<pir::ModuleOp>() && op->num_regions() > 0;
}

std::unique_ptr<pir::Pass> CreatePdOpToCinnOpPass() {
  return std::make_unique<PdOpToCinnOpPass>();
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
