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
#include "paddle/fluid/pir/drr/api/drr_pattern_base.h"
#include "paddle/fluid/pir/drr/api/match_context.h"
#include "paddle/pir/core/builtin_dialect.h"
#include "paddle/pir/core/builtin_op.h"
#include "paddle/pir/pass/pass.h"
#include "paddle/pir/pass/pass_manager.h"
#include "paddle/pir/pattern_rewrite/pattern_rewrite_driver.h"

namespace cinn {
namespace dialect {
namespace ir {

class SumOpPattern : public pir::drr::DrrPatternBase<SumOpPattern> {
 public:
  void operator()(pir::drr::DrrPatternContext *ctx) const override {
    // Source Pattern
    pir::drr::SourcePattern pattern = ctx->SourcePattern();
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
    pir::drr::ResultPattern res = pattern.ResultPattern();
    const auto &cinn_reduce_sum =
        res.Op(cinn::dialect::ReduceSumOp::name(),
               {{"dim", pattern.Attr("axis_info")},
                {"keep_dim", pattern.Attr("keep_dim")}});
    res.Tensor("ret") = cinn_reduce_sum(res.Tensor("arg0"));
  }
};

class MaxOpPattern : public pir::drr::DrrPatternBase<MaxOpPattern> {
 public:
  void operator()(pir::drr::DrrPatternContext *ctx) const override {
    // Source Pattern
    pir::drr::SourcePattern pattern = ctx->SourcePattern();
    const auto &full_int_array =
        pattern.Op(paddle::dialect::FullIntArrayOp::name(),
                   {{"value", pattern.Attr("axis_info")},
                    {"dtype", pattern.Attr("dtype_2")},
                    {"place", pattern.Attr("place_2")}});

    const auto &pd_max = pattern.Op(paddle::dialect::MaxOp::name(),
                                    {{"keepdim", pattern.Attr("keep_dim")}});
    pattern.Tensor("ret") = pd_max(pattern.Tensor("arg0"), full_int_array());

    // Result patterns
    pir::drr::ResultPattern res = pattern.ResultPattern();
    const auto &cinn_reduce_max =
        res.Op(cinn::dialect::ReduceMaxOp::name(),
               {{"dim", pattern.Attr("axis_info")},
                {"keep_dim", pattern.Attr("keep_dim")}});
    res.Tensor("ret") = cinn_reduce_max(res.Tensor("arg0"));
  }
};

class MinOpPattern : public pir::drr::DrrPatternBase<MinOpPattern> {
 public:
  void operator()(pir::drr::DrrPatternContext *ctx) const override {
    // Source Pattern
    pir::drr::SourcePattern pattern = ctx->SourcePattern();
    const auto &full_int_array =
        pattern.Op(paddle::dialect::FullIntArrayOp::name(),
                   {{"value", pattern.Attr("axis_info")},
                    {"dtype", pattern.Attr("dtype_2")},
                    {"place", pattern.Attr("place_2")}});

    const auto &pd_max = pattern.Op(paddle::dialect::MinOp::name(),
                                    {{"keepdim", pattern.Attr("keep_dim")}});
    pattern.Tensor("ret") = pd_max(pattern.Tensor("arg0"), full_int_array());

    // Result patterns
    pir::drr::ResultPattern res = pattern.ResultPattern();
    const auto &cinn_reduce_max =
        res.Op(cinn::dialect::ReduceMinOp::name(),
               {{"dim", pattern.Attr("axis_info")},
                {"keep_dim", pattern.Attr("keep_dim")}});
    res.Tensor("ret") = cinn_reduce_max(res.Tensor("arg0"));
  }
};

class ProdOpPattern : public pir::drr::DrrPatternBase<ProdOpPattern> {
 public:
  void operator()(pir::drr::DrrPatternContext *ctx) const override {
    // Source Pattern
    pir::drr::SourcePattern pattern = ctx->SourcePattern();
    const auto &full_int_array =
        pattern.Op(paddle::dialect::FullIntArrayOp::name(),
                   {{"value", pattern.Attr("axis_info")},
                    {"dtype", pattern.Attr("dtype_2")},
                    {"place", pattern.Attr("place_2")}});

    const auto &pd_max = pattern.Op(paddle::dialect::ProdOp::name(),
                                    {{"keepdim", pattern.Attr("keep_dim")}});
    pattern.Tensor("ret") = pd_max(pattern.Tensor("arg0"), full_int_array());

    // Result patterns
    pir::drr::ResultPattern res = pattern.ResultPattern();
    const auto &cinn_reduce_max =
        res.Op(cinn::dialect::ReduceProdOp::name(),
               {{"dim", pattern.Attr("axis_info")},
                {"keep_dim", pattern.Attr("keep_dim")}});
    res.Tensor("ret") = cinn_reduce_max(res.Tensor("arg0"));
  }
};

class ScaleOpPattern : public pir::OpRewritePattern<paddle::dialect::ScaleOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::ScaleOp>::OpRewritePattern;

  bool MatchAndRewrite(paddle::dialect::ScaleOp op,
                       pir::PatternRewriter &rewriter) const override {
    auto scale_factor_gen_op =
        op->operand_source(1).dyn_cast<pir::OpResult>().owner();

    if (auto full_op =
            scale_factor_gen_op->dyn_cast<paddle::dialect::FullOp>()) {
      // sacle is generator by full op
      // get attribute value from full op
      auto scale_value =
          full_op.attribute("value").dyn_cast<pir::FloatAttribute>().data();

      auto cinn_scale = rewriter.Build<cinn::dialect::ScaleOp>(
          op->operand_source(0).dyn_cast<pir::OpResult>(),
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

    return true;
  }
};

class ReshapeOpPattern
    : public pir::OpRewritePattern<paddle::dialect::ReshapeOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::ReshapeOp>::OpRewritePattern;

  bool MatchAndRewrite(paddle::dialect::ReshapeOp op,
                       pir::PatternRewriter &rewriter) const override {
    auto scale_factor_gen_op =
        op->operand_source(1).dyn_cast<pir::OpResult>().owner();

    if (auto full_op =
            scale_factor_gen_op->dyn_cast<paddle::dialect::FullIntArrayOp>()) {
      // sacle is generator by full op
      // get attribute value from full op

      auto out_shape_attr =
          full_op.attribute("value").dyn_cast<pir::ArrayAttribute>().AsVector();

      std::vector<int> vec_out_shape;
      if (out_shape_attr.size() > 0) {
        PADDLE_ENFORCE_EQ(
            out_shape_attr[0].isa<::pir::Int64Attribute>(),
            true,
            phi::errors::Unimplemented(
                "the 0th elementwise MUST be ir::Int64Attribute"));
        for (size_t i = 0; i < out_shape_attr.size(); ++i) {
          vec_out_shape.push_back(
              out_shape_attr[i].dyn_cast<::pir::Int64Attribute>().data());
        }
      }

      auto cinn_reshape = rewriter.Build<cinn::dialect::ReshapeOp>(
          op->operand_source(0).dyn_cast<pir::OpResult>(), vec_out_shape);
      rewriter.ReplaceAllUsesWith(op.result(0), cinn_reshape.result(0));
      rewriter.EraseOp(op);

      return true;
    }
    return false;
  }
};

class SliceOpPattern : public pir::OpRewritePattern<paddle::dialect::SliceOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::SliceOp>::OpRewritePattern;

  bool MatchAndRewrite(paddle::dialect::SliceOp op,
                       pir::PatternRewriter &rewriter) const override {
    auto start_gen_op = op->operand_source(1)
                            .dyn_cast<pir::OpResult>()
                            .owner()
                            ->dyn_cast<paddle::dialect::FullIntArrayOp>();

    auto end_gen_op = op->operand_source(2)
                          .dyn_cast<pir::OpResult>()
                          .owner()
                          ->dyn_cast<paddle::dialect::FullIntArrayOp>();

    if (start_gen_op && end_gen_op) {
      // sacle is generator by full op
      // get attribute value from full op
      auto start_vec = cinn::dialect::ir::GetVectorAttr(start_gen_op, "value");
      auto end_vec = cinn::dialect::ir::GetVectorAttr(end_gen_op, "value");
      auto axes = cinn::dialect::ir::GetVectorAttr(op, "axes");
      auto decrease_axis =
          cinn::dialect::ir::GetVectorAttr(op, "decrease_axis");
      auto infer_flags = cinn::dialect::ir::GetVectorAttr(op, "infer_flags");

      auto cinn_slice = rewriter.Build<cinn::dialect::SliceOp>(
          op->operand_source(0).dyn_cast<pir::OpResult>(),
          axes,
          start_vec,
          end_vec,
          infer_flags,
          decrease_axis);
      rewriter.ReplaceAllUsesWith(op.result(0), cinn_slice.result(0));
      rewriter.EraseOp(op);

      return true;
    }
    return false;
  }
};

class ConcatOpPattern
    : public pir::OpRewritePattern<paddle::dialect::ConcatOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::ConcatOp>::OpRewritePattern;

  bool MatchAndRewrite(paddle::dialect::ConcatOp op,
                       pir::PatternRewriter &rewriter) const override {
    auto axis_gen_op = op->operand_source(1).dyn_cast<pir::OpResult>().owner();
    if (auto full_op = axis_gen_op->dyn_cast<paddle::dialect::FullOp>()) {
      int axis = phi::Scalar(full_op.attribute("value")
                                 .dyn_cast<::pir::FloatAttribute>()
                                 .data())
                     .to<int>();

      auto input_ops = op->operand_source(0)
                           .dyn_cast<pir::OpResult>()
                           .owner()
                           ->dyn_cast<pir::CombineOp>()
                           .inputs();

      auto cinn_concat =
          rewriter.Build<cinn::dialect::ConcatOp>(input_ops, axis);
      rewriter.ReplaceAllUsesWith(op.result(0), cinn_concat.result(0));
      rewriter.EraseOp(op);

      return true;
    }
    return false;
  }
};

class PowOpPattern : public pir::OpRewritePattern<paddle::dialect::PowOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::PowOp>::OpRewritePattern;

  bool MatchAndRewrite(paddle::dialect::PowOp op,
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

    return true;
  }
};

class SplitOpPattern : public pir::OpRewritePattern<paddle::dialect::SplitOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::SplitOp>::OpRewritePattern;

  bool MatchAndRewrite(paddle::dialect::SplitOp op,
                       pir::PatternRewriter &rewriter) const override {
    auto sections_gen_op = op->operand_source(1)
                               .dyn_cast<pir::OpResult>()
                               .owner()
                               ->dyn_cast<paddle::dialect::FullIntArrayOp>();
    auto axis_gen_op = op->operand_source(2)
                           .dyn_cast<pir::OpResult>()
                           .owner()
                           ->dyn_cast<paddle::dialect::FullOp>();
    if (sections_gen_op && axis_gen_op) {
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

      int axis = phi::Scalar(axis_gen_op.attribute("value")
                                 .dyn_cast<::pir::FloatAttribute>()
                                 .data())
                     .to<int>();

      auto input_ele = op->operand_source(0)
                           .type()
                           .dyn_cast<paddle::dialect::DenseTensorType>();
      if (axis < 0) {
        axis += input_ele.dims().size();
      }

      auto cinn_split = rewriter.Build<cinn::dialect::SplitOp>(
          op->operand_source(0), vec_sections, axis);

      auto build_split =
          op->result(0).first_use().owner()->dyn_cast<::pir::SplitOp>();

      for (size_t i = 0; i < build_split->num_results(); ++i) {
        rewriter.ReplaceAllUsesWith(build_split->result(i),
                                    cinn_split.result(i));
      }

      rewriter.EraseOp(build_split);

      rewriter.EraseOp(op);

      return true;
    }
    return false;
  }
};

class AddNOpPattern : public pir::OpRewritePattern<paddle::dialect::AddNOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::AddNOp>::OpRewritePattern;

  bool MatchAndRewrite(paddle::dialect::AddNOp op,
                       pir::PatternRewriter &rewriter) const override {
    auto combine_op = op->operand_source(0)
                          .dyn_cast<pir::OpResult>()
                          .owner()
                          ->dyn_cast<pir::CombineOp>();
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

  bool MatchAndRewrite(paddle::dialect::ExpandOp op,
                       pir::PatternRewriter &rewriter) const override {
    auto out_shape_gen_op = op->operand_source(1)
                                .dyn_cast<pir::OpResult>()
                                .owner()
                                ->dyn_cast<paddle::dialect::FullIntArrayOp>();

    if (out_shape_gen_op) {
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
      return true;
    }

    return false;
  }
};

class SplitWithNumOpPattern
    : public pir::OpRewritePattern<paddle::dialect::SplitWithNumOp> {
 public:
  using pir::OpRewritePattern<
      paddle::dialect::SplitWithNumOp>::OpRewritePattern;

  bool MatchAndRewrite(paddle::dialect::SplitWithNumOp op,
                       pir::PatternRewriter &rewriter) const override {
    auto axis_gen_op = op->operand_source(1).dyn_cast<pir::OpResult>().owner();
    if (auto full_op = axis_gen_op->dyn_cast<paddle::dialect::FullOp>()) {
      int axis = phi::Scalar(full_op.attribute("value")
                                 .dyn_cast<::pir::FloatAttribute>()
                                 .data())
                     .to<int>();

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

      int index = 0;
      auto orig_out = op.result(0);
      for (auto it = orig_out.use_begin(); it != orig_out.use_end();) {
        auto split_op = (it++)->owner();
        rewriter.ReplaceAllUsesWith(split_op->result(0),
                                    cinn_split.result(index++));
        rewriter.EraseOp(split_op);
      }

      rewriter.EraseOp(op);

      return true;
    }
    return false;
  }
};

class UniformOpPattern : public pir::drr::DrrPatternBase<UniformOpPattern> {
 public:
  void operator()(pir::drr::DrrPatternContext *ctx) const override {
    // Source Pattern
    pir::drr::SourcePattern pattern = ctx->SourcePattern();
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
    pir::drr::ResultPattern res = pattern.ResultPattern();
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
  ps.Add<ReshapeOpPattern>(context);
  ps.Add<ConcatOpPattern>(context);
  ps.Add<SliceOpPattern>(context);
  ps.Add<PowOpPattern>(context);
  ps.Add<SplitWithNumOpPattern>(context);
  ps.Add<AddNOpPattern>(context);
  ps.Add<SplitOpPattern>(context);
  ps.Add<ExpandOpPattern>(context);
  // ps.Add(UniformOpPattern().Build(context));

  return ps;
}

bool PdOpToCinnOpPass::CanApplyOn(pir::Operation *op) const {
  return op->isa<pir::ModuleOp>() && op->num_regions() > 0;
}

void PdOp2CinnOpConverter(::pir::Program *program) {
  pir::IrContext *ctx = pir::IrContext::Instance();

  pir::PassManager pm(ctx);
  pm.AddPass(std::make_unique<PdOpToCinnOpPass>());

  pm.Run(program);
}
}  // namespace ir
}  // namespace dialect
}  // namespace cinn
