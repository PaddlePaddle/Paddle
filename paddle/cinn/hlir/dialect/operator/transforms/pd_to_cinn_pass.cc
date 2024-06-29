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
#include "paddle/cinn/hlir/dialect/operator/transforms/refresh_combine_pattern.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_manager.h"
#include "paddle/pir/include/pattern_rewrite/pattern_rewrite_driver.h"

namespace cinn {
namespace dialect {
namespace ir {
using CompatibleInfo = cinn::hlir::framework::pir::CompatibleInfo;

namespace {

template <typename T = int>
std::vector<T> GetVectorFromIntArrayAttribute(
    const pir::ArrayAttribute &array_attr) {
  const auto &vector_attr = array_attr.AsVector();

  std::vector<T> result;
  if (vector_attr.size() > 0) {
    PADDLE_ENFORCE_EQ(vector_attr[0].isa<::pir::Int64Attribute>(),
                      true,
                      phi::errors::Unimplemented(
                          "the 0th elementwise MUST be ir::Int64Attribute"));
    for (size_t i = 0; i < vector_attr.size(); ++i) {
      result.push_back(vector_attr[i].dyn_cast<::pir::Int64Attribute>().data());
    }
  }
  return result;
}

template <typename OpT>
void ReplaceWithCinnReshapeOp(OpT op,
                              pir::PatternRewriter &rewriter,  // NOLINT
                              const std::vector<int> &out_shape) {
  PADDLE_ENFORCE_EQ(
      op->num_results(),
      2U,
      ::common::errors::PreconditionNotMet(
          "The size of source op outputs must be 2, but received %d.",
          op->num_results()));
  auto cinn_reshape = rewriter.Build<cinn::dialect::ReshapeOp>(
      op->operand_source(0), out_shape);
  auto generate_xshape =
      rewriter.Build<cinn::dialect::GenerateXShapeOp>(op->operand_source(0));
  rewriter.ReplaceAllUsesWith(op.result(0), cinn_reshape.result(0));
  rewriter.ReplaceAllUsesWith(op.result(1), generate_xshape.result(0));
}

}  // namespace

class SumOpPattern : public pir::OpRewritePattern<paddle::dialect::SumOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::SumOp>::OpRewritePattern;

  bool Match(paddle::dialect::SumOp op) const override {
    if (CompatibleInfo::IsDeniedForCinn(*op.operation())) return false;
    auto *axes_op = op->operand_source(1).defining_op();
    return axes_op && axes_op->isa<paddle::dialect::FullIntArrayOp>();
  }

  void Rewrite(paddle::dialect::SumOp op,
               pir::PatternRewriter &rewriter) const override {
    auto *axes_op = op->operand_source(1).defining_op();
    auto full_int_array_op =
        axes_op->dyn_cast<paddle::dialect::FullIntArrayOp>();

    // get attribute value from full_int_array op
    const std::vector<int64_t> axis = GetVectorFromIntArrayAttribute<int64_t>(
        full_int_array_op.attribute("value").dyn_cast<pir::ArrayAttribute>());
    const bool keepdim =
        op.attribute("keepdim").dyn_cast<::pir::BoolAttribute>().data();
    const auto &dtype = op.attribute("dtype")
                            .dyn_cast<paddle::dialect::DataTypeAttribute>()
                            .data();

    auto cinn_reduce = rewriter.Build<cinn::dialect::ReduceSumOp>(
        op->operand_source(0), axis, keepdim, dtype);
    rewriter.ReplaceAllUsesWith(op.result(0), cinn_reduce.result(0));
    rewriter.EraseOp(op);
    if (full_int_array_op->use_empty()) {
      rewriter.EraseOp(full_int_array_op);
    }
  }
};

template <typename SOURCE_OP, typename TARGET_OP>
class ReduceMinMaxOpPattern : public pir::OpRewritePattern<SOURCE_OP> {
 public:
  using pir::OpRewritePattern<SOURCE_OP>::OpRewritePattern;

  bool Match(SOURCE_OP op) const override {
    if (CompatibleInfo::IsDeniedForCinn(*op.operation())) return false;
    auto *axes_op = op->operand_source(1).defining_op();
    return axes_op && axes_op->template isa<paddle::dialect::FullIntArrayOp>();
  }

  void Rewrite(SOURCE_OP op, pir::PatternRewriter &rewriter) const override {
    auto *axes_op = op->operand_source(1).defining_op();
    auto full_int_array_op =
        axes_op->template dyn_cast<paddle::dialect::FullIntArrayOp>();

    // get attribute value from full_int_array op
    const std::vector<int64_t> axis = GetVectorFromIntArrayAttribute<int64_t>(
        full_int_array_op.attribute("value")
            .template dyn_cast<pir::ArrayAttribute>());
    const bool keepdim = op.attribute("keepdim")
                             .template dyn_cast<::pir::BoolAttribute>()
                             .data();

    auto cinn_reduce =
        rewriter.Build<TARGET_OP>(op->operand_source(0), axis, keepdim);
    rewriter.ReplaceAllUsesWith(op.result(0), cinn_reduce.result(0));
    rewriter.EraseOp(op);
    if (full_int_array_op->use_empty()) {
      rewriter.EraseOp(full_int_array_op);
    }
  }
};

class ProdOpPattern : public pir::OpRewritePattern<paddle::dialect::ProdOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::ProdOp>::OpRewritePattern;

  bool Match(paddle::dialect::ProdOp op) const override {
    if (CompatibleInfo::IsDeniedForCinn(*op.operation())) return false;
    auto *axes_op = op->operand_source(1).defining_op();
    return axes_op && axes_op->isa<paddle::dialect::FullIntArrayOp>();
  }

  void Rewrite(paddle::dialect::ProdOp op,
               pir::PatternRewriter &rewriter) const override {
    auto *axes_op = op->operand_source(1).defining_op();
    auto full_int_array_op =
        axes_op->dyn_cast<paddle::dialect::FullIntArrayOp>();

    // get attribute value from full_int_array op
    const std::vector<int64_t> axis = GetVectorFromIntArrayAttribute<int64_t>(
        full_int_array_op.attribute("value").dyn_cast<pir::ArrayAttribute>());
    const bool keepdim =
        op.attribute("keepdim").dyn_cast<::pir::BoolAttribute>().data();
    const bool reduce_all =
        op.attribute("reduce_all").dyn_cast<::pir::BoolAttribute>().data();

    auto cinn_reduce = rewriter.Build<cinn::dialect::ReduceProdOp>(
        op->operand_source(0), axis, keepdim, reduce_all);
    rewriter.ReplaceAllUsesWith(op.result(0), cinn_reduce.result(0));
    rewriter.EraseOp(op);
    if (full_int_array_op->use_empty()) {
      rewriter.EraseOp(full_int_array_op);
    }
  }
};

class ScaleOpPattern : public pir::OpRewritePattern<paddle::dialect::ScaleOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::ScaleOp>::OpRewritePattern;

  bool Match(paddle::dialect::ScaleOp op) const override {
    const bool is_denied = CompatibleInfo::IsDeniedForCinn(*op.operation());
    return !is_denied;
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
    const bool is_denied = CompatibleInfo::IsDeniedForCinn(*op.operation());
    auto scale_factor_gen_op = op->operand_source(1).defining_op();
    auto full_op =
        scale_factor_gen_op->dyn_cast<paddle::dialect::FullIntArrayOp>();
    return !is_denied && full_op;
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
    ReplaceWithCinnReshapeOp(op, rewriter, vec_out_shape);
    rewriter.EraseOp(op);
  }
};

class FlipOpPattern : public pir::OpRewritePattern<paddle::dialect::FlipOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::FlipOp>::OpRewritePattern;

  bool Match(paddle::dialect::FlipOp op) const override {
    const bool is_denied = CompatibleInfo::IsDeniedForCinn(*op.operation());
    return !is_denied;
  }

  void Rewrite(paddle::dialect::FlipOp op,
               pir::PatternRewriter &rewriter) const override {
    std::vector<int> axis_value;
    auto axis_attr =
        op.attribute("axis").dyn_cast<pir::ArrayAttribute>().AsVector();
    if (axis_attr.size() > 0) {
      for (size_t i = 0; i < axis_attr.size(); ++i) {
        PADDLE_ENFORCE(axis_attr[i].dyn_cast<::pir::Int32Attribute>(),
                       ::common::errors::PreconditionNotMet(
                           "Reqiured attr element must be Int32Attribute."));
        axis_value.push_back(
            axis_attr[i].dyn_cast<::pir::Int32Attribute>().data());
      }
    }
    auto cinn_reverse = rewriter.Build<cinn::dialect::ReverseOp>(
        op->operand_source(0), axis_value);
    rewriter.ReplaceAllUsesWith(op.result(0), cinn_reverse.result(0));
    rewriter.EraseOp(op);
  }
};
class Pool2dOpPattern
    : public pir::OpRewritePattern<paddle::dialect::Pool2dOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::Pool2dOp>::OpRewritePattern;

  bool Match(paddle::dialect::Pool2dOp op) const override {
    const bool is_denied = CompatibleInfo::IsDeniedForCinn(*op.operation());
    auto kernel_size_gen_op = op->operand_source(1).defining_op();
    auto full_op =
        kernel_size_gen_op->dyn_cast<paddle::dialect::FullIntArrayOp>();
    return !is_denied && full_op;
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
    attrs.erase("strides");
    attrs.erase("paddings");

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
    const bool is_denied = CompatibleInfo::IsDeniedForCinn(*op.operation());
    auto rtol_op = op->operand_source(2)
                       .defining_op()
                       ->dyn_cast<paddle::dialect::FullOp>();
    auto atol_op = op->operand_source(3)
                       .defining_op()
                       ->dyn_cast<paddle::dialect::FullOp>();
    return !is_denied && rtol_op && atol_op;
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
    const bool is_denied = CompatibleInfo::IsDeniedForCinn(*op.operation());
    auto start_gen_op = op->operand_source(1)
                            .defining_op()
                            ->dyn_cast<paddle::dialect::FullIntArrayOp>();

    auto end_gen_op = op->operand_source(2)
                          .defining_op()
                          ->dyn_cast<paddle::dialect::FullIntArrayOp>();
    return !is_denied && start_gen_op && end_gen_op;
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
    const bool is_denied = CompatibleInfo::IsDeniedForCinn(*op.operation());
    return !is_denied && PatternConstraint(op);
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

 private:
  bool PatternConstraint(paddle::dialect::ConcatOp op) const {
    const pir::Operation *inputs_gen_op = op->operand_source(0).defining_op();
    const pir::Operation *axis_gen_op = op->operand_source(1).defining_op();
    return axis_gen_op->isa<paddle::dialect::FullOp>() &&
           inputs_gen_op->isa<pir::CombineOp>();
  }
};

class PowOpPattern : public pir::OpRewritePattern<paddle::dialect::PowOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::PowOp>::OpRewritePattern;

  bool Match(paddle::dialect::PowOp op) const override {
    const bool is_denied = CompatibleInfo::IsDeniedForCinn(*op.operation());
    return !is_denied;
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

class ElementwisePowOpPattern
    : public pir::OpRewritePattern<paddle::dialect::ElementwisePowOp> {
 public:
  using pir::OpRewritePattern<
      paddle::dialect::ElementwisePowOp>::OpRewritePattern;

  bool Match(paddle::dialect::ElementwisePowOp op) const override {
    const bool is_denied = CompatibleInfo::IsDeniedForCinn(*op.operation());
    auto y_op = op->operand_source(1)
                    .defining_op()
                    ->dyn_cast<paddle::dialect::FullOp>();
    return !is_denied && y_op;
  }

  void Rewrite(paddle::dialect::ElementwisePowOp op,
               pir::PatternRewriter &rewriter) const override {
    auto y_op = op->operand_source(1)
                    .defining_op()
                    ->dyn_cast<paddle::dialect::FullOp>();
    auto factor =
        y_op.attribute("value").dyn_cast<::pir::FloatAttribute>().data();
    if (factor == 2.0) {
      auto multiply = rewriter.Build<paddle::dialect::MultiplyOp>(
          op->operand_source(0), op->operand_source(0));
      rewriter.ReplaceAllUsesWith(op.result(0), multiply.result(0));
      rewriter.EraseOp(op);
    } else if (factor == -0.5) {
      auto rsqrt =
          rewriter.Build<paddle::dialect::RsqrtOp>(op->operand_source(0));
      rewriter.ReplaceAllUsesWith(op.result(0), rsqrt.result(0));
      rewriter.EraseOp(op);
    } else if (factor == 0.5) {
      auto sqrt =
          rewriter.Build<paddle::dialect::SqrtOp>(op->operand_source(0));
      rewriter.ReplaceAllUsesWith(op.result(0), sqrt.result(0));
      rewriter.EraseOp(op);
    }
  }
};

class SplitOpPattern : public pir::OpRewritePattern<paddle::dialect::SplitOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::SplitOp>::OpRewritePattern;

  bool Match(paddle::dialect::SplitOp op) const override {
    const bool is_denied = CompatibleInfo::IsDeniedForCinn(*op.operation());
    return !is_denied && PatternConstraint(op);
  }

  void Rewrite(paddle::dialect::SplitOp op,
               pir::PatternRewriter &rewriter) const override {
    for (auto it = op.out().use_begin(); it != op.out().use_end();) {
      auto downstream_op = (it++)->owner();
      if (downstream_op->isa<::pir::SliceOp>()) {
        ReplaceSplitSliceBySlice(
            op, downstream_op->dyn_cast<::pir::SliceOp>(), rewriter);
      } else if (downstream_op->isa<::pir::SplitOp>()) {
        ReplaceSplitSplitBySlice(
            op, downstream_op->dyn_cast<::pir::SplitOp>(), rewriter);
      } else {
        CHECK(false) << "Currently only support pir::slice/split as downstream "
                        "op, but got: "
                     << downstream_op->name();
      }
    }
  }

 private:
  bool PatternConstraint(paddle::dialect::SplitOp op) const {
    const auto &OnlyUsedBySplitOrSlice = [&]() -> bool {
      for (auto it = op.out().use_begin(); it != op.out().use_end();) {
        const pir::Operation *downstream_op = (it++)->owner();
        if (!downstream_op->isa<::pir::SliceOp>() ||
            !downstream_op->isa<::pir::SplitOp>()) {
          return false;
        }
      }
      return true;
    };
    const pir::Operation *sections_gen_op = op->operand_source(1).defining_op();
    const pir::Operation *axis_gen_op = op->operand_source(2).defining_op();
    return sections_gen_op->isa<paddle::dialect::FullIntArrayOp>() &&
           axis_gen_op->isa<paddle::dialect::FullOp>() &&
           OnlyUsedBySplitOrSlice();
  }
  int GetAxis(paddle::dialect::SplitOp op) const {
    auto axis_gen_op = op->operand_source(2).defining_op();
    auto full_op = axis_gen_op->dyn_cast<paddle::dialect::FullOp>();
    int axis = static_cast<int>(
        full_op.attribute("value").dyn_cast<::pir::FloatAttribute>().data());
    if (axis < 0) {
      axis += op.x()
                  .type()
                  .dyn_cast<paddle::dialect::DenseTensorType>()
                  .dims()
                  .size();
    }
    return axis;
  }

  std::vector<int64_t> GetSections(paddle::dialect::SplitOp op) const {
    std::vector<int64_t> result;
    auto sections_gen_op = op->operand_source(1)
                               .defining_op()
                               ->dyn_cast<paddle::dialect::FullIntArrayOp>();
    auto section_attr =
        sections_gen_op.attribute<pir::ArrayAttribute>("value").AsVector();
    if (section_attr.size() > 0) {
      for (size_t i = 0; i < section_attr.size(); ++i) {
        result.push_back(
            section_attr[i].dyn_cast<::pir::Int64Attribute>().data());
      }
    }
    return result;
  }

  void ReplaceSplitSliceBySlice(
      paddle::dialect::SplitOp split,
      ::pir::SliceOp slice,
      pir::PatternRewriter &rewriter) const {  // NOLINT
    const int axis = GetAxis(split);
    const std::vector<int64_t> &sections = GetSections(split);
    const int index = slice->attribute<::pir::Int32Attribute>("index").data();
    int64_t start =
        std::accumulate(sections.begin(), sections.begin() + index, 0);
    int64_t end = start + sections[index];
    auto paddle_slice =
        rewriter.Build<paddle::dialect::SliceOp>(split.x(),
                                                 std::vector<int64_t>({axis}),
                                                 std::vector<int64_t>({start}),
                                                 std::vector<int64_t>({end}),
                                                 std::vector<int64_t>({}),
                                                 std::vector<int64_t>({}));

    rewriter.ReplaceAllUsesWith(slice->result(0), paddle_slice.result(0));
    rewriter.EraseOp(slice);
    if (split->use_empty()) {
      rewriter.EraseOp(split);
    }
  }

  void ReplaceSplitSplitBySlice(
      paddle::dialect::SplitOp split,
      ::pir::SplitOp pir_split,
      pir::PatternRewriter &rewriter) const {  // NOLINT
    const int axis = GetAxis(split);
    const std::vector<int64_t> &sections = GetSections(split);
    int64_t start = 0, end = 0;
    for (size_t i = 0; i < pir_split->num_results(); ++i) {
      start = end;
      end += sections.at(i);
      auto paddle_slice = rewriter.Build<paddle::dialect::SliceOp>(
          split.x(),
          std::vector<int64_t>({axis}),
          std::vector<int64_t>({start}),
          std::vector<int64_t>({end}),
          std::vector<int64_t>({}),
          std::vector<int64_t>({}));
      rewriter.ReplaceAllUsesWith(pir_split->result(i),
                                  paddle_slice->result(0));
    }
    rewriter.EraseOp(pir_split);
    if (split->use_empty()) {
      rewriter.EraseOp(split);
    }
  }
};

class SplitWithNumOpPattern
    : public pir::OpRewritePattern<paddle::dialect::SplitWithNumOp> {
 public:
  using pir::OpRewritePattern<
      paddle::dialect::SplitWithNumOp>::OpRewritePattern;

  bool Match(paddle::dialect::SplitWithNumOp op) const override {
    auto axis_gen_op = op->operand_source(1).defining_op();
    return axis_gen_op->isa<paddle::dialect::FullOp>();
  }

  void Rewrite(paddle::dialect::SplitWithNumOp op,
               pir::PatternRewriter &rewriter) const override {
    const int axis = GetAxis(op);
    const std::vector<int64_t> &sections = GetSections(op, axis);
    auto split_op =
        rewriter.Build<paddle::dialect::SplitOp>(op.x(), sections, axis);
    rewriter.ReplaceAllUsesWith(op.out(), split_op.out());
    rewriter.EraseOp(op);
  }

 protected:
  int GetAxis(paddle::dialect::SplitWithNumOp op) const {
    auto axis_gen_op = op->operand_source(1).defining_op();
    auto full_op = axis_gen_op->dyn_cast<paddle::dialect::FullOp>();
    int axis = static_cast<int>(
        full_op.attribute<::pir::FloatAttribute>("value").data());
    if (axis < 0) {
      axis += op.x()
                  .type()
                  .dyn_cast<paddle::dialect::DenseTensorType>()
                  .dims()
                  .size();
    }
    return axis;
  }

  std::vector<int64_t> GetSections(paddle::dialect::SplitWithNumOp op,
                                   int axis) const {
    std::vector<int64_t> result;
    auto split_dim =
        op.x().type().dyn_cast<paddle::dialect::DenseTensorType>().dims()[axis];
    auto split_num = op->attribute<::pir::Int32Attribute>("num").data();
    auto part_ele = (split_dim + split_num - 1) / split_num;
    int total_split_num = 0;
    for (int i = 0; i < split_num - 1; ++i) {
      result.push_back(part_ele);
      total_split_num += part_ele;
    }

    result.push_back(split_dim - total_split_num);
    return result;
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
    const bool is_denied = CompatibleInfo::IsDeniedForCinn(*op.operation());
    auto out_shape_gen_op = op->operand_source(1)
                                .defining_op()
                                ->dyn_cast<paddle::dialect::FullIntArrayOp>();
    return !is_denied && out_shape_gen_op;
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
  std::string name() const override { return "ProdOpPattern"; }

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
                {"diag_val", pattern.Attr("min_value")},
                {"place", pattern.Attr("uniform_place")}});
    res.Tensor("ret") = cinn_uniform();
  }
};

class FullWithTensorOpPattern
    : public pir::OpRewritePattern<paddle::dialect::FullWithTensorOp> {
 public:
  using pir::OpRewritePattern<
      paddle::dialect::FullWithTensorOp>::OpRewritePattern;

  bool MatchAndRewrite(paddle::dialect::FullWithTensorOp op,
                       pir::PatternRewriter &rewriter) const override {
    auto value = op->operand_source(0);
    auto shape = op->operand_source(1);

    if (paddle::dialect::TransToPhiDataType(
            value.type()
                .dyn_cast<paddle::dialect::DenseTensorType>()
                .dtype()) != op.attribute("dtype")
                                 .dyn_cast<paddle::dialect::DataTypeAttribute>()
                                 .data()) {
      value = rewriter
                  .Build<paddle::dialect::CastOp>(
                      value,
                      op.attribute("dtype")
                          .dyn_cast<paddle::dialect::DataTypeAttribute>()
                          .data())
                  .result(0);
    }

    auto out =
        rewriter.Build<paddle::dialect::ExpandOp>(value, shape).result(0);

    rewriter.ReplaceAllUsesWith(op.result(0), out);

    rewriter.EraseOp(op);

    return true;
  }
};

class SqueezeOpPattern
    : public pir::OpRewritePattern<paddle::dialect::SqueezeOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::SqueezeOp>::OpRewritePattern;

  bool MatchAndRewrite(paddle::dialect::SqueezeOp op,
                       pir::PatternRewriter &rewriter) const override {
    auto axis_full_op = op->operand_source(1)
                            .defining_op()
                            ->dyn_cast<paddle::dialect::FullIntArrayOp>();

    bool is_dyshape = op->operand_source(0)
                          .type()
                          .dyn_cast<pir::ShapedTypeInterface>()
                          .IsDynamicShape();
    if (axis_full_op && !is_dyshape) {
      auto axis_vec = cinn::dialect::ir::GetVectorAttr(axis_full_op, "value");
      std::set<int64_t> axis_set(axis_vec.begin(), axis_vec.end());

      auto in_shape =
          phi::vectorize(op.operand_source(0)
                             .type()
                             .dyn_cast<paddle::dialect::DenseTensorType>()
                             .dims());

      std::vector<int> output_shape;

      for (size_t i = 0; i < in_shape.size(); ++i) {
        if (!axis_set.count(i)) {
          output_shape.push_back(in_shape[i]);
        } else {
          PADDLE_ENFORCE_EQ(
              in_shape[i],
              1,
              phi::errors::PreconditionNotMet(
                  "sequeze dim MUST be 1, but recive axis [%d] is [%d]",
                  i,
                  in_shape[i]));
        }
      }

      ReplaceWithCinnReshapeOp(op, rewriter, output_shape);
      rewriter.EraseOp(op);

      return true;
    }

    return false;
  }
};

class UnsqueezeOpPattern
    : public pir::OpRewritePattern<paddle::dialect::UnsqueezeOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::UnsqueezeOp>::OpRewritePattern;

  bool MatchAndRewrite(paddle::dialect::UnsqueezeOp op,
                       pir::PatternRewriter &rewriter) const override {
    auto axis_full_op = op->operand_source(1)
                            .defining_op()
                            ->dyn_cast<paddle::dialect::FullIntArrayOp>();
    bool is_dyshape = op->operand_source(0)
                          .type()
                          .dyn_cast<pir::ShapedTypeInterface>()
                          .IsDynamicShape();
    if (axis_full_op && !is_dyshape) {
      auto axis_vec = cinn::dialect::ir::GetVectorAttr(axis_full_op, "value");
      std::set<int64_t> axis_set(axis_vec.begin(), axis_vec.end());

      auto in_shape =
          phi::vectorize(op.operand_source(0)
                             .type()
                             .dyn_cast<paddle::dialect::DenseTensorType>()
                             .dims());

      std::vector<int> output_shape;

      for (size_t i = 0; i < in_shape.size(); ++i) {
        output_shape.push_back(in_shape[i]);
        if (axis_set.count(i)) {
          output_shape.push_back(1);
        }
      }

      ReplaceWithCinnReshapeOp(op, rewriter, output_shape);
      rewriter.EraseOp(op);

      return true;
    }

    return false;
  }
};

class FlattenOpPattern
    : public pir::OpRewritePattern<paddle::dialect::FlattenOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::FlattenOp>::OpRewritePattern;

  bool Match(paddle::dialect::FlattenOp op) const override {
    const bool is_denied = CompatibleInfo::IsDeniedForCinn(*op.operation());

    bool is_dyshape = op->operand_source(0)
                          .type()
                          .dyn_cast<pir::ShapedTypeInterface>()
                          .IsDynamicShape();
    return !is_denied && is_dyshape;
  }

  void Rewrite(paddle::dialect::FlattenOp op,
               pir::PatternRewriter &rewriter) const override {
    int start_axis =
        op.attribute("start_axis").dyn_cast<::pir::Int32Attribute>().data();
    int end_axis =
        op.attribute("stop_axis").dyn_cast<::pir::Int32Attribute>().data();

    // build output shape
    std::vector<pir::Value> out_shape;
    auto x_rank = op->operand_source(0)
                      .type()
                      .dyn_cast<paddle::dialect::DenseTensorType>()
                      .dims()
                      .size();
    auto x_shape =
        rewriter.Build<paddle::dialect::ShapeOp>(op->operand_source(0))
            .result(0);
    for (size_t i = 0; i < x_rank;) {
      if (i == static_cast<size_t>(start_axis)) {
        auto new_single_dim =
            rewriter
                .Build<cinn::dialect::SliceOp>(x_shape,
                                               std::vector<int64_t>({0}),
                                               std::vector<int64_t>({i}),
                                               std::vector<int64_t>({i + 1}),
                                               std::vector<int64_t>({}),
                                               std::vector<int64_t>({}))
                .result(0);

        for (auto t = start_axis + 1; t <= end_axis; ++t) {
          auto dim_t =
              rewriter
                  .Build<cinn::dialect::SliceOp>(x_shape,
                                                 std::vector<int64_t>({0}),
                                                 std::vector<int64_t>({t}),
                                                 std::vector<int64_t>({t + 1}),
                                                 std::vector<int64_t>({}),
                                                 std::vector<int64_t>({}))
                  .result(0);
          new_single_dim =
              rewriter.Build<paddle::dialect::MultiplyOp>(new_single_dim, dim_t)
                  .result(0);
        }
        out_shape.push_back(new_single_dim);
        i = end_axis + 1;
      } else {
        auto t =
            rewriter
                .Build<cinn::dialect::SliceOp>(x_shape,
                                               std::vector<int64_t>({0}),
                                               std::vector<int64_t>({i}),
                                               std::vector<int64_t>({i + 1}),
                                               std::vector<int64_t>({}),
                                               std::vector<int64_t>({}))
                .result(0);
        out_shape.push_back(t);
        i++;
      }
    }

    auto new_shape =
        rewriter.Build<cinn::dialect::ConcatOp>(out_shape, -1).result(0);

    auto reshape_op = rewriter.Build<paddle::dialect::ReshapeOp>(
        op->operand_source(0), new_shape);

    reshape_op.result(0).set_type(op.result(0).type());

    rewriter.ReplaceAllUsesWith(op.result(0), reshape_op.result(0));
    rewriter.ReplaceAllUsesWith(op.result(1), reshape_op.result(1));

    rewriter.EraseOp(op);
  }
};

class SigmoidOpPattern
    : public pir::OpRewritePattern<paddle::dialect::SigmoidOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::SigmoidOp>::OpRewritePattern;
  bool Match(paddle::dialect::SigmoidOp op) const override {
    return !CompatibleInfo::IsDeniedForCinn(*op.operation());
  }

  void Rewrite(paddle::dialect::SigmoidOp op,
               pir::PatternRewriter &rewriter) const override {
    auto input_dtype = paddle::dialect::TransToPhiDataType(
        op->operand_source(0)
            .type()
            .dyn_cast<paddle::dialect::DenseTensorType>()
            .dtype());

    auto in = op->operand_source(0);
    bool need_cast = (input_dtype == phi::DataType::FLOAT16 ||
                      input_dtype == phi::DataType::BFLOAT16 ||
                      input_dtype == phi::DataType::UINT16);
    if (need_cast) {
      in = rewriter.Build<paddle::dialect::CastOp>(in, phi::DataType::FLOAT32)
               .result(0);
    }

    // 1 / ( 1 + exp(-x))
    auto one = rewriter
                   .Build<paddle::dialect::FullOp>(
                       std::vector<int64_t>({1}), 1.0, phi::DataType::FLOAT32)
                   .result(0);
    auto minus_x =
        rewriter.Build<paddle::dialect::ScaleOp>(in, -1.0, 0.0).result(0);
    auto exp = rewriter.Build<paddle::dialect::ExpOp>(minus_x).result(0);
    auto add_exp = rewriter.Build<paddle::dialect::AddOp>(one, exp).result(0);
    auto div =
        rewriter.Build<paddle::dialect::DivideOp>(one, add_exp).result(0);

    if (need_cast) {
      div = rewriter.Build<paddle::dialect::CastOp>(div, input_dtype).result(0);
    }

    rewriter.ReplaceAllUsesWith(op.result(0), div);
    rewriter.EraseOp(op);
  }
};

class GatherOpPattern
    : public pir::OpRewritePattern<paddle::dialect::GatherOp> {
 public:
  using pir::OpRewritePattern<paddle::dialect::GatherOp>::OpRewritePattern;

  bool Match(paddle::dialect::GatherOp op) const override {
    const bool is_denied = CompatibleInfo::IsDeniedForCinn(*op.operation());
    auto axis_gen_op = op->operand_source(2).defining_op();
    auto full_op = axis_gen_op->dyn_cast<paddle::dialect::FullOp>();
    return !is_denied && full_op;
  }

  void Rewrite(paddle::dialect::GatherOp op,
               pir::PatternRewriter &rewriter) const override {
    auto gather_op = op->dyn_cast<paddle::dialect::GatherOp>();
    auto x = op.operand_source(0);
    auto index = op->operand_source(1);
    const int axis = [&]() -> int {
      auto axis_gen_op = op.operand_source(2).defining_op();
      PADDLE_ENFORCE_EQ(axis_gen_op->isa<paddle::dialect::FullOp>(),
                        true,
                        ::phi::errors::InvalidArgument(
                            "Not Supported: The gather operator for CINN "
                            "only supports constant value"));
      auto full_op = axis_gen_op->dyn_cast<paddle::dialect::FullOp>();
      return static_cast<int>(
          full_op.attribute("value").dyn_cast<::pir::FloatAttribute>().data());
    }();
    auto out =
        rewriter.Build<cinn::dialect::GatherOp>(x, index, axis)->result(0);
    rewriter.ReplaceAllUsesWith(op->result(0), out);
    rewriter.EraseOp(op);
  }
};

PdOpToCinnOpPass::PdOpToCinnOpPass()
    : pir::PatternRewritePass("pd_to_cinn_pass", 1) {}

pir::RewritePatternSet PdOpToCinnOpPass::InitializePatterns(
    pir::IrContext *context) {
  pir::RewritePatternSet ps(context);
  ps.Add<ScaleOpPattern>(
      context);  // NOTE, scale op pattern should before AddBroadcastTo
  ps.Add<SumOpPattern>(context);
  ps.Add<ReduceMinMaxOpPattern<paddle::dialect::MinOp,
                               cinn::dialect::ReduceMinOp>>(context);
  ps.Add<ReduceMinMaxOpPattern<paddle::dialect::MaxOp,
                               cinn::dialect::ReduceMaxOp>>(context);
  ps.Add<ProdOpPattern>(context);
  ps.Add<ReshapeOpPattern>(context);
  ps.Add<PowOpPattern>(context);
  ps.Add<ConcatOpPattern>(context);
  ps.Add<SliceOpPattern>(context);
  ps.Add<AddNOpPattern>(context);
  ps.Add<SplitWithNumOpPattern>(context);
  ps.Add<SplitOpPattern>(context);
  ps.Add<ExpandOpPattern>(context);
  ps.Add<FlipOpPattern>(context);
  ps.Add<IsCloseOpPattern>(context);
  ps.Add<ElementwisePowOpPattern>(context);
  ps.Add<FullWithTensorOpPattern>(context);
  ps.Add<RefreshCombineOpPattern>(context);
  ps.Add<SqueezeOpPattern>(context);
  ps.Add<UnsqueezeOpPattern>(context);
  ps.Add<SigmoidOpPattern>(context);
  ps.Add<GatherOpPattern>(context);
  ps.Add<FlattenOpPattern>(context);

  return ps;
}

bool PdOpToCinnOpPass::CanApplyOn(pir::Operation *op) const {
  return op->num_regions() > 0;
}

std::unique_ptr<pir::Pass> CreatePdOpToCinnOpPass() {
  return std::make_unique<PdOpToCinnOpPass>();
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
