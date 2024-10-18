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

#include "paddle/fluid/pir/transforms/gpu/tensor_fusion_pass.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

class CastTensorFusionPattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "CastTensorFusionPattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &cast_op1 = pat.Op(paddle::dialect::CastOp::name());
    const auto &cast_op2 = pat.Op(paddle::dialect::CastOp::name());
    pat.Tensor("cast_out1") = cast_op1(pat.Tensor("x1"));
    pat.Tensor("cast_out2") = cast_op2(pat.Tensor("x2"));

    paddle::drr::ResultPattern res = pat.ResultPattern();
    const auto &x_dtype = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> phi::DataType {
          auto x_dtype = pir::GetDataTypeFromValue(match_ctx.Tensor("x1"));
          return paddle::dialect::TransToPhiDataType(x_dtype);
        });

    const auto &coalesce_tensor_op =
        res.Op(paddle::dialect::CoalesceTensorOp::name(),
               {{"dtype", x_dtype},
                {"copy_data", res.BoolAttr(false)},
                {"set_constant", res.BoolAttr(false)},
                {"persist_output", res.BoolAttr(false)},
                {"constant", res.Float32Attr(0.0)},
                {"use_align", res.BoolAttr(true)},
                {"align_size", res.Int32Attr(-1)},
                {"size_of_dtype", res.Int32Attr(-1)},
                {"concated_shapes", res.VectorInt64Attr({})},
                {"concated_ranks", res.VectorInt64Attr({})}});

    const auto &cast_op = res.Op(paddle::dialect::MatmulOp::name());
    const auto &split_op =
        res.Op(paddle::dialect::SplitOp::name(), {{"axis", res.Int32Attr(1)}});
    const auto &combine = res.Op("builtin.combine");
    combine({&res.Tensor("x1"), &res.Tensor("x2")},
            {&res.Tensor("combine_out")});
    res.Tensor("fused_x") = coalesce_tensor_op(res.Tensor("combine_out"));
    res.Tensor("fuesd_out") = cast_op(res.Tensor("fused_x"));
    res.Tensor("split_outs") = split_op(res.Tensor("fuesd_out"));
  }
};

class SplitConcatFusionPattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "SplitConcatFusionPattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &slice_op = pat.Op(paddle::dialect::SliceOp::name());
    const auto &combine = pat.Op("builtin.combine");
    const auto &coalesce_tensor_op =
        pat.Op(paddle::dialect::CoalesceTensorOp::name());
    const auto &split_with_num_op =
        pat.Op(paddle::dialect::SplitWithNumOp::name());

    pat.Tensor("split_outs") = split_with_num_op(pat.Tensor("fused_x"));
    slice_op({&pat.Tensor("split_outs")},
             {&pat.Tensor("slice_out1"), &pat.Tensor("slice_out2")});
    combine({&pat.Tensor("slice_out1"), &pat.Tensor("slice_out2")},
            {&pat.Tensor("combine_out")});
    coalesce_tensor_op({&pat.Tensor("combine_out")},
                       {&pat.Tensor("vec_x"), &pat.Tensor("fused_y")});
    paddle::drr::ResultPattern res = pat.ResultPattern();
    res.Tensor("fused_y").Assign(res.Tensor("fused_x"));
  }
};

class MatmulTensorFusionPattern : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "MatmulTensorFusionPattern"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &matmul_op_1 =
        pat.Op(paddle::dialect::MatmulOp::name(),
               {{"transpose_x", pat.Attr("matmul_1_transpose_x")},
                {"transpose_y", pat.Attr("matmul_1_transpose_y")}});
    const auto &matmul_op_2 =
        pat.Op(paddle::dialect::MatmulOp::name(),
               {{"transpose_x", pat.Attr("matmul_1_transpose_x")},
                {"transpose_y", pat.Attr("matmul_1_transpose_y")}});

    const auto &matmul_grad_op_1 =
        pat.Op(paddle::dialect::MatmulGradOp::name(),
               {{"transpose_x", pat.Attr("matmul_1_transpose_x")},
                {"transpose_y", pat.Attr("matmul_1_transpose_y")}});
    const auto &matmul_grad_op_2 =
        pat.Op(paddle::dialect::MatmulGradOp::name(),
               {{"transpose_x", pat.Attr("matmul_1_transpose_x")},
                {"transpose_y", pat.Attr("matmul_1_transpose_y")}});

    pat.Tensor("mut_out1") = matmul_op_1(pat.Tensor("w"), pat.Tensor("x1"));
    pat.Tensor("mut_out2") = matmul_op_2(pat.Tensor("w"), pat.Tensor("x2"));

    pat.Tensor("mut_grad_out1") = matmul_grad_op_1(
        pat.Tensor("w"), pat.Tensor("x1"), pat.Tensor("mut_grad_op1_in"));
    pat.Tensor("mut_grad_out2") = matmul_grad_op_2(
        pat.Tensor("w"), pat.Tensor("x2"), pat.Tensor("mut_grad_op2_in"));

    paddle::drr::ResultPattern res = pat.ResultPattern();
    const auto &x_dtype = res.ComputeAttr(
        [](const paddle::drr::MatchContext &match_ctx) -> phi::DataType {
          auto x_dtype = pir::GetDataTypeFromValue(match_ctx.Tensor("x1"));
          return paddle::dialect::TransToPhiDataType(x_dtype);
        });

    // const auto &sections_1 = res.ComputeAttr(
    //     [](const paddle::drr::MatchContext &match_ctx) ->
    //     std::vector<int32_t> {
    //       auto x1_shape = pir::GetShapeFromValue(match_ctx.Tensor("x1"));
    //       auto x2_shape = pir::GetShapeFromValue(match_ctx.Tensor("x2"));
    //       return {static_cast<int>(x1_shape.at(1)),
    //       static_cast<int>(x2_shape.at(1))};
    //     });

    // const auto &sections_2 = res.ComputeAttr(
    //     [](const paddle::drr::MatchContext &match_ctx) ->
    //     std::vector<int32_t> {
    //       auto x1_shape =
    //       pir::GetShapeFromValue(match_ctx.Tensor("mut_grad_op1_in")); auto
    //       x2_shape =
    //       pir::GetShapeFromValue(match_ctx.Tensor("mut_grad_op2_in")); return
    //       {static_cast<int>(x1_shape.at(1)),
    //       static_cast<int>(x2_shape.at(1))};
    //     });

    const auto &coalesce_tensor_op_1 =
        res.Op(paddle::dialect::CoalesceTensorOp::name(),
               {{"dtype", x_dtype},
                {"copy_data", res.BoolAttr(false)},
                {"set_constant", res.BoolAttr(false)},
                {"persist_output", res.BoolAttr(false)},
                {"constant", res.Float32Attr(0.0)},
                {"use_align", res.BoolAttr(true)},
                {"align_size", res.Int32Attr(-1)},
                {"size_of_dtype", res.Int32Attr(-1)},
                {"concated_shapes", res.VectorInt64Attr({})},
                {"concated_ranks", res.VectorInt64Attr({})}});

    // const auto &coalesce_tensor_op_2 =
    //     res.Op(paddle::dialect::CoalesceTensorOp::name(),
    //            {{"dtype", x_dtype},
    //             {"copy_data", res.BoolAttr(false)},
    //             {"set_constant", res.BoolAttr(false)},
    //             {"persist_output", res.BoolAttr(false)},
    //             {"constant", res.Float32Attr(0.0)},
    //             {"use_align", res.BoolAttr(true)},
    //             {"align_size", res.Int32Attr(-1)},
    //             {"size_of_dtype", res.Int32Attr(-1)},
    //             {"concated_shapes", res.VectorInt64Attr({})},
    //             {"concated_ranks", res.VectorInt64Attr({})}});

    // const auto &coalesce_tensor_op_3 =
    //     res.Op(paddle::dialect::CoalesceTensorOp::name(),
    //            {{"dtype", x_dtype},
    //             {"copy_data", res.BoolAttr(false)},
    //             {"set_constant", res.BoolAttr(false)},
    //             {"persist_output", res.BoolAttr(false)},
    //             {"constant", res.Float32Attr(0.0)},
    //             {"use_align", res.BoolAttr(true)},
    //             {"align_size", res.Int32Attr(-1)},
    //             {"size_of_dtype", res.Int32Attr(-1)},
    //             {"concated_shapes", res.VectorInt64Attr({})},
    //             {"concated_ranks", res.VectorInt64Attr({})}});

    const auto &matmul_op =
        res.Op(paddle::dialect::MatmulOp::name(),
               {{"transpose_x", pat.Attr("matmul_1_transpose_x")},
                {"transpose_y", pat.Attr("matmul_1_transpose_y")}});
    // const auto &matmul_grad_op =
    //     res.Op(paddle::dialect::MatmulGradOp::name(),
    //            {{"transpose_x", pat.Attr("matmul_1_transpose_x")},
    //             {"transpose_y", pat.Attr("matmul_1_transpose_y")}});

    // const auto &split_op_1 =
    //     res.Op(paddle::dialect::SplitOp::name(),
    //            {{"axis", res.Int32Attr(1)}, {"sections", sections_1}});
    // const auto &split_op_2 =
    //     res.Op(paddle::dialect::SplitOp::name(),
    //            {{"axis", res.Int32Attr(1)}, {"sections", sections_2}});

    // const auto &slice_op_1 = res.Op(paddle::dialect::SliceOp::name());
    const auto &combine_1 = res.Op("builtin.combine");

    // const auto &slice_op_2 = res.Op(paddle::dialect::SliceOp::name());
    // const auto &combine_2 = res.Op("builtin.combine");

    // const auto &combine_3 = res.Op("builtin.combine");

    combine_1({&res.Tensor("x1"), &res.Tensor("x2")},
              {&res.Tensor("combine_out_1")});
    coalesce_tensor_op_1({&res.Tensor("combine_out_1")},
                         {&res.Tensor("vec_x_1"), &res.Tensor("fused_x")});
    res.Tensor("fuesd_fwd_out_1") =
        matmul_op(res.Tensor("w"), res.Tensor("fused_x"));
    // res.Tensor("split_outs") = split_op_1(res.Tensor("fuesd_fwd_out_1"));
    // slice_op_1({&res.Tensor("split_outs")},
    //            {&res.Tensor("slice_out1"), &res.Tensor("slice_out2")});

    // combine_2({&res.Tensor("mut_grad_op1_in"),
    // &res.Tensor("mut_grad_op2_in")},
    //           {&res.Tensor("combine_out_2")});
    // coalesce_tensor_op_2({&res.Tensor("combine_out_2")},
    //                      {&res.Tensor("vec_mut_grad"),
    //                      &res.Tensor("fused_mut_grad")});

    // combine_3({&res.Tensor("slice_out1"), &res.Tensor("slice_out2")},
    //           {&res.Tensor("combine_out_3")});

    // coalesce_tensor_op_3({&res.Tensor("combine_out_3")},
    //                      {&res.Tensor("vec_fwd_out"),
    //                      &res.Tensor("fused_fwd_out_2")});

    // res.Tensor("fuesd_bwd_out") = matmul_grad_op(
    //     res.Tensor("w"), res.Tensor("fused_fwd_out_2"),
    //     res.Tensor("fused_mut_grad"));

    // res.Tensor("split_outs") = split_op_2(res.Tensor("fuesd_bwd_out"));
    // slice_op_2({&res.Tensor("split_outs")},
    //            {&res.Tensor("slice_out3"), &res.Tensor("slice_out4")});
  }
};

class TensorFusionPass : public pir::PatternRewritePass {
 public:
  TensorFusionPass() : pir::PatternRewritePass("tensor_fusion_pass", 2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<CastTensorFusionPattern>(context));
    ps.Add(paddle::drr::Create<MatmulTensorFusionPattern>(context));
    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateTensorFusionPass() {
  return std::make_unique<TensorFusionPass>();
}

}  // namespace pir

REGISTER_IR_PASS(tensor_fusion_pass, TensorFusionPass);
