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

#include "paddle/fluid/pir/dialect/distributed/transforms/fuse_allreduce_split_to_reducescatter_pass.h"

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/drr/include/drr_pattern_base.h"
#include "paddle/fluid/pir/utils/general_functions.h"

#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {
// matmul+all_reduce_+assign+full+split_with_num+builtin_slice ->
// reduce_scatter
class FusedAllReduceSplitPattern1 : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "FusedAllReduceSplitPattern1"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    const auto &matmul = pat.Op(paddle::dialect::MatmulOp::name(),
                                {{"transpose_x", pat.Attr("trans_x")},
                                 {"transpose_y", pat.Attr("trans_y")}});
    const auto &all_reduce_ =
        pat.Op(paddle::dialect::AllReduce_Op::name(),
               {{"ring_id", pat.Attr("ring_id")},
                {"reduce_type", pat.Attr("reduce_type")},
                {"execution_stream", pat.Attr("execution_stream")},
                {"force_record_event", pat.Attr("force_record_event")},
                {"event_to_record", pat.Attr("event_to_record")},
                {"events_to_wait", pat.Attr("events_to_wait")}});
    const auto &assign = pat.Op(paddle::dialect::AssignOp::name());
    const auto &full = pat.Op(paddle::dialect::FullOp::name());
    const auto &split_with_num = pat.Op(paddle::dialect::SplitWithNumOp::name(),
                                        {{"num", pat.Attr("num")}});
    const auto &builtin_slice =
        pat.Op(pir::SliceOp::name(), {{"index", pat.Attr("index")}});

    pat.Tensor("input_grad_partial") =
        matmul(pat.Tensor("out_grad"), pat.Tensor("weight"));
    pat.Tensor("input_grad") = all_reduce_(pat.Tensor("input_grad_partial"));
    pat.Tensor("input_grad_tmp") = assign(pat.Tensor("input_grad"));
    pat.Tensor("split_num") = full();
    pat.Tensor("input_grad_group") =
        split_with_num(pat.Tensor("input_grad_tmp"), pat.Tensor("split_num"));
    pat.Tensor("out") = builtin_slice(pat.Tensor("input_grad_group"));

    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      const auto &x_trans = match_ctx.Attr<bool>("trans_x");
      const auto &y_trans = match_ctx.Attr<bool>("trans_y");
      auto input_grad_partial_count =
          match_ctx.Tensor("input_grad_partial").use_count();
      auto input_grad_count = match_ctx.Tensor("input_grad").use_count();
      auto input_grad_tmp_count =
          match_ctx.Tensor("input_grad_tmp").use_count();
      auto input_grad_group_count =
          match_ctx.Tensor("input_grad_group").use_count();
      return (x_trans == false && y_trans == true &&
              input_grad_partial_count == 1 && input_grad_count == 1 &&
              input_grad_tmp_count == 1 && input_grad_group_count == 1);
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();

    const auto &reduce_scatter =
        res.Op(paddle::dialect::ReduceScatterOp::name(),
               {{"ring_id", pat.Attr("ring_id")}, {"nranks", pat.Attr("num")}},
               {{"execution_stream", pat.Attr("execution_stream")},
                {"force_record_event", pat.Attr("force_record_event")},
                {"event_to_record", pat.Attr("event_to_record")},
                {"events_to_wait", pat.Attr("events_to_wait")}});

    reduce_scatter({&res.Tensor("input_grad_partial")}, {&res.Tensor("out")});
  }
};

//                          input_g  weight_g
//                             |--------|
//     input   weight     matmul_grad
//       |-------|-------------|
//    matmul                   |
//       |                     |--------|
//      out1                out2_g    bias_g
//       |                     |--------|
// all_reduce_                 add_grad
//       | _ _ _ _ _ _ _ _ _ _ _ _ _ _ _|
//       |/      |                      |
//      out2    bias                out_g_all
//       |-------|                      |
//      add          full          all_gather
//       |            |                 |
//      out3        index          out_g_assign
//       |------------|                 |
// split_with_num                    assign1
//       |                              |
//      out4                       out_g_assign
//       |                              |
//  builtin_slice                     out_g
//       |
//      out5
//       |
//     assign
//       |
//      out6
//
// --> fused to
//                          input_g  weight_g
//                             |--------|
//     input   weight     matmul_grad
//       |-------|-------------|
//    matmul                out_g_all
//       |                     |
//      out1               all_gather
//       |                     |
// reduce_scatter        out_g_assign  bias_g
//       |                     |
//      out2    bias       add_grad
//       |-------|-------------|
//      add                  out_g
//       |
//      out6
class FusedAllReduceSplitPattern2 : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "FusedAllReduceSplitPattern2"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();
    // forward
    // out1 = matmul(input, weight)
    const auto &matmul = pat.Op(paddle::dialect::MatmulOp::name(),
                                {{"transpose_x", pat.Attr("trans_x")},
                                 {"transpose_y", pat.Attr("trans_y")}});
    // out2 = all_reduce_(out1)
    const auto &all_reduce_ =
        pat.Op(paddle::dialect::AllReduce_Op::name(),
               {{"ring_id", pat.Attr("ring_id")},
                {"reduce_type", pat.Attr("reduce_type")},
                {"force_record_event", pat.Attr("force_record_event")},
                {"event_to_record", pat.Attr("event_to_record")},
                {"events_to_wait", pat.Attr("events_to_wait")}});
    const auto &add = pat.Op(paddle::dialect::AddOp::name());
    const auto &full = pat.Op(paddle::dialect::FullOp::name());
    const auto &split_with_num = pat.Op(paddle::dialect::SplitWithNumOp::name(),
                                        {{"num", pat.Attr("num")}});
    const auto &builtin_slice =
        pat.Op(pir::SliceOp::name(), {{"index", pat.Attr("index")}});
    const auto &assign = pat.Op(paddle::dialect::AssignOp::name());
    const auto &assign1 = pat.Op(paddle::dialect::AssignOp::name());
    const auto &all_gather = pat.Op(paddle::dialect::AllGatherOp::name(),
                                    {{"ring_id", pat.Attr("gather_ring_id")},
                                     {"nranks", pat.Attr("gather_nranks")}});
    const auto &add_grad = pat.Op(paddle::dialect::AddGradOp::name(),
                                  {{"axis", pat.Attr("axis")}});
    const auto &add_ = pat.Op(paddle::dialect::Add_Op::name());
    const auto &matmul_grad =
        pat.Op(paddle::dialect::MatmulGradOp::name(),
               {{"transpose_x", pat.Attr("mm_g_trans_x")},
                {"transpose_y", pat.Attr("mm_g_trans_y")}});

    pat.Tensor("out1") = matmul(pat.Tensor("input"), pat.Tensor("weight"));
    pat.Tensor("out2") = all_reduce_(pat.Tensor("out1"));
    pat.Tensor("out3") = add(pat.Tensor("out2"), pat.Tensor("bias"));
    pat.Tensor("index") = full();
    pat.Tensor("out4") =
        split_with_num(pat.Tensor("out3"), pat.Tensor("index"));
    pat.Tensor("out5") = builtin_slice(pat.Tensor("out4"));
    pat.Tensor("out6") = assign(pat.Tensor("out5"));

    pat.Tensor("out_g_assign") = assign1(pat.Tensor("out_g"));
    pat.Tensor("out_g_all") = all_gather(pat.Tensor("out_g_assign"));
    add_grad(
        {&pat.Tensor("out2"), &pat.Tensor("bias"), &pat.Tensor("out_g_all")},
        {&pat.Tensor("out2_g"), &pat.Tensor("bias_g")});
    pat.Tensor("bias_g_m2") =
        add_(pat.Tensor("bias_g_m1"), pat.Tensor("bias_g"));
    matmul_grad(
        {&pat.Tensor("input"), &pat.Tensor("weight"), &pat.Tensor("out2_g")},
        {&pat.Tensor("input_g"), &pat.Tensor("weight_g")});

    paddle::drr::ResultPattern res = pat.ResultPattern();
    const auto &res_matmul = res.Op(paddle::dialect::MatmulOp::name(),
                                    {{"transpose_x", pat.Attr("trans_x")},
                                     {"transpose_y", pat.Attr("trans_y")}});
    const auto &res_reduce_scatter =
        res.Op(paddle::dialect::ReduceScatterOp::name(),
               {{"ring_id", pat.Attr("ring_id")}, {"nranks", pat.Attr("num")}},
               {{"force_record_event", pat.Attr("force_record_event")},
                {"event_to_record", pat.Attr("event_to_record")},
                {"events_to_wait", pat.Attr("events_to_wait")}});
    const auto &res_add = res.Op(paddle::dialect::AddOp::name());
    const auto &res_add_grad = res.Op(paddle::dialect::AddGradOp::name(),
                                      {{"axis", pat.Attr("axis")}});
    const auto &res_add_ = res.Op(paddle::dialect::Add_Op::name());
    const auto &res_all_gather =
        res.Op(paddle::dialect::AllGatherOp::name(),
               {{"ring_id", pat.Attr("gather_ring_id")},
                {"nranks", pat.Attr("gather_nranks")}});
    const auto &res_matmul_grad =
        res.Op(paddle::dialect::MatmulGradOp::name(),
               {{"transpose_x", pat.Attr("mm_g_trans_x")},
                {"transpose_y", pat.Attr("mm_g_trans_y")}});

    res.Tensor("out1") = res_matmul(res.Tensor("input"), res.Tensor("weight"));
    res.Tensor("out2") = res_reduce_scatter(res.Tensor("out1"));
    res.Tensor("out6") = res_add(res.Tensor("out2"), res.Tensor("bias"));

    res_add_grad(
        {&res.Tensor("out2"), &res.Tensor("bias"), &res.Tensor("out_g")},
        {&res.Tensor("out_g_assign"), &res.Tensor("bias_g")});
    res.Tensor("bias_g_m2") =
        res_add_(res.Tensor("bias_g_m1"), res.Tensor("bias_g"));
    res.Tensor("out_g_all") = res_all_gather(res.Tensor("out_g_assign"));
    res_matmul_grad(
        {&res.Tensor("input"), &res.Tensor("weight"), &res.Tensor("out_g_all")},
        {&res.Tensor("input_g"), &res.Tensor("weight_g")});
  }
};

class FuseAllreduceSplitToReducescatterPass : public pir::PatternRewritePass {
 public:
  FuseAllreduceSplitToReducescatterPass()
      : pir::PatternRewritePass("fuse_allreduce_split_to_reducescatter_pass",
                                2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    ps.Add(paddle::drr::Create<FusedAllReduceSplitPattern1>(context));
    ps.Add(paddle::drr::Create<FusedAllReduceSplitPattern2>(context));

    return ps;
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateFuseAllreduceSplitToReducescatterPass() {
  return std::make_unique<FuseAllreduceSplitToReducescatterPass>();
}

}  // namespace pir

REGISTER_IR_PASS(fuse_allreduce_split_to_reducescatter_pass,
                 FuseAllreduceSplitToReducescatterPass);
