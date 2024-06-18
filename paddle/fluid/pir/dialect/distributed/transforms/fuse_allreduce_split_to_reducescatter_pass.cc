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
// c_allreduce_sum_+assign+full+split_with_num+builtin_slice -> c_reducescatter
class FusedAllReduceSplitPattern1 : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "FusedAllReduceSplitPattern1"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    const auto &c_allreduce_sum_ =
        pat.Op(paddle::dialect::CAllreduceSum_Op::name(),
               {{"ring_id", pat.Attr("ring_id")},
                {"use_calc_stream", pat.Attr("use_calc_stream")},
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

    pat.Tensor("input_grad") =
        c_allreduce_sum_(pat.Tensor("input_grad_partial"));
    pat.Tensor("input_grad_tmp") = assign(pat.Tensor("input_grad"));
    pat.Tensor("split_num") = full();
    pat.Tensor("input_grad_group") =
        split_with_num(pat.Tensor("input_grad_tmp"), pat.Tensor("split_num"));
    pat.Tensor("out") = builtin_slice(pat.Tensor("input_grad_group"));

    pat.AddConstraint([&](const paddle::drr::MatchContext &match_ctx) {
      auto input_grad_partial_count =
          match_ctx.Tensor("input_grad_partial").use_count();
      auto input_grad_count = match_ctx.Tensor("input_grad").use_count();
      auto input_grad_tmp_count =
          match_ctx.Tensor("input_grad_tmp").use_count();
      auto input_grad_group_count =
          match_ctx.Tensor("input_grad_group").use_count();
      return (input_grad_partial_count == 1 && input_grad_count == 1 &&
              input_grad_tmp_count == 1 && input_grad_group_count == 1);
    });

    paddle::drr::ResultPattern res = pat.ResultPattern();

    const auto &c_reducescatter =
        res.Op(paddle::dialect::CReducescatterOp::name(),
               {{"ring_id", pat.Attr("ring_id")},
                {"nranks", pat.Attr("num")},
                {"use_calc_stream", pat.Attr("use_calc_stream")}},
               {{"execution_stream", pat.Attr("execution_stream")},
                {"force_record_event", pat.Attr("force_record_event")},
                {"event_to_record", pat.Attr("event_to_record")},
                {"events_to_wait", pat.Attr("events_to_wait")}});

    c_reducescatter({&res.Tensor("input_grad_partial")}, {&res.Tensor("out")});
  }
};

// c_allreduce_sum_+add+full+split_with_num+builtin_slice+assign ->
// c_reducescatter+add
class FusedAllReduceSplitPattern2 : public paddle::drr::DrrPatternBase {
 public:
  std::string name() const override { return "FusedAllReduceSplitPattern2"; }

  void operator()(paddle::drr::DrrPatternContext *ctx) const override {
    paddle::drr::SourcePattern pat = ctx->SourcePattern();

    const auto &c_allreduce_sum_ =
        pat.Op(paddle::dialect::CAllreduceSum_Op::name(),
               {{"ring_id", pat.Attr("ring_id")},
                {"use_calc_stream", pat.Attr("use_calc_stream")},
                {"execution_stream", pat.Attr("execution_stream")},
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
    // const auto &add_grad = pat.Op(paddle::dialect::AddGradOp::name());

    pat.Tensor("b") = c_allreduce_sum_(pat.Tensor("a"));
    pat.Tensor("d") = add(pat.Tensor("b"), pat.Tensor("c"));
    pat.Tensor("e") = full();
    pat.Tensor("f") = split_with_num(pat.Tensor("d"), pat.Tensor("e"));
    pat.Tensor("g") = builtin_slice(pat.Tensor("f"));
    pat.Tensor("h") = assign(pat.Tensor("g"));
    // add_grad({&pat.Tensor("b"), &pat.Tensor("c"), &pat.Tensor("grad")},
    //          {&pat.Tensor("b_g"), &pat.Tensor("c_g")});

    paddle::drr::ResultPattern res = pat.ResultPattern();

    const auto &c_reducescatter =
        res.Op(paddle::dialect::CReducescatterOp::name(),
               {{"ring_id", pat.Attr("ring_id")},
                {"nranks", pat.Attr("num")},
                {"use_calc_stream", pat.Attr("use_calc_stream")}},
               {{"execution_stream", pat.Attr("execution_stream")},
                {"force_record_event", pat.Attr("force_record_event")},
                {"event_to_record", pat.Attr("event_to_record")},
                {"events_to_wait", pat.Attr("events_to_wait")}});
    const auto &add1 = res.Op(paddle::dialect::AddOp::name());
    // const auto &add_grad1 = res.Op(paddle::dialect::AddGradOp::name());

    c_reducescatter({&res.Tensor("a")}, {&res.Tensor("b")});
    add1({&res.Tensor("b"), &res.Tensor("c")}, {&res.Tensor("h")});
    // add_grad1({&res.Tensor("b"), &res.Tensor("c"), &res.Tensor("grad")},
    //           {&res.Tensor("b_g"), &res.Tensor("c_g")});
  }
};

class FuseAllreduceSplitToReducescatterPass : public pir::PatternRewritePass {
 public:
  FuseAllreduceSplitToReducescatterPass()
      : pir::PatternRewritePass("fuse_allreduce_split_to_reducescatter_pass",
                                2) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext *context) override {
    pir::RewritePatternSet ps(context);
    // ps.Add(paddle::drr::Create<FusedAllReduceSplitPattern1>(context));
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
