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

#pragma once

#include <vector>

#include "paddle/ir/pattern_rewrite/drr/api/drr_pattern_context.h"
#include "paddle/ir/pattern_rewrite/drr/ir_operation_creator.h"
#include "paddle/ir/pattern_rewrite/drr/match_context_impl.h"
#include "paddle/ir/pattern_rewrite/drr/pattern_graph.h"
#include "paddle/ir/pattern_rewrite/pattern_match.h"

namespace ir {
namespace drr {

template <typename SourceOp, typename DrrFunctor>
struct DrrRewritePattern : public ir::OpRewritePattern<SourceOp> {
  DrrRewritePattern(ir::IrContext* context, ir::PatternBenefit benefit)
      : ir::OpRewritePattern<SourceOp>(context, benefit) {
    DrrPatternContext drr_context;
    DrrFunctor functor;
    functor(&drr_context);

    source_pattern_graph_ = drr_context.source_pattern_graph();
    constraints_ = drr_context.constraints();
    result_pattern_graph_ = drr_context.result_pattern_graph();

    source_pattern_graph_->Print();
    result_pattern_graph_->Print();
  }

  bool Match(SourceOp op) const override {
    // Match

    return true;
  }

  void Rewrite(SourceOp op,
               ir::PatternRewriter& rewriter) const override {  // NOLINT
    // 1. Create Operations in result_pattern_graph
    CreateOperations(*result_pattern_graph_,
                     rewriter,
                     source_pattern_match_ctx_,
                     result_pattern_match_ctx_);

    // 2. Replace Output Values in source_pattern_graph by Output Values in
    // result_pattern_graph
    ReplaceOutputTensor(
        source_pattern_match_ctx_, result_pattern_match_ctx_, rewriter);

    // 3. Delete Operations in source_pattern_graph
    DeleteSourcePatternOp(source_pattern_match_ctx_, rewriter);
  }

  void CreateOperations(const ResultPatternGraph& result_pattern_graph,
                        ir::PatternRewriter& rewriter,  // NOLINT
                        const MatchContextImpl& src_match_ctx,
                        MatchContextImpl* res_match_ctx) {
    // add input tensors info for result_pattern_match_ctx;
    const auto& input_tensors = result_pattern_graph.input_tensors();
    for (const auto& in_tensor : input_tensors) {
      res_match_ctx->BindIrTensor(
          std::make_shared<IrTensor>(src_match_ctx.GetIrTensor(in_tensor)));
    }

    // topo order visit result_pattern_graph
    GraphTopo graph_topo_visit(&result_pattern_graph);
    graph_topo_visit.WalkGraphNodesTopoOrder(
        [&rewriter, &src_match_ctx](const OpCall& op_call) {
          CreateOperation(op_call, rewriter, src_match_ctx);
        });
  }

  void ReplaceOutputTensor(const MatchContextImpl& src_match_ctx,
                           const MatchContextImpl& res_match_ctx,
                           ir::PatternRewriter& rewriter) const {  // NOLINT
    for (const auto& output_name : source_pattern_graph_->output_tensors()) {
      const auto& src_ir_tensor = src_match_ctx.GetIrTensor(output_name);
      const auto& res_ir_tensor = res_match_ctx.GetIrTensor(output_name);
      rewriter.ReplaceAllUsesWith(*src_ir_tensor.ir_value(),
                                  *res_ir_tensor.ir_value())
    }
  }

  void DeleteSourcePatternOp(const MatchContextImpl& src_match_ctx,
                             ir::PatternRewriter& rewriter) const {  // NOLINT
    for (const auto& kv : src_match_ctx.op_map()) {
      rewriter.EraseOp(kv.second);
    }
  }

 private:
  std::shared_ptr<SourcePatternGraph> source_pattern_graph_;
  std::vector<Constraint> constraints_;
  std::shared_ptr<ResultPatternGraph> result_pattern_graph_;

  std::shared_ptr<MatchContextImpl> source_pattern_match_ctx_;
  std::shared_ptr<MatchContextImpl> result_pattern_match_ctx_;
};

}  // namespace drr
}  // namespace ir
