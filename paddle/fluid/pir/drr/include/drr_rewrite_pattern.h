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

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/pir/drr/include/drr_pattern_context.h"
#include "paddle/pir/include/pattern_rewrite/pattern_match.h"

namespace pir {
class IrContext;
}

namespace paddle {
namespace drr {

class OpCall;
class Constraint;
class DrrPatternBase;
class DrrPatternContext;
class MatchContextImpl;
class SourcePatternGraph;
class ResultPatternGraph;

class DrrRewritePattern : public pir::RewritePattern {
 public:
  DrrRewritePattern(const std::string& pattern_name,
                    const DrrPatternContext& drr_context,
                    pir::IrContext* context,
                    pir::PatternBenefit benefit,
                    std::shared_ptr<const DrrPatternBase> drr_pattern_owner);

  bool MatchAndRewrite(
      pir::Operation* op,
      pir::PatternRewriter& rewriter) const override;  // // NOLINT

 private:
  bool PatternGraphMatch(pir::Operation* op,
                         MatchContextImpl* source_pattern_match_ctx) const;

  std::unordered_map<const OpCall*, std::unordered_set<pir::Operation*>>
  FindCandidateIrOutputOp(pir::Operation* op,
                          const OpCall* anchor,
                          const SourcePatternGraph& source_pattern_graph) const;

  void DfsVisitor(
      const OpCall* drr_op,
      pir::Operation* ir_op,
      const std::unordered_set<const OpCall*>& drr_output_op_set,
      std::unordered_set<const OpCall*>* drr_visited_ops,
      std::unordered_map<const OpCall*, std::unordered_set<pir::Operation*>>*
          output_op_bind_map) const;

  bool MatchFromOutputToInput(
      std::unordered_map<const OpCall*, pir::Operation*> output_op_map,
      const SourcePatternGraph& source_pattern_graph,
      const std::shared_ptr<MatchContextImpl>& source_pattern_match_ctx) const;

  void PatternGraphRewrite(const MatchContextImpl& source_pattern_match_ctx,
                           pir::PatternRewriter& rewriter) const;  // NOLINT

 private:
  MatchContextImpl CreateOperations(
      const SourcePatternGraph& source_pattern_graph,
      const ResultPatternGraph& result_pattern_graph,
      const MatchContextImpl& src_match_ctx,
      pir::PatternRewriter& rewriter) const;  // NOLINT

  void RebindIrTensorForAssignTensor(
      const ResultPatternGraph& result_pattern_graph,
      MatchContextImpl* res_match_ctx) const;

  void ReplaceOutputTensor(const MatchContextImpl& src_match_ctx,
                           const MatchContextImpl& res_match_ctx,
                           pir::PatternRewriter& rewriter) const;  // NOLINT

  void DeleteSourcePatternOp(const SourcePatternGraph& source_pattern_graph,
                             const ResultPatternGraph& result_pattern_graph,
                             const MatchContextImpl& src_match_ctx,
                             pir::PatternRewriter& rewriter) const;  // NOLINT

 private:
  const std::string pattern_name_;
  const std::shared_ptr<SourcePatternGraph> source_pattern_graph_;
  const std::vector<Constraint> constraints_;
  const std::vector<PostProcess> post_processes_;
  const std::shared_ptr<ResultPatternGraph> result_pattern_graph_;

  // Not used, just for hold it's life cycle.
  const std::shared_ptr<const DrrPatternBase> drr_pattern_owner_;
};

}  // namespace drr
}  // namespace paddle
