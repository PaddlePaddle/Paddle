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

#include "paddle/ir/pattern_rewrite/pattern_match.h"

namespace ir {

template <typename SourceOp, typename DrrFunctor>
struct DrrRewritePattern : public ir::OpRewritePattern<SourceOp> {
  DrrRewritePattern(ir::IrContext* context, ir::PatternBenefit benefit)
      : ir::OpRewritePattern<SourceOp>(context, benefit) {
    DrrPatternContext drr_context;
    DrrFunctor functor;
    functor(&drr_context);
    source_pattern_graph_ = drr_context.SourcePatternGraph();
    constraints_ = drr_context.Constraints();
    result_pattern_graph_ = drr_context.ResultPatternGraph();
  }

  bool Match(SourceOp op) const override {
    // Match

    return true;
  }

  void Rewrite(SourceOp op,
               ir::PatternRewriter& rewriter) const override {  // NOLINT
    // Rewrite
  }

  const SourcePatternGraph* source_pattern_graph_;
  const std::vector<Constraint*>& constraints_;
  const ResultPatternGraph* result_pattern_graph_;
};

}  // namespace ir
