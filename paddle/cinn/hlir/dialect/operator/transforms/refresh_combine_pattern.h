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
#pragma once

#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/pattern_rewrite/pattern_match.h"

class RefreshCombineOpPattern
    : public ::pir::OpRewritePattern<::pir::CombineOp> {
 public:
  using ::pir::OpRewritePattern<::pir::CombineOp>::OpRewritePattern;
  bool MatchAndRewrite(pir::CombineOp op,
                       pir::PatternRewriter& rewriter) const override {
    auto new_combine_op = rewriter.Build<::pir::CombineOp>(op.inputs());
    rewriter.ReplaceAllUsesWith(op.result(0), new_combine_op.result(0));
    rewriter.EraseOp(op);
    return true;
  }
};
