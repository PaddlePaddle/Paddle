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

#include "paddle/cinn/operator_fusion/pattern_fuser.h"

namespace cinn::fusion {

StmtPattern ConvertToStmtPattern(const PatternContent& content) {
  const auto& kind = GetOpPatternKind(content.op);
  if (kind == hlir::framework::kReduction) {
    return ReducePattern({content.op});
  } else if (kind == hlir::framework::kElementWise ||
             kind == hlir::framework::kBroadcast ||
             kind == hlir::framework::kInjective) {
    return TrivialPattern({content.op}, content.op);
  } else {
    return UnsupportPattern({content.op});
  }
}

StmtPattern MergePatternImpl(const ReduceTreePattern& first,
                             const TrivialPattern& second) {
  return ReduceTreePlusTrivialPattern(first, second);
}

StmtPattern MergePatternImpl(const TrivialPattern& first,
                             const ReducePattern& second) {
  const auto& contents =
      UniqueConcatVector(GetOpsInPattern(first), GetOpsInPattern(second));
  return ReducePattern(contents);
}

StmtPattern MergePatternImpl(const TrivialPattern& first,
                             const TrivialPattern& second) {
  const auto& contents =
      UniqueConcatVector(GetOpsInPattern(first), GetOpsInPattern(second));
  return TrivialPattern(contents, second.sink_op());
}

StmtPattern MergePatternImpl(const TrivialPattern& first,
                             const AnchorPattern& second) {
  return AnchorPattern(
      UniqueConcatVector(GetOpsInPattern(first), GetOpsInPattern(second)),
      second.anchor(),
      second.anchor_state);
}

StmtPattern MergePatternImpl(const AnchorPattern& source,
                             const AnchorPattern& dest) {
  const auto& contents =
      UniqueConcatVector(GetOpsInPattern(source), GetOpsInPattern(dest));
  return AnchorPattern(contents, source.anchor(), AnchorState({}));
}

ExprPromise InitExprPromiseImpl(const TrivialPattern& pattern,
                                pir::Value anchor) {
  return ExprPromise(anchor);
}

ExprPromise InitExprPromiseImpl(const ReducePattern& pattern,
                                pir::Value anchor) {
  return ExprPromise(anchor);
}

TrivialPattern RecoverAnchorPatternToTrivial(
    const AnchorPattern& anchor_pattern) {
  PADDLE_ENFORCE_EQ(anchor_pattern.anchor_state.promise.size(),
                    1,
                    phi::errors::PreconditionNotMet(
                        "Can only recover AnchorPattern whose anchor_state "
                        "size is 1 (exact %d)",
                        anchor_pattern.anchor_state.promise.size()));

  return TrivialPattern(anchor_pattern.ops(),
                        anchor_pattern.anchor().defining_op());
}

}  // namespace cinn::fusion
