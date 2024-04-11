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

#include "paddle/cinn/operator_fusion/frontend/pattern.h"
#include "paddle/cinn/operator_fusion/frontend/pattern_api.h"

namespace cinn::fusion {

template <>
StmtPattern<FrontendStage> ConvertToStmtPattern(const PatternContent<FrontendStage>& content) {
  const auto& kind = GetOpPatternKind(content.op);
  if (kind == hlir::framework::kReduction) {
    return ReducePattern<FrontendStage>({content.op});
  } else if (kind == hlir::framework::kElementWise ||
             kind == hlir::framework::kBroadcast ||
             kind == hlir::framework::kInjective) {
    return TrivialPattern<FrontendStage>({content.op});
  } else {
    return UnsupportPattern<FrontendStage>({content.op});
  }
}

template <>
StmtPattern<FrontendStage> RT_x_RT(const ReduceTreePattern<FrontendStage>& first,
                       const ReduceTreePattern<FrontendStage>& second) {
    const auto& merged = ConcatVector(first.reduce_patterns(),
                                      second.reduce_patterns());
    return ReduceTreePattern<FrontendStage>(merged, second.GetRootPattern());
}

template <>
StmtPattern<FrontendStage> RT_x_Trivial(const ReduceTreePattern<FrontendStage>& first,
                            const TrivialPattern<FrontendStage>& second) {
  return ReduceTreePlusTrivialPattern<FrontendStage>(first, second);
}

template <>
StmtPattern<FrontendStage> Trivial_x_Reduce(const TrivialPattern<FrontendStage>& first,
                            const ReducePattern<FrontendStage>& second) {
  const auto& contents =
      MergeVector(GetOpsInPattern<FrontendStage>(first), GetOpsInPattern<FrontendStage>(second));
  return ReducePattern<FrontendStage>(contents);
}

template <>
StmtPattern<FrontendStage> Trivial_x_Trivial(const TrivialPattern<FrontendStage>& first,
                            const TrivialPattern<FrontendStage>& second) {
  const auto& contents =
      MergeVector(GetOpsInPattern<FrontendStage>(first), GetOpsInPattern<FrontendStage>(second));
  return TrivialPattern<FrontendStage>(contents);
}

template <>
StmtPattern<FrontendStage> H_x_H(const HorizontalFusionPattern<FrontendStage>& first,
                     const HorizontalFusionPattern<FrontendStage>& second) {
  const auto& contents =
      MergeVector(GetOpsInPattern<FrontendStage>(first), GetOpsInPattern<FrontendStage>(second));
  return HorizontalFusionPattern<FrontendStage>({first, second});
}


}
