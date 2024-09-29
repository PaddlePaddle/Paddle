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

#include "paddle/cinn/operator_fusion/pir_graph_analyzing/shardable_axes_base.h"

namespace cinn::fusion {

using FusionIters = std::vector<std::string>;
struct FusionItersSignature {
  FusionItersSignature() = default;
  FusionItersSignature(pir::Operation* op, const ShardableAxesSignature& axes);
  std::string DebugStr() const;

  FusionIters loop_iters = {};
  std::vector<FusionIters> input_iters = {};
  std::vector<FusionIters> output_iters = {};
  std::vector<pir::Value> input_values = {};
  std::vector<pir::Value> output_values = {};
};

class PatternNode;
using PatternNodePtr = std::shared_ptr<PatternNode>;
FusionItersSignature SingleDownstreamItersFusion(PatternNodePtr upstream,
                                                 PatternNodePtr downstream,
                                                 bool is_sink);

}  // namespace cinn::fusion
