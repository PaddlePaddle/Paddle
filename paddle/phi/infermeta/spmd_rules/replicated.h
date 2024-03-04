/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <vector>

#include "paddle/phi/core/distributed/auto_parallel/dist_meta_tensor.h"
#include "paddle/phi/core/distributed/type_defs.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi {
namespace distributed {
/**
 * A Bottom Line Rule that enforces input(s) and output(s) of the Op to be
 * replicated among the given mesh.
 *
 * This rule is used to support any op that have not been assign a specific rule
 * in auto parallel, and once there is a specific rule for that op,  replicated
 * rule would not effect that op any more.
 *
 * Vector of input tensors and output tensors used as argumnets (for both
 * inferfw & inferbw) to support any kind of op.
 *
 */
SpmdInfo ReplicatedInferSpmd(const std::vector<const DistMetaTensor*>& ins,
                             const std::vector<const DistMetaTensor*>& outs);

SpmdInfo ReplicatedInferSpmdReverse(
    const std::vector<const DistMetaTensor*>& ins,
    const std::vector<const DistMetaTensor*>& outs);

SpmdInfo ReplicatedInferDynamic(
    const std::vector<paddle::variant<const DistMetaTensor*,
                                      const std::vector<DistMetaTensor>*>>&
        inputs);

// For phi api
template <typename... Args>
SpmdInfo VariadicReplicatedInferSpmdDynamic(const Args&... args) {
  return detail::ReplicateInferSpmdDynamicHelper<ReplicatedInferDynamic>()
      .apply(args...)
      .Infer();
}

// For phi api
template <typename... Args>
SpmdInfo VariadicReplicatedInferSpmd(const Args&... args) {
  return detail::VariadicSpmdRuleArgumentParser<ReplicatedInferSpmd>()
      .apply(args...)
      .InferForward();
}

template <typename... Args>
SpmdInfo VariadicReplicatedInferSpmdReverse(const Args&... args) {
  return detail::VariadicSpmdRuleArgumentParser<ReplicatedInferSpmdReverse>()
      .apply(args...)
      .InferBackward();
}

}  // namespace distributed
}  // namespace phi
