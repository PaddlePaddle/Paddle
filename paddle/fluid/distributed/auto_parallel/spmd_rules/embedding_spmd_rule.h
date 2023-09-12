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

#include "paddle/fluid/distributed/auto_parallel/spmd_rules/common.h"

namespace paddle {
namespace distributed {
namespace auto_parallel {

// (TODO) Support 3 parallel cases for embedding:
// 1. Batch dimensions of input ids is sharded on mesh.
// 2. Row-wise Parallel of embedding table. (NOTE: Row-wise Parallel need to
// change the embedding kernel for miss ids.)
// 3. Column-wise Parallel of embedding table.
// 4. Hybrid Parallelism of above 3 cases.
class EmbeddingSPMDRule : public SPMDRuleBase {
 public:
  std::pair<std::vector<TensorDistAttr>, std::vector<TensorDistAttr>>
  InferForward(const std::vector<DistTensorSpec>& input_specs,
               const paddle::framework::AttributeMap& attrs) override;

  std::pair<std::vector<TensorDistAttr>, std::vector<TensorDistAttr>>
  InferBackward(const std::vector<DistTensorSpec>& input_specs,
                const std::vector<DistTensorSpec>& output_specs,
                const paddle::framework::AttributeMap& attrs) override;
};
}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
