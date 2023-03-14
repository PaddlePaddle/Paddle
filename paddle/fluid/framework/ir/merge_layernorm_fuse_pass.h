// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>

#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"

namespace paddle {
namespace framework {
namespace ir {

// Fusing of path merge and layer_norm
// op: ss=stride_slice
// shape: [ss] = [?x28x28x96]
//       input
//         | [?x3136x96]
//       reshape2                                 input
//         | [?x56x56x96]                           | [?x3136x96]
//         |------|------|------|              merge_layernorm
//        ss     ss     ss      ss      ->          | [?x784x384]
//         | [ss] | [ss] | [ss] | [ss]  fused      output
//         |------|------|------|
//       concat
//         | [?x28x28x384]
//       reshape2
//         | [?x784x384]
//       layer_norm
//         | [?x784x384]
//        output
class MergeLayernormFusePass : public FusePassBase {
 public:
  MergeLayernormFusePass();
  virtual ~MergeLayernormFusePass() {}

 protected:
  void ApplyImpl(ir::Graph* graph) const override;
};
}  // namespace ir
}  // namespace framework
}  // namespace paddle
