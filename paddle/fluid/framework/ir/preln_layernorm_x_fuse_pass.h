/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/fluid/framework/ir/fuse_pass_base.h"

namespace paddle {
namespace framework {
namespace ir {
//
//     |           |                            |            |
// other_op1     other_op2                  other_op1    other_op2
//     |           |              fuse           \          /
//     |------elementwise_add      ->    preln_layernorm_shift_partition
//             |          |                        |      |
//        other_op4  layernorm_shift_partition  other_op4  other_op3
//                        |
//                   other_op3
//                                 or
//     |           |                            |            |
// other_op1     other_op2                  other_op1    other_op2
//     |           |              fuse           \          /
//     |------elementwise_add      ->        preln_merge_layernorm
//             |          |                        |      |
//        other_op4  merge_layernorm          other_op4  other_op3
//                        |
//                   other_op3
class Graph;

class PrelnLayerNormXFusePass : public FusePassBase {
 public:
  PrelnLayerNormXFusePass() {
    AddOpCompat(OpCompat("elementwise_add"))
        .AddInput("X")
        .IsTensor()
        .End()
        .AddInput("Y")
        .IsTensor()
        .End()
        .AddOutput("Out")
        .IsTensor()
        .End()
        .AddAttr("axis")
        .IsIntIn({0, -1, 2})
        .End();
  }

  virtual ~PrelnLayerNormXFusePass() {}

 protected:
  void ApplyImpl(ir::Graph* graph) const override;
  int ApplyLayerNormShiftPattern(ir::Graph* graph) const;
  int ApplyMergeLayerNormPattern(ir::Graph* graph) const;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
