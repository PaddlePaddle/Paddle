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

//             other_op2
//     |           |                            |            |
// other_op1  elementwise_add               other_op1    other_op2
//     |           |              fuse           \          /
//     |------elementwise_add      ->          preln_residual_bias
//             |          |                        |      |
//        other_op4    layer_norm            other_op4  other_op3
//                       |
//                   other_op3
<<<<<<< HEAD
//                                 or
//
//     |           |                            |            |
// other_op1     other_op2                  other_op1    other_op2
//     |           |              fuse           \          /
//     |------elementwise_add      ->          preln_residual_bias
//             |          |                        |      |
//        other_op4    layer_norm            other_op4  other_op3
//                       |
//                   other_op3
=======
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
class Graph;

class PrelnResidualBiasFusePass : public FusePassBase {
 public:
  PrelnResidualBiasFusePass() {
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
<<<<<<< HEAD
        .IsIntIn({0, -1, 2})
=======
        .IsIntIn({0, -1})
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
        .End();

    AddOpCompat(OpCompat("layer_norm"))
        .AddInput("X")
        .IsTensor()
        .End()
        .AddInput("Scale")
        .IsTensor()
        .End()
        .AddInput("Bias")
        .IsTensor()
        .End()
        .AddOutput("Y")
        .IsTensor()
        .End()
        .AddOutput("Mean")
        .IsTensor()
        .End()
        .AddOutput("Variance")
        .IsTensor()
        .End()
        .AddAttr("epsilon")
        .IsNumGE(0.0f)
        .IsNumLE(0.001f)
        .End()
        .AddAttr("begin_norm_axis")
        .IsNumGT(0)
        .End();
  }

  virtual ~PrelnResidualBiasFusePass() {}

 protected:
  void ApplyImpl(ir::Graph* graph) const override;
<<<<<<< HEAD
  int ApplyPattern(ir::Graph* graph, bool with_bias) const;
=======
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
