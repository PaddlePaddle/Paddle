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
=======
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
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
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
        .IsIntIn({0, -1})
=======
        .IsIntIn({0, -1, 2})
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
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
=======
  int ApplyPattern(ir::Graph* graph, bool with_bias) const;
>>>>>>> d828ca460a89c2ce88be15bb5cdb76c676decf91
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
