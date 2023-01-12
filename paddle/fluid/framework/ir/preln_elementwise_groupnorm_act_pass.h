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
//  |           |
// elementwise_add            fuse         |       |
//   |         |               ->        preln_gn_act
// other op  group_norm                    |       |
//             |                        other op
//            silu
//             |

class Graph;

class PrelnGroupNormActFusePass : public FusePassBase {
 public:
  PrelnGroupNormActFusePass() {
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
        .IsIntIn({0, -1})
        .End();
    AddOpCompat(OpCompat("group_norm"))
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
        .IsNumLE(1.0f)
        .End()
        .AddAttr("groups")
        .IsNumGE(1)
        .End()
        .AddAttr("data_layout")
        .IsStringIn({"NCHW"})
        .End();
    AddOpCompat(OpCompat("silu"))
        .AddInput("X")
        .IsTensor()
        .End()
        .AddOutput("Out")
        .IsTensor()
        .End();
  }

  virtual ~PrelnGroupNormActFusePass() {}

 protected:
  void ApplyImpl(ir::Graph* graph) const override;
  int ApplyGNSiluPattern(ir::Graph* graph) const;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
