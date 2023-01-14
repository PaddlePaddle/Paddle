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
//    conv2d
//     |
//   transpose
//     |
//    reshape
//    |    |
//   out  layernorm
//         |
//        layernorm_out
//
// ->fuse to
//   conv2d
//    |
//  trans_layernorm
//   |      |
//  out    layernorm_out
class Graph;

class TransLayernormFusePass : public FusePassBase {
 public:
  TransLayernormFusePass() {
    AddOpCompat(OpCompat("reshape2"))
        .AddInput("X")
        .IsTensor()
        .End()
        .AddInput("Shape")
        .IsTensor()
        .IsOptional()
        .End()
        .AddInput("ShapeTensor")
        .IsOptional()
        .End()
        .AddOutput("Out")
        .IsTensor()
        .End()
        .AddOutput("XShape")
        .IsOptional()
        .IsTensor()
        .End()
        .AddAttr("shape")
        .IsType<std::vector<int>>()
        .End();
    AddOpCompat(OpCompat("flatten_contiguous_range"))
        .AddInput("X")
        .IsTensor()
        .End()
        .AddOutput("Out")
        .IsTensor()
        .End()
        .AddOutput("XShape")
        .IsOptional()
        .IsTensor()
        .End()
        .AddAttr("start_axis")
        .IsNumEQ(1)
        .End()
        .AddAttr("stop_axis")
        .IsNumEQ(2)
        .End();
    AddOpCompat(OpCompat("transpose2"))
        .AddInput("X")
        .IsTensor()
        .End()
        .AddOutput("Out")
        .IsTensor()
        .End()
        .AddOutput("XShape")
        .IsOptional()
        .IsTensor()
        .End()
        .AddAttr("axis")  // {0, 2, 1, 3}
        .IsType<std::vector<int>>()
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
        .AddAttr("begin_norm_axis")
        .IsNumGT(0)
        .End()
        .AddAttr("epsilon")
        .IsNumGE(0.0f)
        .IsNumLE(1.0f)
        .End();
  }
  virtual ~TransLayernormFusePass() {}

 protected:
  void ApplyImpl(ir::Graph* graph) const override;
  int ApplyConvTransLayernormPattern(ir::Graph* graph) const;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
