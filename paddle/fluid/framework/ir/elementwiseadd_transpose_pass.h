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
//  input_x  input_y
//    |        |
//  elementwise_add
//    |
//   reshape
//    |
//   transpose
//    |
//
// fuse ->
//
//   |
//  elementwiseadd_transpose
//   |

class Graph;

class ElementwiseAddTransposeFusePass : public FusePassBase {
 public:
  ElementwiseAddTransposeFusePass() {
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
        .IsIntIn({-1})
        .End();
    AddOpCompat(OpCompat("reshape2"))
        .AddInput("X")
        .IsTensor()
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
        .AddAttr("axis")
        .IsType<std::vector<int>>()  // 0,3,2,1 nchw->nhwc
        .End();
  }
  virtual ~ElementwiseAddTransposeFusePass() {}

 protected:
  void ApplyImpl(ir::Graph* graph) const override;
  int ApplyEleTransPattern(ir::Graph* graph) const;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
