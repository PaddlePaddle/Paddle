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

#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace framework {
namespace ir {

//    |
// reshape2
//    |
// reshape2
//    |
// transpose2     ->           reverse_roll (shift_size=0)
//    |          fuse
// reshape2
//    |
// reshape2
//    |
//

class SplitGeluMulFusePass : public FusePassBase {
 public:
  SplitGeluMulFusePass() {
    AddOpCompat(OpCompat("split"))
        .AddInput("X")
        .IsTensor()
        .End()
        .AddOutput("Out")
        .IsTensorList()
        .End()
        .AddAttr("axis")
        .IsNumEQ(2)
        .End()
        .AddAttr("num")
        .IsNumEQ(2)
        .End();
    AddOpCompat(OpCompat("gelu"))
        .AddInput("X")
        .IsTensor()
        .End()
        .AddOutput("Out")
        .IsTensor()
        .End();
    AddOpCompat(OpCompat("elementwise_mul"))
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
  }
  virtual ~SplitGeluMulFusePass() {}

 protected:
  void ApplyImpl(ir::Graph *graph) const override;
  int ApplyPattern(ir::Graph *graph) const;

 private:
  const std::string scope_name_{"split_gelu_mul_fuse"};
};
}  // namespace ir
}  // namespace framework
}  // namespace paddle
