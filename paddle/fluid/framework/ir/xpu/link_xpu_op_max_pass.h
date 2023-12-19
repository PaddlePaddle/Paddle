// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/framework/ir/pass.h"

namespace phi {
class DenseTensor;
}  // namespace phi

namespace paddle {
namespace framework {
class Scope;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace ir {

class LinkXPUOpMaxPass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  /*
  Origin subgraph:
          fusion_xpu_op0
            /       \
            |       |
          out0   out0_max
            |
            \
            fc_xpu
Fused subgraph:
          fusion_xpu_op0
            /       \
            |       |
          out0   out0_max
            |       |
            \       /
            fc_xpu
  */
  void LinkFcMax(ir::Graph* graph) const;

  /*
  Origin subgraph:
          fusion_xpu_op0     fusion_xpu_op1
            /       \         /          \
            |       |         |          |
          out0   out0_max    out1      out1_max
            |                 |
        (x) \                / (branch)
              conv2d_xpu
Fused subgraph:
          fusion_xpu_op0     fusion_xpu_op1
            /       \         /           \
            |       |         |            |
          out0   out0_max    out1      out1_max
            |       |          |           |
        (x) \       |(x_max)   |(branch)  /(branch_max)
             \      |          |         /
              \     |          |        /
               \    |          |       /
                   conv2d_xpu
  */
  void LinkConv2dMax(ir::Graph* graph, bool with_branch) const;

  /*
  Origin subgraph:
            fusion_xpu_op0     fusion_xpu_op1
              /       \         /          \
              |       |         |          |
            out0   out0_max    out1      out1_max
              |                 |
          (x) \                / (y)
                add_act_xpu
  Fused subgraph:
            fusion_xpu_op0     fusion_xpu_op1
              /       \         /           \
              |       |         |            |
            out0   out0_max    out1      out1_max
              |       |          |           |
          (x) \       |(x_max)   |(y)  /(y_max)
               \      |          |         /
                \     |          |        /
                 \    |          |       /
                     add_act_xpu
  */
  void LinkAddActMax(ir::Graph* graph) const;

  bool IsQuant(Node* weight_node) const;
  const std::string name_scope_{"link_xpu_op_max_pass"};
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
