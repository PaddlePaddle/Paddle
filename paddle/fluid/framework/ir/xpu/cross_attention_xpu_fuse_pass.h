// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

/*
This pass is used to fuse the cross attention op into one op in decoder.
models .

Origin subgraph:

  mask      input_q          input_kv
   |           |          |           |
   |           |          |-----------|
   |         matmul      matmul      matmul
   |          |q          |k          |v
   |          |           |           |
   |          |           |           |
   |         add         add         add
   |          |           |           |
   |          |           |           |
   |       reshape     reshape     reshape
   |          |           |           |
   |          |           |           |
   |      transpose   transpose  transpose
   |          |           |           |
   |          |           |           |
   |       (scale)        |           |
   |          |           |           |
     \        |(x)        |(y)        |
       \       \        /             |
         \     qk_matmul              |
           \      |                   |
             \    |                   |
                add                  /
                  |                 /
                  |               /
               softmax          /
                  \           /
                   \        /
                   qkv_matmul
                       |
                       |
                   transpose
                       |
                       |
                    reshape
                       |
                       |
                     output

-------------------------------------------------------
Fused subgraph:
                    input_q   input_kv
                       |        |
                       |        |
                       |        |
                   cross_attention_xpu
                           |
                           |
                           |
                         output

*/

class CrossAttentionXPUFusePass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  void ApplyCrossAttentionXPUFuse(ir::Graph* graph, bool with_q_scale) const;

  // 1. Generate q/k/v_w_max tensor
  // 2. Quant q/k/v_w to int16
  void PrepareQKVWeight(Graph* graph,
                        Scope* scope,
                        BlockDesc* block,
                        Node* w,
                        Node** real_w,
                        Node** w_max) const;

  // Cast fc_bias to fp32
  void PrepareQKVBias(Graph* graph,
                      Scope* scope,
                      BlockDesc* block,
                      Node* q_bias,
                      Node* k_bias,
                      Node* v_bias,
                      Node** real_q_bias,
                      Node** real_k_bias,
                      Node** real_v_bias) const;

  const std::string name_scope_{"cross_attention_xpu_fuse_pass"};
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
