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

/*
This pass is used to fuse the QKV attention subgraph into one op in visual
models .

Origin subgraph:
                         input
                           |
                           |
                           |
                        reshape
                           |
                           |
                           |
                      transpose2
                    /      |     \
                   /       |      \
                  /        |       \
                slice    slice     slice
                  |        |         |
                  |        |         |
                  |        |         |
                  |     (scale)   transpose2
                  |        |         |
                  |         \       /
                  |          \    /
                  |         qk_matmul
                  |           |
                  |           |
                  |           |
                  |         (scale)
                  |           |
                  |           |
                  |           |
                   \        qk_softmax
                     \        |
                       \      |
                         \    |
                        qkv_matmul
                              |
                              |
                              |
                          transpose2
                              |
                              |
                              |
                           reshape
                              |
                              |
                              |
                            output

The scale can be placed either above or below the qk_matmul. These two
positions are mathematically equivalent.
-------------------------------------------------------
Fused subgraph:
                         input
                           |
                           |
                           |
                   qkv_attention_xpu
                           |
                           |
                           |
                         output

*/

class QkQkvAttentionXPUFusePass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  void ApplyQkQkvAttentionXPUFuse(ir::Graph* graph,
                                  bool with_q_scale,
                                  bool scale_above_qk) const;

  const std::string name_scope_{"qk_qkv_attention_xpu_fuse_pass"};
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
