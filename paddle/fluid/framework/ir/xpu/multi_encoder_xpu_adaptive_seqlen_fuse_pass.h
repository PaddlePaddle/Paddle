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
case0 before:
support adaptive seq len for bert/ernie
    inpu_var*   mask_var**
        |             |
        |             |
  embedding_xpu     matmul
        |             |
        |             |
    layer_norm       scale
        |             |
        |             |
        |           stack
        \            /
         \          /
      multi_encoder_xpu
              |
              |
          out_var*

case0 after:
      inpu_var*   mask_var**
        \             /
         \           /
        embedding_xpu
        /           \
       /             \
embedding_out_var*  seq_lod_var*
      |               |
      |               |
  layer_norm          |
      \               /
       \             /
      multi_encoder_xpu
            |
            |
         out_var*
*/
class MultiEncoderXPUAdaptiveSeqlenFusePass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  const std::string name_scope_{"multi_encoder_xpu_adaptive_seqlen_fuse_pass"};
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
