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

// support adaptive seq len for bert/ernie
class MultiEncoderXPUAdaptiveSeqlenFusePass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  /*
  adaptive seqlen V1, before:

      input_var*     mask_var*
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

  after:

        input_var*    mask_var*
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
  int ApplyAdaptiveSeqlenPassV1(ir::Graph* graph,
                                const std::string& matmul_type) const;

  /*
  adaptive seqlen V2, before:

      input_var*          mask_var*
          |                 |
          |                 |
    embedding_xpu        not_equal
          |                 |
          |                 |
      layer_norm          cast
          |                 |
          |                 |
          |             unsqueeze2
          |                 |
          |                 |
          |              matmul_v2
          |                 |
          |                 |
          |               scale
          |                 |
          |                 |
          |               scale
          |                 |
          |                 |
          |             unsqueeze2
          |                 |
          |                 |
          |                tile
          \                 /
           \               /
           multi_encoder_xpu
                 |
                 |
              out_var*

  after:

        input_var*    mask_var*
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
  int ApplyAdaptiveSeqlenPassV2(ir::Graph* graph,
                                const std::string& matmul_type) const;

  /*
    adaptive seqlen V3, before:
                      x            mask
                      |             |
                      |             |
                      |           matmul
                      |             |
                      |             |
                      |           scale
                      |           /
                      |        stack
                      |         |
                      |        /
                      |      /
                  multi_encoder_xpu
                      |
                      |
                     out
  -------------------------------------------
   After the pass apply:
                      x      mask
                      |        |
                      |        |
                      | xpu_adaptive_mask
                      |        |     |
            sequence_unpad<--length  |
                      |              |
                      |            pad_seq_len
                      |            seq_lod
                      |            /
                      |          /
                      |        /
              multi_encoder_xpu
                      |
                      |
                    out
  -------------------------------------------
  */
  int ApplyAdaptiveSeqlenPassV3(ir::Graph* graph,
                                const std::string& matmul_type) const;

 private:
  const std::string name_scope_{"multi_encoder_xpu_adaptive_seqlen_fuse_pass"};
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
