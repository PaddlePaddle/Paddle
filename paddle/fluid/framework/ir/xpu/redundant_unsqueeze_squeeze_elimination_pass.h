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

class RedundantUnsqueeze2EliminationPass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  /*
   Origin subgraph:
                      x
                      |
                   transpose2
                      |
             unsqueeze2(axes={-2})
                      |
                  reduce_sum
                      |
                     act
                      |
                  transpose2
                      |
   Fused subgraph:
                      x
                      |
                     act
                      |
  */
  void FoldTranspose2Ops(ir::Graph* graph, const std::string& act_type) const;
  /*
      Origin subgraph:
                      x
                      |
             unsqueeze2(axes={-2})
                      |
                    gather
                      |
                   squeeze2
                      |
   Fused subgraph:
                      x
                      |
                    gather
                      |
  */
  void FoldGatherSqueeze2Ops(ir::Graph* graph) const;
  /*
   Origin subgraph:
           x                    filter
           |                      |
      unsqueeze2(axes={-2})   unsqueeze2(axes={-2})
            \                   /
              \               /
                conv2d(conv1d)
                      |
                elementwise_add
                      |
                squeeze2(axes={-2})
                      |
                 batch_norm
                      |
                     act
                      |
                  unsqueeze2
                      |
                  conv2d(conv1d)
   Fused subgraph:
           x                    filter
           |                      |
      unsqueeze2(axes={-2})   unsqueeze2(axes={-2})
            \                   /
              \               /
                conv2d(conv1d)
                      |
                elementwise_add
                      |
                  batch_norm
                      |
                     act
                      |
                  conv2d(conv1d)
  */
  void FoldConv1dSqueeze2Ops(ir::Graph* graph,
                             const std::string& act_type) const;

  const std::string name_scope_{"redundant_unsqueeze_squeeze_elimination_pass"};
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
