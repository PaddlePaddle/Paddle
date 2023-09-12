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

class OneBeamSizeFusePass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  /*
   Origin subgraph:
               fused_multi_transformer
                |        |        |
              assign   assign    ...
                |        |        |
              gather   gather    ...

   Fused subgraph:
               fused_multi_transformer
  */
  void RemoveAssignGather(ir::Graph* graph) const;

  /*
   Origin subgraph:
                        shape
                       /  |  \
                      /   |   \
        elementwise_div   |   scale
                    |     |     |
                  cast  cast   cast
                    \     |     /
                        range
                          |
                      unsqueeze2
                          |
                        scale   (add_x)
                          |     /
                  elementwise_add
                          |
              flatten_contiguous_range

   Fused subgraph:
                      (add_x)
                         |
               flatten_contiguous_range
  */
  void FoldShapeAssociatedOps(ir::Graph* graph) const;

  /*
  Origin subgraph:
                       lod_reset lod_reset
                           |         |
                         (ids)    (scores)
                            \        |
                            beam_search
                          /      |      \
                         /       |       \
                        /        |        \
          (selected_ids) (selected_scores) (parent_idx)
            /       |            |               |
  write_to_array is_empty  write_to_array      cast
                    |                            |
                    |                       (cast_out)
                    |                            |
                logical_not                 write_to_array

  Fused subgraph:
              lod_reset           lod_reset        (cast_out: fill 0)
                  |                   |                   |
                (ids)             (scores)            write_to_array
                /   \                 |
   write_to_array   not_equal    write_to_array

  */
  void RemoveBeamSearchAssociatedOps(ir::Graph* graph) const;

  /*
  Origin subgraph:
    (x: persistable)    (index)
            \            /
            write_to_array
                  |
            read_from_array
                  |
                any_op

  Fused subgraph:
              (x: persistable)
                  |
                any_op
  */
  void RemoveWriteReadArrayOps(ir::Graph* graph) const;

  /*
  Origin subgraph:
      (x: dims0=1)   (index=[0])
                \    /
              gather(axis=0)
                  |
                any_op

  Fused subgraph:
                 (x)
                  |
                any_op
  */
  void RemoveGatherOps(ir::Graph* graph) const;

  const std::string name_scope_{"one_beam_size_fuse_pass"};
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
