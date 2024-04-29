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

class DeleteCastOpPass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  /*
 Origin subgraph:
     main_graph:             while subgraph:

     write_to_array          cast(fp16->fp32)
         |                         |
    (write_var:fp32)          write_to_array
                                   |
                             (write_var:fp32)
                                   |
                             read_from_array
                                   |
                             cast(fp32->fp16)

 Optimized subgraph:
     main_graph:             while subgraph:

       cast                  write_to_array
        |                           |
   write_to_array            (write_var:fp16)
        |                           |
  (write_var:fp16)           read_from_array
  */
  int ApplyCastWriteReadPass(ir::Graph* graph) const;

  /*
 Origin subgraph:
     main_graph:             while subgraph:

     write_to_array          cast(fp16->fp32)
         |                         |
    (write_var:fp32)          lod_reset
         |                         |
       while                  write_to_array
         |                         |
  (write_var:fp32)           (write_var:fp32)
         |                         |
  beam_search_decode          read_from_array
         |                         |
   (out_score:fp32)           cast(fp32->fp16)

 Optimized subgraph:
     main_graph:             while subgraph:

        cast                   lod_reset
         |                         |
   write_to_array            write_to_array
         |                         |
  (write_var:fp16)          (write_var:fp16)
         |                         |
       while                  read_from_array
         |
  (write_var:fp16)
         |
  beam_search_decode
         |
   cast(fp16->fp32)
         |
    (out_score:fp32)
  */
  int ApplyCastLodResetWriteReadPass(ir::Graph* graph) const;

  /*
 Origin subgraph:
       cast(fp16->fp32)
         |
    index_sample
         |
       cast(fp32->fp16)

 Optimized subgraph:
     index_sample
  */
  int ApplyCastIndexSamplePass(ir::Graph* graph) const;

  /*
 Origin subgraph:
       cast(fp16->fp32) cast(fp16->fp32)
                  \       /
                   scatter
                      |
              cast(fp32->fp16)

 Optimized subgraph:
                   scatter
  */
  int ApplyCastScatterPass(ir::Graph* graph) const;

  /*
 Origin subgraph:
     ids      w(fp32)
      \       /
     lookup_table
          |
    cast(fp32->fp16)

 Optimized subgraph:
     ids      w(fp16)
      \       /
     lookup_table
  */
  int ApplyCastLookupTablePass(ir::Graph* graph) const;

  // Delete cast if its "in_dtype" is the same as "out_dtype"
  int ApplyCastPass(ir::Graph* graph) const;

  const std::string name_scope_{"delete_cast_op_pass"};
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
