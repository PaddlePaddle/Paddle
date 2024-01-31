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
#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"

namespace paddle {
namespace framework {
namespace ir {

class Graph;

class TransferLayoutElimPass : public FusePassBase {
 public:
  virtual ~TransferLayoutElimPass() {}

 protected:
  void ApplyImpl(ir::Graph *graph) const override;
  bool AllInputIsTransferlayout(const Node *op_node) const;
  void PutTransferlayoutAfterOp(Node *op_node,
                                ir::Graph *graph,
                                std::string *transfer_info) const;
  void ElimTwoTransferlayout(Node *op_node,
                             ir::Graph *graph,
                             bool *modify) const;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
