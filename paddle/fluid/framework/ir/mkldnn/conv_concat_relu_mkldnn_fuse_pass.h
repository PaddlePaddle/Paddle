// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include <unordered_map>

#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace framework {
namespace ir {

/*
 * Fuse the (multi conv) -> Concat -> ReLU -> next_op
 * to a:
 * (multi ConvReLU) -> Concat -> next_op.
 */
class Graph;

class ConvConcatReLUFusePass : public FusePassBase {
 public:
  virtual ~ConvConcatReLUFusePass() {}

 protected:
  void ApplyImpl(ir::Graph* graph) const override;

  void FindConcatWithConvs(
      Graph* graph,
      std::unordered_map<const Node*, int>* concat_with_convs_counter) const;

  void FuseConvConcatReLU(
      Graph* graph,
      std::unordered_map<const Node*, int>* concat_with_convs_counter) const;

  const std::string name_scope_{"conv_concat_relu_mkldnn_fuse"};
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
