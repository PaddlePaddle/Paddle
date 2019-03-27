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

#include <memory>
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
 * Squash dequantize->quantize pair pattern into requantize op
 */
class CPUQuantizeSquashPass : public FusePassBase {
 public:
  virtual ~CPUQuantizeSquashPass() {}

 protected:
  ir::Graph* ApplyImpl(ir::Graph* graph) const override;

  /*
   * For each dequantize's output find the number of operators it is an input to
   */
  void FindNodesToKeep(
      Graph* graph,
      std::unordered_map<const Node*, int>* nodes_keep_counter) const;

  /*
   * Squash dequantize-quantize ops pairs into requantize or nothing
   */
  void Squash(Graph* graph,
              std::unordered_map<const Node*, int>* nodes_keep_counter) const;

  const std::string name_scope_{"squash"};
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
