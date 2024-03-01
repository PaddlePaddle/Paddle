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

#include <memory>
#include <string>
#include <unordered_map>

#include "paddle/fluid/framework/ir/fuse_pass_base.h"

namespace paddle {
namespace framework {
namespace ir {

/*
 * Squash dequantize->quantize pair pattern into requantize op
 */

class XPUQuantizeSquashPass : public FusePassBase {
 public:
  XPUQuantizeSquashPass();
  virtual ~XPUQuantizeSquashPass() {}

 protected:
  void ApplyImpl(ir::Graph* graph) const override;

  /*
   * For each dequantize's output find the number of operators it is an input to
   */
  void FindNodesToKeep(
      Graph* graph,
      std::unordered_map<const Node*, int>* nodes_keep_counter) const;

  /*
   * Squash dequantize-quantize ops pairs into nothing
   */
  void DequantQuantSquash(
      Graph* graph,
      std::unordered_map<const Node*, int>* nodes_keep_counter) const;

  /*
   * Squash dequant if the previous operator support fp32 out
   */
  void OpDequantSquash(Graph* graph) const;

  /*
   * Squash quantize if several quantize ops have the same scale
   */
  void MultipleQuantizeSquash(Graph* graph) const;

  /*
   * Squash quantize if is before conv2d_xpu/fc_xpuy
   */
  void QuantOpSquash(Graph* graph) const;

  /*
   * Squash quantize(branch) + dequantize(out) in conv2d_xpu
   */
  void QuantConv2dFusionDequantSquash(Graph* graph) const;

  const std::string name_scope_{"squash"};
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
