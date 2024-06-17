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
#include <utility>
#include <vector>

#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"

namespace paddle {
namespace framework {
namespace ir {

/*
 * Quantize all supported operators.
 */
class XPUQuantizeOpPass : public FusePassBase {
 public:
  virtual ~XPUQuantizeOpPass() {}

 protected:
  void ApplyImpl(Graph* graph) const override;
  void QuantizeConv(Graph* graph) const;
  void QuantizeQkvAttention(Graph* graph) const;
  void QuantizeFC(Graph* graph) const;

 private:
  void QuantizeInput(Graph* g,
                     Node* op,
                     Node* input,
                     std::string input_arg_name) const;

  void DequantizeOutput(Graph* g,
                        Node* op,
                        Node* output,
                        std::string output_arg_name) const;

  void GetQuantInfo(Graph* graph) const;

  mutable std::unordered_map<std::string, std::vector<float>> var_quant_scales_;
  const std::string name_scope_{"xpu_quantize_op_pass"};
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
