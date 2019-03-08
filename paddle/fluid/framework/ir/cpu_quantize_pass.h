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

#include <map>
#include <memory>
#include <string>
#include <utility>
#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/inference/api/paddle_quantizer_config.h"  // for QuantMax

namespace paddle {
namespace framework {
namespace ir {

using VarQuantMaxAndScale =
    std::map<std::string, std::pair<QuantMax, LoDTensor>>;

/*
 * Quantize all supported operators.
 */
class CPUQuantizePass : public FusePassBase {
 public:
  virtual ~CPUQuantizePass() {}

 protected:
  std::unique_ptr<ir::Graph> ApplyImpl(
      std::unique_ptr<ir::Graph> graph) const override;

  void QuantizeConv(Graph* graph, bool with_residual_data = false) const;

  void QuantizePool(Graph* graph) const;

  template <typename OutT>
  void QuantizeInput(Graph* g, Node* op, Node* input, std::string input_name,
                     std::string prefix, float scale, bool is_negative) const;

  template <typename InT>
  void DequantizeOutput(Graph* g, Node* op, Node* output,
                        std::string output_name, std::string prefix,
                        float scale) const;

  const std::string name_scope_{"quantize"};
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
