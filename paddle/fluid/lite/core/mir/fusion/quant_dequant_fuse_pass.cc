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

#include "paddle/fluid/lite/core/mir/fusion/quant_dequant_fuse_pass.h"
#include <memory>
#include <vector>
#include "paddle/fluid/lite/core/mir/fusion/quant_dequant_op_fuser.h"
#include "paddle/fluid/lite/core/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

void QuantDequantFusePass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  std::unordered_set<std::string> quant_types = {
      "fake_quantize_range_abs_max", "fake_quantize_moving_average_abs_max"};
  std::unordered_set<std::string> quantized_op_types = {"conv2d", "mul",
                                                        "depthwise_conv2d"};
  for (auto& quant_type : quant_types) {
    for (auto& op_type : quantized_op_types) {
      for (int i = 6; i >= 1; i--) {
        fusion::QuantDequantOpFuser fuser(op_type, quant_type, i);
        fuser(graph.get());
      }
    }
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(lite_quant_dequant_fuse_pass,
                  paddle::lite::mir::QuantDequantFusePass);
