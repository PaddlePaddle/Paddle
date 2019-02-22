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

#include "paddle/fluid/framework/ir/cpu_quantize_scale_out_pass.h"

namespace paddle {
namespace framework {
namespace ir {

std::unique_ptr<ir::Graph> CPUQuantizeScaleOutPass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  PADDLE_ENFORCE(graph.get());
  VLOG(3) << "Fuses conv2d->deuqntize pattrn." << std::endl;

  // TODO(sfraczek): Fuse conv2d->dequantize pattern to a conv2d op with output
  // scale

  return graph;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(cpu_quantize_scale_out_pass,
              paddle::framework::ir::CPUQuantizeScaleOutPass);
