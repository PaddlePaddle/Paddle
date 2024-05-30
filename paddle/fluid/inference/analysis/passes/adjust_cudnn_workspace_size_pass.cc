// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/analysis/passes/adjust_cudnn_workspace_size_pass.h"

#include "paddle/fluid/inference/analysis/argument.h"

namespace paddle::inference::analysis {

void AdjustCudnnWorkSpacePass::RunImpl(Argument* argument) {
  if (!argument->use_gpu()) return;
  auto& graph = argument->main_graph();
  auto nodes = graph.Nodes();
  const int cudnn_workspace_size_MB = 64;
  const std::string attr_name = "workspace_size_MB";

  for (auto& node : nodes) {
    if (!node->IsOp()) continue;
    auto* op_desc = node->Op();
    if (!op_desc->HasAttr(attr_name)) continue;
    op_desc->SetAttr(attr_name, cudnn_workspace_size_MB);
    op_desc->Flush();
  }
}

std::string AdjustCudnnWorkSpacePass::repr() const {
  return "adjust-cudnn-work-space-pass";
}

}  // namespace paddle::inference::analysis
