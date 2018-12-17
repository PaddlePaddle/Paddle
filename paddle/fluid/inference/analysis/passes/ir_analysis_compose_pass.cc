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

#include "paddle/fluid/inference/analysis/passes/ir_analysis_compose_pass.h"
#include <string>
#include <vector>
#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/inference/analysis/ir_pass_manager.h"
#include "paddle/fluid/inference/analysis/ir_passes/subgraph_detector.h"
#include "paddle/fluid/string/pretty_log.h"

namespace paddle {
namespace inference {
namespace analysis {

void IrAnalysisComposePass::RunImpl(Argument *argument) {
  ARGUMENT_CHECK_FIELD(argument, ir_analysis_passes);
  if (argument->use_tensorrt_valid() && argument->use_tensorrt()) {
    InitTensorRTAttrs(argument);
  }
  ApplyIrPasses(argument);
  CollectFusionStatis(argument);
}

std::string IrAnalysisComposePass::repr() const {
  return "ir-analysis-compose-pass";
}

void IrAnalysisComposePass::InitTensorRTAttrs(Argument *argument) {
  if (argument->use_tensorrt_valid() && argument->use_tensorrt()) {
    LOG(INFO) << "Initing TensorRT pass";
    argument->SetTensorRtNodeTeller([](const framework::ir::Node *node) {
      std::unordered_set<std::string> teller_set(
          {"mul", "conv2d", "pool2d", "relu", "softmax", "sigmoid",
           "depthwise_conv2d", "batch_norm", "concat", "tanh", "pad",
           "elementwise_add", "elementwise_mul", "dropout", "split", "prelu",
           "conv2d_transpose", "leaky_relu"});
      if (!node->IsOp()) return false;

      if (teller_set.count(node->Op()->Type())) {
        return true;
      } else {
        return false;
      }
    });
  }
}

void IrAnalysisComposePass::ApplyIrPasses(Argument *argument) {
  std::vector<std::string> passes({
      "ir_graph_build_pass", "ir_analysis_pass",
      "ir_params_sync_among_devices_pass",
  });
  for (const auto &pass : passes) {
    VLOG(2) << "Run pass " << pass;
    auto *the_pass = PassRegistry::Global().Retreive(pass);
    the_pass->Run(argument);
  }
}

void IrAnalysisComposePass::CollectFusionStatis(Argument *argument) {
  if (!argument->main_graph().Has(framework::ir::kFuseStatisAttr)) {
    LOG(INFO) << "argument has no fuse statis";
    return;
  }
  argument->SetFusionStatis(
      argument->main_graph().Get<Argument::fusion_statis_t>(
          framework::ir::kFuseStatisAttr));
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
