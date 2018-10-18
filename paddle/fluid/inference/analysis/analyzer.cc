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

#include "paddle/fluid/inference/analysis/analyzer.h"
#include <string>
#include <vector>
#include "paddle/fluid/inference/analysis/passes/ir_analysis_compose_pass.h"

namespace paddle {
namespace inference {
namespace analysis {

Analyzer::Analyzer() {}

void Analyzer::Run(Argument *argument) { RunIrAnalysis(argument); }

void Analyzer::RunIrAnalysis(Argument *argument) {
  IrAnalysisComposePass pass;
  pass.RunImpl(argument);
}

/*
class PassManagerImpl final : public AnalysisPassManager {
 public:
  PassManagerImpl() {
    // TODO(Superjomn) set the key with pass reprs.
    if (!FLAGS_IA_enable_ir) {
      AddPass("fluid-to-data-flow-graph", new FluidToDataFlowGraphPass);
    } else {
      AddPass("fluid-to-ir-pass", new FluidToIrPass);
    }
    TryAddTensorRtPass();
  }

  std::string repr() const override { return "dfg-pass-manager"; }
  std::string description() const override { return "DFG pass manager."; }

 private:
  void AddPass(const std::string& name, AnalysisPass* pass) {
    VLOG(3) << "Adding pass " << name;
    Register(name, pass);
  }

  void TryAddTensorRtPass() {
    if (FLAGS_IA_enable_tensorrt_subgraph_engine) {
      auto trt_teller = [&](const framework::ir::Node* node) {
        std::unordered_set<std::string> teller_set(
            {"mul", "conv2d", "pool2d", "relu", "softmax", "sigmoid",
             "depthwise_conv2d", "batch_norm", "concat", "tanh", "pad",
             "elementwise_add", "dropout"});
        if (!node->IsOp()) return false;

        if (teller_set.count(node->Op()->Type())) {
          return true;
        } else {
          return false;
        }
      };

      AddPass("tensorrt-subgraph", new TensorRTSubGraphPass(trt_teller));
    }
  }
};

Analyzer::Analyzer(const std::vector<std::string>& ir_passes)
    : ir_passes_(ir_passes) {
  Register("program-analysis", new PassManagerImpl);
}

void Analyzer::Run(Argument* argument) {
  argument->Set(kFluidToIrPassesAttr, new std::vector<std::string>(ir_passes_));

  for (auto& x : elements_) {
    PADDLE_ENFORCE(x->Initialize(argument));
    x->RunAll();
    PADDLE_ENFORCE(x->Finalize());
  }
}
 */

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
