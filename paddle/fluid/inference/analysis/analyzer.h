/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

/*
 * This file contains Analyzer, an class that exposed as a library that analyze
 * and optimize Fluid ProgramDesc for inference. Similar to LLVM, it has
 * multiple flags to
 * control whether an process is applied on the program.
 *
 * The processes are called Passes in analysis, the Passes are placed in a
 * pipeline, the first Pass is the FluidToDataFlowGraphPass which transforms a
 * Fluid ProgramDesc to
 * a data flow graph, the last Pass is DataFlowGraphToFluidPass which transforms
 * a data flow graph to a Fluid ProgramDesc. The passes in the middle of the
 * pipeline can be any Passes
 * which take a node or data flow graph as input.
 *
 * The Analyzer can be used in two methods, the first is a executable file which
 * can be used to pre-process the inference model and can be controlled by
 * passing difference command flags;
 * the other way is to compose inside the inference API as a runtime pre-process
 * phase in the inference service.
 */

#include <gflags/gflags.h>
#include <string>
#include <vector>
#include "paddle/fluid/inference/analysis/analysis_pass.h"
#include "paddle/fluid/inference/analysis/flags.h"
#include "paddle/fluid/inference/analysis/pass_manager.h"

namespace paddle {
namespace inference {
namespace analysis {

class Analyzer : public OrderedRegistry<PassManager> {
 public:
  // Register all the pass-managers.
  Analyzer();

  void Run(Argument* argument);

  Analyzer& DisableIrPasses(const std::vector<std::string>& passes);
  Analyzer& IncludeIrPasses(const std::vector<std::string>& passes);
  Analyzer& IncludeAllIrPasses();
  Analyzer& SetUseMkldnn(bool use_mkldnn);

  DISABLE_COPY_AND_ASSIGN(Analyzer);

 private:
  // All avaiable IR passes.
  // The bigger fuse comes first, so that the small operators prefer to be
  // merged in a larger fuse op. The small fusion will not break the pattern of
  // larger fusion.
  const std::vector<std::string> all_ir_passes_{{
      // Manual update the passes here.
      "attention_lstm_fuse_pass",       //
      "seqconv_eltadd_relu_fuse_pass",  //
      "embedding_fc_lstm_fuse_pass",    //
      "fc_lstm_fuse_pass",              //
      "mul_lstm_fuse_pass",             //
      "fc_gru_fuse_pass",               //
      "mul_gru_fuse_pass",              //
      "seq_concat_fc_fuse_pass",        //
      "fc_fuse_pass",                   //
      "conv_bn_fuse_pass",              //
      "conv_eltwiseadd_bn_fuse_pass",   //
#ifdef PADDLE_WITH_MKLDNN
      "depthwise_conv_mkldnn_pass",             //
      "conv_bias_mkldnn_fuse_pass",             //
      "conv_relu_mkldnn_fuse_pass",             //
      "conv_elementwise_add_mkldnn_fuse_pass",  //
#endif
  }};

  std::unordered_set<std::string> disabled_ir_passes_;
  // Ir passes to run
  std::vector<std::string> ir_passes_;
  bool use_mkldnn_;
};

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
