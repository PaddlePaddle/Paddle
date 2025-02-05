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

#include <vector>

#include "paddle/common/flags.h"
#include "paddle/fluid/inference/analysis/analysis_pass.h"
#include "paddle/utils/test_macros.h"

namespace paddle {
namespace inference {
namespace analysis {

class TEST_API Analyzer final {
 public:
  Analyzer();

  void Run(Argument* argument);

  DISABLE_COPY_AND_ASSIGN(Analyzer);

 protected:
  void RunAnalysis(Argument* argument);
};

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
