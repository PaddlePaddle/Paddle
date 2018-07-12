/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/program_desc.h"

namespace paddle {
namespace framework {
namespace ir_optimizer {

// User-Side selections
enum class OptimizeLevel { NONE, MEM_FIRST, PERF_FIRST, MULTI_DEV, MULTI_NODE };

// Internal optimization passes
enum class OptimizePass {
  SSA,
  CONTROL_FLOW,
  MULTI_DEV,
  MEM_OPT,
  DISTRIBUTED_REPLICATED,
  DISTRIBUTED_COLLECTIVE
};

class IROptimizerBase {
 public:
  virtual ~IROptimizerBase() {}
  // Builders can modify the input program to achieve the build pass.
  virtual void Build(framework::ProgramDesc* prog) = 0;
};

class IROptimizer {
 public:
  IROptimizer() {}
  static framework::ProgramDesc Optimize(const OptimizeLevel& level,
                                         const framework::ProgramDesc& prog);

 protected:
  DISABLE_COPY_AND_ASSIGN(IROptimizer);
};

}  // namespace ir_optimizer
}  // namespace framework
}  // namespace paddle
