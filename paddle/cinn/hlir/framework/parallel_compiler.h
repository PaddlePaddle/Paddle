// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include <mutex>
#include <vector>

#include "paddle/cinn/backends/llvm/execution_engine.h"
#include "paddle/cinn/common/target.h"
#include "paddle/cinn/hlir/framework/graph.h"
#include "paddle/cinn/hlir/framework/graph_compiler_util.h"
#include "paddle/cinn/hlir/framework/instruction.h"
#include "paddle/cinn/hlir/framework/op_lowering.h"
#include "paddle/cinn/ir/lowered_func.h"
#ifdef CINN_WITH_CUDA
#include "paddle/cinn/runtime/cuda/cuda_module.h"
#endif
namespace cinn {
namespace hlir {
namespace framework {

class ParallelCompiler {
 public:
  struct Task {
    Task(ParallelCompiler* compiler, CompilationContext* context)
        : compiler(compiler), context(context) {}
    void Lowering();
    void CodegenAndJit();
    void BuildInstruction();

    ParallelCompiler* compiler;
    CompilationContext* context;

    CompilationStatus status = CompilationStatus::SUCCESS;
    std::string message;

    int start_gidx;
    int stop_gidx;
    std::vector<std::unique_ptr<Instruction>> instructions;
    std::vector<std::vector<ir::LoweredFunc>> lowered_funcs;
    std::vector<std::string> source_codes;
    std::vector<std::string> source_ptxs;

    std::unique_ptr<backends::ExecutionEngine> engine;
#ifdef CINN_WITH_CUDA
    std::unique_ptr<runtime::cuda::CUDAModule> cumodule;
#endif
  };

  explicit ParallelCompiler(CompilationContext* context) : context_(context) {}
  ~ParallelCompiler() = default;
  CompilationResult operator()();

 private:
  void SplitTask();
  void LaunchTask();
  void RunTask(Task* task);
  CompilationResult MergeResult();

  std::vector<Task> tasks_;
  CompilationContext* context_;
};

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
