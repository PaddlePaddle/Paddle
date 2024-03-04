// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/cinn/auto_schedule/tuning.h"
#include "paddle/cinn/common/common.h"
#include "paddle/cinn/hlir/framework/graph.h"
#include "paddle/cinn/hlir/framework/program.h"
#include "paddle/cinn/ir/lowered_func.h"

namespace cinn {
namespace hlir {
namespace framework {

// An enum class used to control the compilation stage.
enum class CompilationStage {
  // Fully compiled by default, the following compilation result can be
  // obtained: lowered_function, source_code, source_ptx, instruction and
  // runtime_program.
  DEFAULT = 0,
  // Just do lowering, we can only get lowered_function from compilation result.
  LOWERING = 1,
  // Stop after codegen and jit, we can get: lowered_function, source_code and
  // source_ptx from compilation result.
  CODEGEN_AND_JIT = 2,
  // Stop after build instruction, we can get: lowered_function, source_code,
  // source_ptx and runtime_program from compilation result.
  BUILD_INSTRUCTION = 3,
};

// An enum class used to represent the compilation status.
enum class CompilationStatus {
  // An unknown error occurred during compilation.
  UNKNOWN_FAIL = 0,
  // An error occurred during lowering.
  LOWERING_FAIL = 1,
  // An error occurred during codegen and jit.
  CODEGEN_JIT_FAIL = 2,
  // An error occurred during build instruction.
  INSTUCTION_FAIL = 3,
  // An error occurred during build runtime program.
  PROGRAM_FAIL = 4,
  // Compile successfully.
  SUCCESS = 5,
};

struct CompilationContext {
  CompilationContext() = default;
  CompilationContext(const std::shared_ptr<Graph>& graph,
                     const std::shared_ptr<Scope>& scope,
                     const Target& target)
      : graph(graph), scope(scope), target(target) {}

  std::string attached_source_code = "";
  // Compile options.
  bool with_instantiate_variables = false;
  bool with_buffer_handle_instruction_inserted = false;
  bool remove_unused_variables = true;
  // Compile stage, full compile by default.
  CompilationStage stage = CompilationStage::DEFAULT;
  // Compile target.
  Target target;
  // Computation graph.
  std::shared_ptr<Graph> graph;
  // Variable scope
  std::shared_ptr<Scope> scope;
  // Fetch var ids in cinn and the corresponding var nodes will not be fused
  // so as to get the result.
  std::unordered_set<std::string> fetch_var_ids;
  // Map dst reuse var to the src var sharing buffer
  absl::flat_hash_map<std::string, std::string> reuse_vars_map;
  // Nodes group, it may come from the result of op fusion or graph tuning.
  // Nodes in a group will be built into an Instruction.
  std::vector<std::shared_ptr<Graph::Group>> groups;
  // Corresponding lowered functions of above grouped nodes,
  // if it is empty then graph_compiler will generate for them.
  std::vector<std::vector<ir::LoweredFunc>> lowered_funcs;
  // CUDA stream.
  void* stream = nullptr;

  // Set attached source code, if code is not empty, these codes will replace
  // the device_module code after SplitCudaAndHostModule.
  void ApplySourceCode(const std::string& code);
  // Apply results of auto-tune to compile.
  // Compilation will start from CompilationStage::CODEGEN_AND_JIT when tuning
  // results are applied.
  void ApplyTuningResult(const auto_schedule::TuningResult& tuning_result);
};

class GraphCompiler;

class CompilationResult {
  friend class GraphCompiler;

 public:
  void InitCompilationResult(int group_size);

  // Setters
  void SetStatus(int idx, const CompilationStatus& status);
  void SetMessage(int idx, const std::string& message);
  void SetLoweredFuncs(int idx, const std::vector<ir::LoweredFunc>& funcs);
  void SetSourceCode(int idx, const std::string& source_code);
  void SetSourcePtx(int idx, const std::string& source_ptx);
  void SetInstruction(int idx, std::unique_ptr<Instruction> instruction);
  void SetRuntimeProgram(std::unique_ptr<Program> runtime_program);

  // Getters
  bool IsSuccess() const;
  int Size() const { return size_; }
  CompilationStatus Status() const;
  CompilationStatus Status(int idx) const;
  std::string Message() const;
  std::string Message(int idx) const;
  std::vector<std::vector<ir::LoweredFunc>> LoweredFuncs() const;
  std::vector<ir::LoweredFunc> LoweredFuncs(int idx) const;
  std::vector<std::string> SourceCodes() const;
  std::string SourceCode(int idx) const;
  std::vector<std::string> SourcePtxs() const;
  std::string SourcePtx(int idx) const;
  const std::vector<std::unique_ptr<Instruction>>& RuntimeInstructions() const;
  const std::unique_ptr<Instruction>& RuntimeInstruction(int idx) const;
  std::unique_ptr<Program> RuntimeProgram();

 private:
  std::vector<CompilationStatus> status_;
  std::vector<std::string> messages_;
  std::vector<std::optional<std::vector<ir::LoweredFunc>>> lowered_funcs_;
  std::vector<std::optional<std::string>> source_codes_;
  std::vector<std::optional<std::string>> source_ptxs_;
  std::vector<std::unique_ptr<Instruction>> instructions_;
  std::unique_ptr<Program> runtime_program_;
  int size_;
};

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
