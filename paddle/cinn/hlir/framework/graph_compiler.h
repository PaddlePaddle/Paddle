// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include <absl/container/flat_hash_map.h>

#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/cinn/auto_schedule/tuning.h"
#include "paddle/cinn/backends/compiler.h"
#include "paddle/cinn/backends/cuda_util.h"
#include "paddle/cinn/common/macros.h"
#include "paddle/cinn/hlir/framework/graph.h"
#include "paddle/cinn/hlir/framework/instruction.h"
#include "paddle/cinn/hlir/framework/op_strategy.h"
#include "paddle/cinn/hlir/framework/parallel_compiler.h"
#include "paddle/cinn/hlir/framework/scope.h"
#include "paddle/cinn/ir/lowered_func.h"
#include "paddle/cinn/lang/packed_func.h"
#include "paddle/cinn/utils/timer.h"

namespace cinn {
namespace hlir {
namespace framework {

/**
 * The Program is the runtime instance for running a computation.
 */
class Program {
 public:
  /**
   * Constructor.
   * @param scope The scope containing all the runtime variables.
   * @param instrs The instructions belonging to this program.
   */
  Program(const std::shared_ptr<Scope>& scope,
          std::vector<std::unique_ptr<Instruction>>&& instrs);

  void PreRun(
      const std::map<std::string, cinn_pod_value_t>* name2podargs = nullptr);

  void Export(const std::vector<std::string>& persistent_vars,
              const std::string& filename);

  /**
   * Execute the program -- that is running all the instructions inside it.
   */
  void Execute(
      const std::map<std::string, cinn_pod_value_t>* name2podargs = nullptr,
      void* stream = nullptr,
      bool use_cache = true);

  void ExecuteTest(int repeat_);

  /**
   * Get the number of instructions.
   */
  size_t size() const { return instrs_.size(); }

  const std::vector<std::unique_ptr<Instruction>>& GetPreRunInstructions() {
    return prerun_instrs_;
  }
  const std::vector<std::unique_ptr<Instruction>>& GetRunInstructions() {
    return instrs_;
  }

 private:
  // We need to hold scope to assure tensors alive used in instructions.
  std::shared_ptr<Scope> scope_;
  // prerun instructions
  std::vector<std::unique_ptr<Instruction>> prerun_instrs_;
  // only runtime instructions
  std::vector<std::unique_ptr<Instruction>> instrs_;
};

/**
 * GraphCompiler compiles a graph and generate the runtime Program.
 */
class GraphCompiler final {
 public:
  GraphCompiler(Target target,
                const std::shared_ptr<Scope>& scope,
                const std::shared_ptr<Graph>& graph)
      : target_(std::move(target)), scope_(scope), graph_(graph) {}

  struct CompilationResult {
    std::unique_ptr<Program> runtime_program;
    std::vector<std::vector<ir::LoweredFunc>> lowered_funcs;
    std::vector<std::string> source_codes;
    std::vector<std::string> source_ptxs;
    std::vector<std::unique_ptr<Instruction>> instructions;
  };

  struct CompileOptions {
    std::string attached_code = "";
    bool with_instantiate_variables = false;
    bool with_buffer_handle_instruction_inserted = false;
    bool remove_unused_variables = true;
    // Compile stage
    ParallelCompiler::Stage stage = ParallelCompiler::Stage::DEFAULT;
    // nodes group, it may come from the result of op fusion or graph tuning.
    // nodes in a group will be built into an Instruction
    std::vector<std::shared_ptr<Graph::Group>> groups;
    // corresponding LoweredFuncs of above grouped nodes,
    // if it is empty then graph_compiler will generate for them
    std::vector<std::vector<ir::LoweredFunc>> lowered_funcs;

    // apply results of auto-tune to compile
    void Apply(const auto_schedule::TuningResult& tuning_result);
  };

  // Compile with a packing option and result, to be extended easily.
  CompilationResult Build(const CompileOptions& options,
                          std::unordered_set<std::string>&& fetch_var_ids = {},
                          void* stream = nullptr);

  std::unique_ptr<Program> Build(const std::string& code = "");

  CompilationResult Lowering(
      const CompileOptions& options,
      std::unordered_set<std::string>&& fetch_var_ids = {},
      void* stream = nullptr);

  CompilationResult CodegenAndJit(
      const CompileOptions& options,
      std::unordered_set<std::string>&& fetch_var_ids = {},
      void* stream = nullptr);

  CompilationResult BuildInstruction(
      const CompileOptions& options,
      std::unordered_set<std::string>&& fetch_var_ids = {},
      void* stream = nullptr);

  const std::shared_ptr<Scope>& GetScope() const { return scope_; }

 private:
  // instantiate all variables on compile time
  void InstantiateVariables();

  // some variables are eliminated by optimized passes(such as OpFusion),
  // we can filter out them according to arguments of the built instructions,
  // and erase them from the scope to avoid unnecessary buffer allocation
  void RemoveInvalidVariables(
      const std::vector<std::unique_ptr<Instruction>>& instructions);

  // find the first and last instruction where a variable used, and mark the
  // variable should allocate buffer before the first instruction runing and
  // can release the buffer after the last instruction finished.
  void AnalyzeVariableLifeTime(
      const std::vector<std::unique_ptr<Instruction>>& instructions,
      std::unordered_map<int, std::vector<std::string>>* step2malloc,
      std::unordered_map<int, std::vector<std::string>>* step2free);

  // insert a buffer malloc instruction applying on variables before they are
  // firstly used in the next instruction, and insert a buffer free instruction
  // applying on variables after no instruction will use them anymore
  void InsertBufferHandlers(
      std::vector<std::unique_ptr<Instruction>>* instructions);

 private:
  // parallel compiler
  std::shared_ptr<ParallelCompiler> parallel_compiler_;

  Target target_;
  std::shared_ptr<Graph> graph_;
  std::shared_ptr<Scope> scope_;
  // fetch var ids in cinn and the corresponding var nodes will not be fused so
  // as to get the result
  std::unordered_set<std::string> fetch_var_ids_;

  // map dst reuse var to the src var sharing buffer
  absl::flat_hash_map<std::string, std::string> reuse_vars_map_;

  CINN_DISALLOW_COPY_AND_ASSIGN(GraphCompiler);
};

std::shared_ptr<Scope> BuildScope(Target target,
                                  const std::shared_ptr<Graph>& graph,
                                  std::shared_ptr<Scope> scope = nullptr);

// Given params, lower the op to LoweredFunc using new IR Schedule
std::vector<ir::LoweredFunc> GetFuncFromImpl(
    const std::shared_ptr<OpImpl>& impl,
    const common::CINNValuePack& cinn_inputs,
    std::vector<ir::Tensor>& tensor_inputs,  // NOLINT
    const std::vector<std::string>& input_output_nodes,
    const std::string& node_id,
    const Target& target);

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
