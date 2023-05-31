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

#include "cinn/auto_schedule/tuning.h"
#include "cinn/backends/compiler.h"
#include "cinn/backends/cuda_util.h"
#include "cinn/common/macros.h"
#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/instruction.h"
#include "cinn/hlir/framework/op_strategy.h"
#include "cinn/hlir/framework/parallel_compiler.h"
#include "cinn/hlir/framework/scope.h"
#include "cinn/ir/lowered_func.h"
#include "cinn/lang/packed_func.h"
#include "cinn/utils/timer.h"

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
      : target_(std::move(target)),
        scope_(scope),
        graph_(graph),
        m_builder_(UniqName("module"), target) {}

  struct CompilationResult {
    std::unique_ptr<Program> runtime_program;
  };

  struct CompileOptions {
    std::string attached_code = "";
    bool with_instantiate_variables = false;
    bool with_buffer_handle_instruction_inserted = false;
    bool remove_unused_variables = true;
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
  void ExportObject(const std::string& path) { compiler_->ExportObject(path); }

  std::unique_ptr<Program> Build(const std::string& code = "");

  std::string GenSourceCode();

  void PrintFunc();

  const std::shared_ptr<Scope>& GetScope() const { return scope_; }

 private:
  std::vector<ir::LoweredFunc> GetOpFunc(const std::vector<Node*>& nodes);

  std::vector<ir::LoweredFunc> GetOpFunc(const Node* node);
  // Given a node, lower it to LoweredFunc using new ir schedule
  std::vector<ir::LoweredFunc> GetOpFuncWithIRSchedule(
      const Node* node,
      const absl::flat_hash_map<std::string, Type>& type_dict_,
      const absl::flat_hash_map<std::string, shape_t>& shape_dict_);

  std::string GenOpFuncName(const Node* node) const {
    return "fn_" + node->id();
  }

  // append a unique number at the end of the function name to distinguish
  // different functions from graphs whose structures are same
  const std::string& GetOrGenFullFuncName(const std::string& prefix);

  // TODO(haozech) add implementation
  std::vector<std::string> OpGetInputNames(const Node* node) const;
  // TODO(haozech) add implementation
  std::vector<std::string> OpGetOutputNames(const Node* node) const;

  std::vector<std::unique_ptr<Instruction>> BuildInstructions(
      const std::vector<std::vector<Node*>>& groups,
      const std::vector<std::shared_ptr<Graph::Group>>& fusion_groups);

  void BuildCublasInstr(const Node& node, Instruction* instr) const;
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

  void ProcessFunction(const std::vector<ir::LoweredFunc>& lowered_funcs);
  void SetSubKernels(Instruction* instr, const std::string& func_name);
  Target target_;
  std::shared_ptr<Graph> graph_;
  std::shared_ptr<Scope> scope_;
  // mapping a function's name to its input artuments' names
  std::map<std::string, std::vector<std::string>> function2input_args_;
  // mapping a function's name to its output artuments' names
  std::map<std::string, std::vector<std::string>> function2output_args_;
  // fetch var ids in cinn and the corresponding var nodes will not be fused so
  // as to get the result
  std::unordered_set<std::string> fetch_var_ids_;

  absl::flat_hash_map<std::string, std::string> prefix2full_namemap_;
  // map dst reuse var to the src var sharing buffer
  absl::flat_hash_map<std::string, std::string> reuse_vars_map_;

  std::unique_ptr<backends::Compiler> compiler_;
  CompileOptions compile_options_;

  ir::Module::Builder m_builder_;

  CINN_DISALLOW_COPY_AND_ASSIGN(GraphCompiler);
};

std::shared_ptr<Scope> BuildScope(Target target,
                                  const std::shared_ptr<Graph>& graph,
                                  std::shared_ptr<Scope> scope = nullptr);

// Given params, lower the op to LoweredFunc using new IR Schedule
std::vector<ir::LoweredFunc> GetFuncFromImpl(
    const std::shared_ptr<OpImpl>& impl,
    const common::CINNValuePack& cinn_inputs,
    std::vector<ir::Tensor>& tensor_inputs,
    const std::vector<std::string>& input_output_nodes,
    const std::string& node_id,
    const Target& target);

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
