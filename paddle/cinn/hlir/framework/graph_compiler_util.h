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

enum class CompilationStage {
  DEFAULT = 0,
  LOWERING = 1,
  CODEGEN_AND_JIT = 2,
  BUILD_INSTRUCTION = 3,
};

enum class CompilationStatus {
  SUCCESS = 0,
  UNKNOWN_FAIL = 1,
  LOWERING_FAIL = 2,
  CODEGEN_JIT_FAIL = 3,
  INSTUCTION_FAIL = 4,
  PROGRAM_FAIL = 5,
};

struct CompilationContext {
  CompilationContext() = default;
  CompilationContext(const std::shared_ptr<Graph>& graph,
                     const std::shared_ptr<Scope>& scope,
                     const Target& target)
      : graph(graph), scope(scope), target(target) {}

  // If attached_source_code is not empty, these code will be compiled directly.
  // When compiling directly, the corresponding stage is
  // CompilationStage::CODEGEN_AND_JIT.
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

  // Apply results of auto-tune to compile.
  void Apply(const auto_schedule::TuningResult& tuning_result);
};

struct CompilationResult {
  CompilationStatus status;
  std::string message;
  std::unique_ptr<Program> runtime_program;
  std::vector<std::vector<ir::LoweredFunc>> lowered_funcs;
  std::vector<std::string> source_codes;
  std::vector<std::string> source_ptxs;
  std::vector<std::unique_ptr<Instruction>> instructions;
};

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
