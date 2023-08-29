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

#include <memory>

#include "paddle/cinn/hlir/framework/instruction.h"
#include "paddle/cinn/hlir/framework/scope.h"

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

  const std::vector<std::unique_ptr<Instruction>>& GetPreRunInstructions()
      const {
    return prerun_instrs_;
  }
  const std::vector<std::unique_ptr<Instruction>>& GetRunInstructions() const {
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

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
