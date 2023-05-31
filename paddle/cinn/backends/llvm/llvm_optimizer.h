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

#include <llvm/IR/Instruction.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/Pass.h>
#include <llvm/Target/TargetMachine.h>

#include <functional>

namespace cinn::backends {

// TODO(fc500110): define class OptimizeOptions

// llvm module optimizer
class LLVMModuleOptimizer final {
 public:
  explicit LLVMModuleOptimizer(llvm::TargetMachine *machine,
                               int opt_level,
                               llvm::FastMathFlags fast_math_flags,
                               bool print_passes = false);
  void operator()(llvm::Module *m);

 private:
  llvm::TargetMachine *machine_;
  int opt_level_{};
  bool print_passes_{};
};
}  // namespace cinn::backends
