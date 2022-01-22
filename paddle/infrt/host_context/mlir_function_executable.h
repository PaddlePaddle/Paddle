// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Region.h>

#include <string>
#include <unordered_map>

#include "paddle/infrt/host_context/core_runtime.h"
#include "paddle/infrt/host_context/function.h"
#include "paddle/infrt/host_context/mlir_to_runtime_translate.h"

namespace infrt {
namespace host_context {

struct KernelRegistry;

/**
 * Executable function for a given MLIR function definition, mainly used in two
 * scenerios:
 * 1. infrt.call op
 * 2. main function call
 *
 * A MlirFunctionExecutable might have one or more arguments and results.
 */
class MlirFunctionExecutable : public Function, public MlirToRuntimeTranslator {
 public:
  using function_defs_t = std::unordered_map<std::string, mlir::FuncOp>;

  MlirFunctionExecutable(mlir::FuncOp func_op,
                         KernelRegistry* kernel_registry,
                         function_defs_t& function_table);  // NOLINT

  MlirFunctionExecutable(
      mlir::Region* region,
      mlir::FunctionType func_type,
      KernelRegistry* kernel_registry,
      MlirToRuntimeTranslator::function_defs_t& function_table);  // NOLINT

  /**
   * Execute the function with the given arguments and results.
   * NOTE the \param arguments and \param results should not be altered.
   */
  void Execute(llvm::ArrayRef<Value*> arguments,
               llvm::MutableArrayRef<ValueRef> results,
               bool is_region = false) const;

 private:
  /**
   * Build the runtime executables once the function call arguments and results
   * are passed in.
   * This will trigger in the first execution.
   */
  void BuildExecutables(llvm::ArrayRef<Value*> arguments,
                        llvm::MutableArrayRef<ValueRef> results,
                        bool is_region);

 private:
  mlir::Region* region_{};
  CoreRuntimeBuilder core_runtime_builder_;
  MlirToRuntimeTranslator::function_defs_t& function_table_;
  std::function<void()> copy_res_fn_;
};

}  // namespace host_context
}  // namespace infrt
