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

#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Function.h>
#include <mlir/IR/Module.h>
#include <mlir/IR/OperationSupport.h>
#include <unordered_map>

#include <memory>
#include <string>

#include "paddle/infrt/host_context/core_runtime.h"
#include "paddle/infrt/host_context/kernel_registry.h"
#include "paddle/infrt/host_context/mlir_function_executable.h"
#include "paddle/infrt/host_context/mlir_to_runtime_translate.h"
#include "paddle/infrt/host_context/op_executable.h"

namespace infrt {
namespace host_context {

/**
 * This get a MLIR program as input, it compiles it into runtime program, and
 * one can retrieve the function and execute
 * it by passing the input arguments.
 */
class MlirProgramExecutor : public MlirToRuntimeTranslator {
 public:
  CoreRuntimeBuilder runtime_builder;
  mlir::ModuleOp module;
  function_defs_t function_defs;

  MlirProgramExecutor(mlir::ModuleOp module, KernelRegistry* registry)
      : MlirToRuntimeTranslator(module, &runtime_builder),
        runtime_builder(registry),
        module(module) {}

  // Build functions and generate executables.
  void BuildFunctions() { EmitFunctions(); }

  void EmitFunction(mlir::FuncOp op) override {
    LOG(INFO) << "Emit function: " << op.getName().str();
    function_defs[op.getName().str()] = op;

    func_executables_.emplace(
        op.getName().str(),
        new MlirFunctionExecutable(
            op, runtime_builder.kernel_registry(), function_defs));
  }

  MlirFunctionExecutable* LookupFunc(const std::string& name) {
    auto it = func_executables_.find(name);
    if (it != func_executables_.end()) {
      return it->second.get();
    }
    return nullptr;
  }

 private:
  std::unordered_map<std::string, std::unique_ptr<MlirFunctionExecutable>>
      func_executables_;
};

}  // namespace host_context
}  // namespace infrt
