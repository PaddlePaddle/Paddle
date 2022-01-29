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
#include <llvm/ADT/ArrayRef.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Region.h>
#include <memory>
#include <string>
#include <unordered_map>

namespace mlir {
class FuncOp;
}  // namespace mlir

namespace infrt {
namespace host_context {

class SymbolTable;
class KernelRegistry;
class KernelFrame;
class Value;
class CoreRuntimeBuilder;
class MlirFunctionExecutable;

/**
 * OpExecutable is a runtime executable instance for an operation. It captures
 * all the information(Tensors, attributes
 * and so on) needed for execution.
 * With the SymbolTable and op definition, it create and hold a KernelFrame once
 * and execute any times.
 */
class OpExecutable {
 public:
  KernelFrame& frame();
  const KernelFrame& frame() const;

  void Execute();

  const std::string& name() const;

  ~OpExecutable();

 protected:
  class Impl;
  explicit OpExecutable(Impl* impl);

  std::unique_ptr<Impl> impl_;
};

/**
 * Builder to help contruct an OpExecutable.
 */
class OpExecutableBuilder : public OpExecutable {
 public:
  using function_defs_t = std::unordered_map<std::string, mlir::FuncOp>;

  OpExecutableBuilder(const std::string& op_name,
                      SymbolTable* symbol_table,
                      KernelRegistry* kernel_registry = nullptr);
  OpExecutableBuilder(OpExecutableBuilder&& other);

  void AppendArgument(const std::string& name);
  void AppendArgument(Value* value);

  void SetResults(llvm::ArrayRef<std::string> result_names);
  void SetResults(llvm::ArrayRef<Value*> results);

  void AppendAttribute(Value* value);

  MlirFunctionExecutable* CreateFunctionExecutable(
      mlir::FuncOp op, function_defs_t* function_defs);

  MlirFunctionExecutable* CreateFunctionExecutable(
      mlir::Region* region,
      mlir::FunctionType func_type,
      function_defs_t* function_defs);
};

}  // namespace host_context
}  // namespace infrt
