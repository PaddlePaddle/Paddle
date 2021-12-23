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

#include <llvm/ADT/SmallVector.h>

#include <boost/optional.hpp>
#include <memory>         // NOLINT
#include <string>         //NOLINT
#include <unordered_map>  // NOLINT

namespace mlir {
class FuncOp;
class ModuleOp;
class Operation;
class Attribute;
class Value;
}  // namespace mlir

namespace infrt::host_context {

class CoreRuntimeBuilder;
class Value;
class ValueRef;
class KernelRegistry;

/**
 * MlirToRuntimeTranslator helps to translate a MLIR program to a CoreRuntime.
 * This is the base class of all the modules those parse a MLIR program and
 * finally generate a CoreRuntime.
 */
class MlirToRuntimeTranslator {
 public:
  //! Holds all the function definitions.
  using function_defs_t = std::unordered_map<std::string, mlir::FuncOp>;

  explicit MlirToRuntimeTranslator(CoreRuntimeBuilder* runtime);
  MlirToRuntimeTranslator(mlir::ModuleOp module, CoreRuntimeBuilder* runtime);

  void Run() { EmitFunctions(); }

  virtual ~MlirToRuntimeTranslator();

 protected:
  //! Emit a "infrt.constant.*" operation, return true if succeed.
  bool EmitConstantOp(mlir::Operation* op);
  //! Emit a "infrt.return" operation.
  bool EmitReturnOp(mlir::Operation* op,
                    llvm::SmallVectorImpl<mlir::Value>* results);
  //! Emit a "ts.build_shape" operation.
  bool EmitBuildShapeOp(mlir::Operation* op);
  //! Emit an operation other than the special cases above.
  bool EmitGeneralOp(mlir::Operation* op);
  //! Emit all the functions.
  bool EmitFunctions();

  //! Emit a single function, this is an API that should be implemented by
  //! inherients.
  virtual void EmitFunction(mlir::FuncOp op);

  bool EmitCallOp(mlir::Operation* op, function_defs_t* function_table);

  template <typename T>
  boost::optional<T> EmitAttribute(const mlir::Attribute* attr);

  Value* GetOpResult(mlir::Operation* op);

  Value* GetValue(mlir::Value value);

  Value* AddValue(mlir::Value value);

  Value* AddValue(mlir::Value mlir_value, Value* value);

  void UpdateCurFuncName(const std::string& name);

 protected:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/**
 * Build a CoreRuntime from a MLIR module.
 */
void MlirToRuntimeTranslate(mlir::ModuleOp module, CoreRuntimeBuilder* runtime);

/**
 * Execute a MLIR program, that is execute all the functions without input
 * arguments.
 * This is mainly used by testcase.
 * @param module a MLIR module.
 * @param registry the kernel registry containing all the valid kernels.
 */
void TestMlir(mlir::ModuleOp module, KernelRegistry* registry);

}  // namespace infrt::host_context
