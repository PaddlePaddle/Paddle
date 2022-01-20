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

#include "paddle/infrt/dialect/mlir_loader.h"

#include <llvm/Support/SourceMgr.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Parser.h>
#include <unordered_map>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "paddle/infrt/dialect/diagnostic_utils.h"
#include "paddle/infrt/dialect/init_infrt_dialects.h"

namespace infrt {
namespace dialect {

mlir::OwningModuleRef LoadMlirSource(mlir::MLIRContext* context,
                                     const std::string& mlir_source) {
  // context->allowUnregisteredDialects();
  mlir::DialectRegistry registry;
  registerCinnDialects(registry);
  context->appendDialectRegistry(registry);
  // Currenetly, We only used the CinnDialect and mlir::BuiltinDialect is
  // enough。Don't need StandardOpsDialect.
  // context->getDialectRegistry().insert<mlir::StandardOpsDialect>();

  mlir::ScopedDiagnosticHandler scope_handler(
      context, [](mlir::Diagnostic& diag) {
        if (diag.getSeverity() != mlir::DiagnosticSeverity::Error)
          return mlir::success();
        LOG(INFO) << "diag: " << diag.str();
        return mlir::failure(true);
      });

  auto res = mlir::parseSourceString(
      llvm::StringRef(mlir_source.data(), mlir_source.length()), context);
  CHECK(*res) << "failed to parse MLIR string";
  return res;
}

mlir::OwningModuleRef LoadMlirFile(const std::string& file_name,
                                   mlir::MLIRContext* context) {
  // context->allowUnregisteredDialects();
  mlir::DialectRegistry registry;
  registerCinnDialects(registry);
  context->appendDialectRegistry(registry);
  mlir::ScopedDiagnosticHandler scope_handler(
      context, [](mlir::Diagnostic& diag) {
        if (diag.getSeverity() != mlir::DiagnosticSeverity::Error)
          return mlir::success();
        LOG(INFO) << "diag: " << diag.str();
        return mlir::failure(true);
      });

  return mlir::parseSourceFile(std::string(file_name), context);
}

}  // namespace dialect
}  // namespace infrt
