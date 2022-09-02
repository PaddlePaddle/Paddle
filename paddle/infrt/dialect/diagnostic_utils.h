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
#include <llvm/Support/SourceMgr.h>
#include <mlir/IR/Diagnostics.h>

#include <memory>

namespace infrt {
namespace dialect {

/**
 * A scoped diagnostic handler to help debug MLIR process.
 */
class MyScopedDiagnosicHandler : public mlir::SourceMgrDiagnosticHandler {
 public:
  MyScopedDiagnosicHandler(mlir::MLIRContext* ctx, bool propagate);

  mlir::LogicalResult handler(mlir::Diagnostic* diag);

  ~MyScopedDiagnosicHandler();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace dialect
}  // namespace infrt
