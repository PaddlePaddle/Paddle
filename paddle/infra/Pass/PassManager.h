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

/// The code and design is mainly from mlir, very thanks to the great project.

#include <memory>
#include <set>
#include "Pass/Pass.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator.h"
#include "llvm/ADT/iterator_range.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LogicalResult.h"

namespace infra {

class PassInstrumentation;
class PassInstrumentor;
class AnalysisManager;
class PassManager;

namespace detail {
class AdaptorPass;
}  // namespace detail

//==----==//
//
//==----==//

class PassManager {
 public:
  ~PassManager();

  explicit PassManager(mlir::MLIRContext* context, int opt_level = 2);

  using pass_iterator = llvm::pointee_iterator<
      llvm::MutableArrayRef<std::unique_ptr<Pass>>::iterator>;
  pass_iterator begin() {
    return llvm::MutableArrayRef<std::unique_ptr<Pass>>{passes_}.begin();
  }
  pass_iterator end() {
    return llvm::MutableArrayRef<std::unique_ptr<Pass>>{passes_}.end();
  }
  llvm::iterator_range<pass_iterator> GetPasses() { return {begin(), end()}; }

  using const_pass_iterator = llvm::pointee_iterator<
      llvm::ArrayRef<std::unique_ptr<Pass>>::const_iterator>;
  const_pass_iterator begin() const {
    return llvm::ArrayRef<std::unique_ptr<Pass>>{passes_}.begin();
  }
  const_pass_iterator end() const {
    return llvm::ArrayRef<std::unique_ptr<Pass>>{passes_}.end();
  }
  llvm::iterator_range<const_pass_iterator> GetPasses() const {
    return {begin(), end()};
  }

  bool empty() const { return begin() == end(); }

  mlir::MLIRContext* GetContext() const { return context_; }

  mlir::LogicalResult Run(mlir::Operation* op);

  void addPass(std::unique_ptr<Pass> pass) {
    passes_.emplace_back(std::move(pass));
  }

  void EnableTiming();

  void AddInstrumentation(std::unique_ptr<PassInstrumentation> pi);

 private:
  mlir::LogicalResult RunPasses(mlir::Operation* op, AnalysisManager am);

  mlir::LogicalResult RunWithCrashRecovery(mlir::Operation* op,
                                           AnalysisManager am);

  mlir::LogicalResult Initialize(mlir::MLIRContext* context);

 private:
  mlir::MLIRContext* context_;

  std::unique_ptr<PassInstrumentor> instrumentor_;

  int opt_level_;

  std::vector<std::unique_ptr<Pass>> passes_;

  // Used to apply pass manager on all nested ir.
  std::unique_ptr<Pass> adaptor_pass_;
  friend class detail::AdaptorPass;

  llvm::hash_code init_key_ =
      llvm::DenseMapInfo<llvm::hash_code>::getTombstoneKey();

  bool verify_{true};
};

}  // namespace infra
