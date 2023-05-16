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

namespace detail {
class AdaptorPass;
}  // namespace detail

//==----==//
//
//==----==//

class PassManager {
 public:
  ~PassManager();

  explicit PassManager(mlir::MLIRContext *context, int opt_level = 2);

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

  mlir::MLIRContext *GetContext() const { return context_; }

  mlir::LogicalResult Run(mlir::Operation *op);

  void addPass(std::unique_ptr<Pass> pass) {
    passes_.emplace_back(std::move(pass));
  }

  void EnableTiming();

  class IRPrinterConfig {
   public:
    using PrintCallBack = std::function<void(llvm::raw_ostream &)>;

    explicit IRPrinterConfig(
        const std::function<bool(Pass *, mlir::Operation *)>
            &enable_print_before =
                [](Pass *, mlir::Operation *) { return true; },
        const std::function<bool(Pass *, mlir::Operation *)> &
            enable_print_after = [](Pass *, mlir::Operation *) { return true; },
        bool print_module = true,
        bool print_on_change = true,
        llvm::raw_ostream &out = llvm::outs(),
        mlir::OpPrintingFlags op_printing_flags = mlir::OpPrintingFlags())
        : enable_print_before_(enable_print_before),
          enable_print_after_(enable_print_after),
          print_module_(print_module),
          print_on_change_(print_on_change),
          out_(out),
          op_printing_flags_(op_printing_flags) {
      assert((enable_print_before_ || enable_print_after_) &&
             "expected at least one valid filter function");
    }

    ~IRPrinterConfig() = default;

    void PrintBeforeIfEnabled(Pass *pass,
                              mlir::Operation *op,
                              const PrintCallBack &print_callback) {
      if (enable_print_before_ && enable_print_before_(pass, op)) {
        print_callback(out_);
      }
    }

    void PrintAfterIfEnabled(Pass *pass,
                             mlir::Operation *op,
                             const PrintCallBack &print_callback) {
      if (enable_print_after_ && enable_print_after_(pass, op)) {
        print_callback(out_);
      }
    }

    bool EnablePrintModule() const { return print_module_; }

    bool EnablePrintOnChange() const { return print_on_change_; }

    mlir::OpPrintingFlags GetOpPrintingFlags() const {
      return op_printing_flags_;
    }

   private:
    // The enable_print_before_ and enable_print_after_ can be used to specify
    // the pass to be printed. The default is to print all passes.
    std::function<bool(Pass *, mlir::Operation *)> enable_print_before_;
    std::function<bool(Pass *, mlir::Operation *)> enable_print_after_;

    bool print_module_;
    bool print_on_change_;

    // TODO(liuyuanle): Replace it with a local implementation.
    // The stream to output to.
    llvm::raw_ostream &out_;
    // Flags to control printing behavior.
    mlir::OpPrintingFlags op_printing_flags_;
  };

  void EnableIRPrinting(std::unique_ptr<IRPrinterConfig> config);

  void AddInstrumentation(std::unique_ptr<PassInstrumentation> pi);

 private:
  mlir::LogicalResult RunPasses(mlir::Operation *op, AnalysisManager am);

  mlir::LogicalResult RunWithCrashRecovery(mlir::Operation *op,
                                           AnalysisManager am);

  mlir::LogicalResult Initialize(mlir::MLIRContext *context);

 private:
  mlir::MLIRContext *context_;

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
