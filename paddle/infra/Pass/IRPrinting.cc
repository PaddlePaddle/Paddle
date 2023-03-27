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

#include <unordered_map>

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/SymbolTable.h"

#include "Pass/PassInstrumentation.h"
#include "Pass/PassManager.h"
// #include "utils/xxhash.h"

// #include "paddle/phi/core/enforce.h"

namespace infra {
// A unique fingerprint for a specific operation, and all of it's internal
// operations.
// class IRFingerPrint {
//  public:
//   IRFingerPrint(mlir::Operation *top_op);

//   IRFingerPrint(const IRFingerPrint &) = default;
//   IRFingerPrint &operator=(const IRFingerPrint &) = default;

//   bool operator==(const IRFingerPrint &other) const {
//     return hash == other.hash;
//   }

//   bool operator!=(const IRFingerPrint &other) const {
//     return !(*this == other);
//   }

//  private:
//   XXH64_hash_t hash;
// };

// namespace {
// TODO(liuyuanle): XXH64_update has "Segmentation fault" bug need to be solved!
// template <typename T>
// void UpdateHash(XXH64_state_t *state, const T &data) {
//   XXH64_update(state, &data, sizeof(T));
// }

// template <typename T>
// void UpdateHash(XXH64_state_t *state, T *data) {
//   llvm::outs() << "here 21\n";
//   XXH64_update(state, &data, sizeof(T *));
// }
// }  // namespace

// IRFingerPrint::IRFingerPrint(mlir::Operation *top_op) {
//   XXH64_state_t *const state = XXH64_createState();
//   // PADDLE_ENFORCE_NOT_NULL(
//   //     state,
//   //     phi::errors::PreconditionNotMet(
//   //         "xxhash create state failed, maybe a environment error."));

//   // PADDLE_ENFORCE_NE(
//   //     XXH64_reset(state, XXH64_hash_t(0)),
//   //     XXH_ERROR,
//   //     phi::errors::PreconditionNotMet(
//   //         "xxhash reset state failed, maybe a environment error."));

//   // Hash each of the operations based upon their mutable bits:
//   top_op->walk([&](mlir::Operation *op) {
//     //   - Operation pointer
//     UpdateHash(state, op);
//     //   - Attributes
//     UpdateHash(state, op->getAttrDictionary());
//     //   - Blocks in Regions
//     for (auto &region : op->getRegions()) {
//       for (auto &block : region) {
//         UpdateHash(state, &block);
//         for (auto arg : block.getArguments()) {
//           UpdateHash(state, arg);
//         }
//       }
//     }
//     //   - Location
//     UpdateHash(state, op->getLoc().getAsOpaquePointer());
//     //   - Operands
//     for (auto operand : op->getOperands()) {
//       UpdateHash(state, operand);
//     }
//     //   - Successors
//     for (unsigned i = 0, e = op->getNumSuccessors(); i != e; ++i) {
//       UpdateHash(state, op->getSuccessor(i));
//     }
//   });
//   hash = XXH64_digest(state);
//   XXH64_freeState(state);
// }

namespace {
void PrintIR(mlir::Operation *op,
             bool print_module,
             llvm::raw_ostream &out,
             mlir::OpPrintingFlags flags) {
  // Otherwise, check to see if we are not printing at module scope.
  if (print_module) {
    op->print(out << "\n", flags);
    return;
  }

  // Otherwise, we are printing at module scope.
  out << " ('" << op->getName() << "' operation";
  if (auto symbol_name = op->getAttrOfType<mlir::StringAttr>(
          mlir::SymbolTable::getSymbolAttrName()))
    out << ": @" << symbol_name.getValue();
  out << ")\n";

  // Find the top-level operation.
  auto *top_level_op = op;
  while (auto *parent_op = top_level_op->getParentOp()) {
    top_level_op = parent_op;
  }
  top_level_op->print(out, flags);
}
}  // namespace

class IRPrinter : public PassInstrumentation {
 public:
  explicit IRPrinter(std::unique_ptr<PassManager::IRPrinterConfig> config)
      : config_(std::move(config)){};

  ~IRPrinter() = default;

  void RunBeforePass(Pass *pass, mlir::Operation *op) override {
    if (config_->EnablePrintOnChange()) {
      ir_fingerprints_.emplace(pass, op);
    }
    config_->PrintBeforeIfEnabled(pass, op, [&](llvm::raw_ostream &out) {
      out << "// *** IR Dump Before " << pass->GetPassInfo().name << " ***";
      PrintIR(
          op, config_->EnablePrintModule(), out, config_->GetOpPrintingFlags());
      out << "\n\n";
    });
  }

  void RunAfterPass(Pass *pass, mlir::Operation *op) override {
    if (config_->EnablePrintOnChange()) {
      const auto &fingerprint = ir_fingerprints_.at(pass);
      if (fingerprint == mlir::OperationFingerPrint(op)) {
        ir_fingerprints_.erase(pass);
        return;
      }
      ir_fingerprints_.erase(pass);
    }

    config_->PrintBeforeIfEnabled(pass, op, [&](llvm::raw_ostream &out) {
      out << "// *** IR Dump After " << pass->GetPassInfo().name << " ***";
      PrintIR(
          op, config_->EnablePrintModule(), out, config_->GetOpPrintingFlags());
      out << "\n\n";
    });
  }

 private:
  std::unique_ptr<PassManager::IRPrinterConfig> config_;

  // TODO(liuyuanle): replace mlir::OperationFingerPrint with IRFingerPrint.
  // Pass -> IR fingerprint before pass.
  std::unordered_map<Pass *, mlir::OperationFingerPrint> ir_fingerprints_;
};

void PassManager::EnableIRPrinting(std::unique_ptr<IRPrinterConfig> config) {
  AddInstrumentation(std::make_unique<IRPrinter>(std::move(config)));
}

}  // namespace infra
