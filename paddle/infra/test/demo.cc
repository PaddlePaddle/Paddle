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

#include "Pass/Pass.h"
#include "Pass/PassManager.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "test/helper.h"

static llvm::cl::opt<std::string> inputFilename(
    llvm::cl::Positional,
    llvm::cl::desc("<input file>"),
    llvm::cl::init("-"),
    llvm::cl::value_desc("filename"));

class TestPass : public infra::Pass {
 public:
  void Run(mlir::Operation* op) override {
    llvm::outs() << "In TestPass: " << op->getName() << "\n";

    for (auto& region : op->getRegions()) {
      for (auto& block : region.getBlocks()) {
        for (auto& iop : block.getOperations()) {
          llvm::outs() << "  visit " << iop.getName() << "\n";
        }
      }
    }
  }

  bool CanScheduleOn(mlir::Operation* op) const override {
    return op->getNumRegions() > 0 &&
           op->getName().getStringRef() != "builtin.module";
  }
};

int main(int argc, char** argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "...");

  mlir::MLIRContext context;
  mlir::registerAllDialects(context);
  context.allowsUnregisteredDialects();

  mlir::OwningOpRef<mlir::ModuleOp> module = LoadMLIR(context, inputFilename);
  llvm::outs() << "src mod\n";
  module->dump();

  infra::PassManager pm(&context);
  auto pass = std::make_unique<TestPass>();
  pm.addPass(std::move(pass));

  (void)pm.Run(module->getOperation());
  return 0;
}
