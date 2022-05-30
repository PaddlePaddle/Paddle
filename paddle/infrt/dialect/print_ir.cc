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
#include <llvm/ADT/Optional.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/ScopedPrinter.h>
#include <llvm/Support/raw_os_ostream.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/Passes.h>
#include <iostream>

#include "paddle/infrt/common/global.h"
#include "paddle/infrt/dialect/init_dialects.h"

namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input toy file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

llvm::raw_ostream &printIndent(int indent = 0) {
  for (int i = 0; i < indent; ++i) llvm::outs() << "    ";
  return llvm::outs();
}

void printOperation(mlir::Operation *op, int indent);
void printRegion(mlir::Region &region, int indent);  // NOLINT
void printBlock(mlir::Block &block, int indent);     // NOLINT

void printOperation(mlir::Operation *op, int indent) {
  llvm::Optional<mlir::ModuleOp> module_op = llvm::None;
  if (llvm::isa<mlir::ModuleOp>(op))
    module_op = llvm::dyn_cast<mlir::ModuleOp>(op);
  llvm::Optional<mlir::FuncOp> func_op = llvm::None;
  if (llvm::isa<mlir::FuncOp>(op)) func_op = llvm::dyn_cast<mlir::FuncOp>(op);

  printIndent(indent) << "op: '" << op->getName();
  // This getName is inherited from Operation::getName
  if (module_op) {
    printIndent() << "@" << module_op->getName();
  }
  // This getName is inherited from SymbolOpInterfaceTrait::getName,
  // which return value of "sym_name" in ModuleOp or FuncOp attributes.
  if (func_op) {
    printIndent() << "@" << func_op->getName();
  }
  printIndent() << "' with " << op->getNumOperands() << " operands"
                << ", " << op->getNumResults() << " results"
                << ", " << op->getAttrs().size() << " attributes"
                << ", " << op->getNumRegions() << " regions"
                << ", " << op->getNumSuccessors() << " successors\n";
  if (!op->getAttrs().empty()) {
    printIndent(indent) << op->getAttrs().size() << " attributes:\n";
    for (mlir::NamedAttribute attr : op->getAttrs()) {
      printIndent(indent + 1) << "- {" << attr.getName() << " : "
                              << attr.getValue() << "}\n";
    }
  }

  if (op->getNumRegions() > 0) {
    printIndent(indent) << op->getNumRegions() << " nested regions:\n";
    for (mlir::Region &region : op->getRegions()) {
      printRegion(region, indent + 1);
    }
  }
}

void printRegion(mlir::Region &region, int indent) {  // NOLINT
  printIndent(indent) << "Region with " << region.getBlocks().size()
                      << " blocks:\n";
  for (mlir::Block &block : region.getBlocks()) {
    printBlock(block, indent + 1);
  }
}

void printBlock(mlir::Block &block, int indent) {  // NOLINT
  printIndent(indent) << "Block with " << block.getNumArguments()
                      << " arguments"
                      << ", " << block.getNumSuccessors() << " successors"
                      << ", " << block.getOperations().size()
                      << " operations\n";

  for (mlir::Operation &operation : block.getOperations()) {
    printOperation(&operation, indent + 1);
  }
}

int main(int argc, char **argv) {
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  cl::ParseCommandLineOptions(argc, argv, "mlir demo");

  mlir::DialectRegistry registry;
  infrt::registerCinnDialects(registry);
  mlir::MLIRContext context(registry);
  // mlir will verify module automatically after parsing.
  // https://github.com/llvm/llvm-project/blob/38d18d93534d290d045bbbfa86337e70f1139dc2/mlir/lib/Parser/Parser.cpp#L2051
  // mlir::OwningModuleRef module_ref = mlir::parseSourceString(mlir_source,
  // context);
  mlir::OwningModuleRef module_ref =
      mlir::parseSourceFile(inputFilename, &context);
  std::cout << "----------print IR Structure begin----------" << std::endl;
  printOperation(module_ref->getOperation(), 0);
  std::cout << "----------print IR Structure end----------" << std::endl;

  module_ref->dump();
  return 0;
}
