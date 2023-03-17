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

#include <memory>
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "test/helper.h"

// pdll inc
#include "test/patterns.h.pdll.inc"

// td inc
static mlir::Value UpdateAddOp(mlir::OpBuilder builder,
                               mlir::Location loc,
                               mlir::Value x,
                               mlir::Value c1,
                               mlir::Value c2) {
  auto add_const_op =
      builder.create<mlir::tosa::AddOp>(loc, c1.getType(), c1, c2);
  auto add_op = builder.create<mlir::tosa::AddOp>(
      loc, x.getType(), x, add_const_op.getOutput());
  return add_op.getOutput();
}
#include "test/patterns.h.td.inc"

static llvm::cl::opt<std::string> inputFilename(
    llvm::cl::Positional,
    llvm::cl::desc("<input file>"),
    llvm::cl::init("-"),
    llvm::cl::value_desc("filename"));

enum GenType { PDLL, TD, CPP };
static llvm::cl::opt<enum GenType> genType(
    "gen",
    llvm::cl::init(PDLL),
    llvm::cl::desc("td or pdll or c++"),
    llvm::cl::values(clEnumValN(PDLL, "pdll", "pdll gen")),
    llvm::cl::values(clEnumValN(TD, "td", "td gen")),
    llvm::cl::values(clEnumValN(CPP, "cpp", "c++")));

//==----------------------------------------------==//
// Just test for AnalysisManager
//==----------------------------------------------==//
struct TestAnalysis {
  explicit TestAnalysis(mlir::Operation* op) {}
  int num = 0;
};

class MLIRPass : public mlir::PassWrapper<MLIRPass, mlir::OperationPass<>> {
 public:
  llvm::StringRef getName() const override { return "MLIRPass"; }

 protected:
  void runOnOperation() override {
    llvm::outs() << "MLIRPass visit "
                 << getOperation()->getName().getStringRef() << "\n";
    auto& a = getAnalysis<TestAnalysis>();
    llvm::outs() << "num is " << a.num << "\n";
    a.num++;

    markAnalysesPreserved<TestAnalysis>();
  }
};

class MLIRPass2 : public mlir::PassWrapper<MLIRPass2, mlir::OperationPass<>> {
 public:
  llvm::StringRef getName() const override { return "MLIRPass2"; }

 protected:
  void runOnOperation() override {
    llvm::outs() << "MLIRPass2 visit "
                 << getOperation()->getName().getStringRef() << "\n";
    auto a = getCachedAnalysis<TestAnalysis>();
    if (a) {
      llvm::outs() << "get Cache, num is " << a->get().num << "\n";
    } else {
      llvm::outs() << "MLIRPass2 getCachedAnalysis failed\n";
    }

    auto& b = getAnalysis<TestAnalysis>();
    llvm::outs() << "b num is " << b.num << "\n";
  }
};

//==----------------------------------------------==//
// Just test for Recursive Pattern
//==----------------------------------------------==//
class AddPattern : public mlir::OpRewritePattern<mlir::tosa::AddOp> {
 public:
  using mlir::OpRewritePattern<mlir::tosa::AddOp>::OpRewritePattern;

  void initialize() {
    /// Signal that this pattern safely handles recursive application.
    /// It is only used in DialectConversion.
    setHasBoundedRewriteRecursion();
  }

  mlir::LogicalResult matchAndRewrite(
      mlir::tosa::AddOp op,
      mlir::PatternRewriter& rewriter) const final {  // NOLINT
    auto add_op = llvm::dyn_cast_or_null<mlir::tosa::AddOp>(
        op->getOperand(0).getDefiningOp());
    if (!add_op) return mlir::failure();

    auto const_op = llvm::dyn_cast_or_null<mlir::tosa::ConstOp>(
        op.getOperand(1).getDefiningOp());
    if (!const_op) return mlir::failure();

    auto const2_op = llvm::dyn_cast_or_null<mlir::tosa::ConstOp>(
        add_op->getOperand(1).getDefiningOp());
    if (!const2_op) return mlir::failure();

    auto in = add_op.getOperand(0);

    auto c1 = const_op.getValue();
    auto c2 = const2_op.getValue();

    if (c1.getNumElements() != 1) return mlir::failure();
    if (c2.getNumElements() != 1) return mlir::failure();

    auto v1 = *c1.getValues<float>().begin();
    auto v2 = *c2.getValues<float>().begin();
    auto v3 = v1 + v2;

    mlir::MLIRContext* ctx = op.getContext();
    auto new_attr = mlir::DenseElementsAttr::get(
        mlir::VectorType::get({1}, mlir::Float32Type::get(ctx))
            .cast<mlir::ShapedType>(),
        llvm::ArrayRef<float>{v3});
    auto new_cst_op = rewriter.create<mlir::tosa::ConstOp>(
        add_op->getLoc(), const_op->getResult(0).getType(), new_attr);
    auto new_add_op =
        rewriter.create<mlir::tosa::AddOp>(add_op->getLoc(),
                                           add_op.getResult().getType(),
                                           in,
                                           new_cst_op->getResult(0));

    rewriter.replaceOp(op, new_add_op->getResults());
    return mlir::success();
  }
};

class FusionPass
    : public mlir::PassWrapper<FusionPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
 public:
 protected:
  void runOnOperation() override {
    mlir::RewritePatternSet patterns(getOperation()->getContext());

    if (genType == CPP) {
      // C++ naive impl.
      patterns.add<AddPattern>(&getContext());
    } else if (genType == PDLL) {
      // PDLL impl.
      populateGeneratedPDLLPatterns(patterns);
    } else {
      // TD impl.
      populateWithGenerated(patterns);
    }

    mlir::GreedyRewriteConfig config;
    config.useTopDownTraversal = true;

    (void)mlir::applyPatternsAndFoldGreedily(
        getOperation(), std::move(patterns), config);
  }
};

int main(int argc, char** argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "...");

  mlir::MLIRContext context;
  mlir::registerAllDialects(context);

  // TODO(wilber): Why must load dialect to use pdll?
  context.loadAllAvailableDialects();

  context.allowsUnregisteredDialects();

  mlir::OwningOpRef<mlir::ModuleOp> module = LoadMLIR(context, inputFilename);
  llvm::outs() << "src mod\n";
  module->dump();

  mlir::PassManager pm(&context);
  pm.enableTiming();
  auto& opm = pm.nest<mlir::func::FuncOp>();
  opm.addPass(std::make_unique<FusionPass>());

  (void)pm.run(module.get());

  llvm::outs() << "\ndst mod\n";
  module->dump();

  return 0;
}
