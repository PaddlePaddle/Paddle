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
#include "IR/PatternMatch.h"
#include "Pass/Pass.h"
#include "Pass/PassManager.h"
#include "Pass/PassRegistry.h"
#include "Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"

#include "mlir/Support/LogicalResult.h"
#include "test/helper.h"

static llvm::cl::opt<std::string> inputFilename(
    llvm::cl::Positional,
    llvm::cl::desc("<input file>"),
    llvm::cl::init("-"),
    llvm::cl::value_desc("filename"));

static llvm::cl::opt<int> opt_level("opt",
                                    llvm::cl::init(2),
                                    llvm::cl::desc("opt_level"));

struct CountOpAnalysis {
  explicit CountOpAnalysis(mlir::Operation* container_op) {
    llvm::outs() << "In CountOpAnalysis, op is "
                 << container_op->getName().getStringRef() << "\n";
    for (auto& region : container_op->getRegions()) {
      for (auto& block : region.getBlocks()) {
        for (auto& op : block.getOperations()) {
          ++count;
        }
      }
    }

    llvm::outs() << "-- count is " << count << "\n";
  }

  int count = 0;
};

class TestPass : public infra::Pass {
 public:
  TestPass() : infra::Pass("TestPass", 1) {}
  void Run(mlir::Operation* op) override {
    llvm::outs() << "In TestPass: " << op->getName() << "\n";

    GetAnalysisManager().GetAnalysis<CountOpAnalysis>();
    this->pass_state_->preserved_analyses.Preserve<CountOpAnalysis>();

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

class TestPattern : public infra::RewritePattern {
 public:
  explicit TestPattern(mlir::MLIRContext* ctx)
      : infra::RewritePattern("testPattern", 1U, ctx) {}
  // void Initialize() override {}
  mlir::LogicalResult MatchAndRewrite(
      mlir::Operation* op, infra::PatternRewriter& rewriter) const override {
    return mlir::success();
  }
};

class AddPattern : public infra::OpRewritePattern<mlir::tosa::AddOp> {
 public:
  using infra::OpRewritePattern<mlir::tosa::AddOp>::OpRewritePattern;

  mlir::LogicalResult MatchAndRewrite(
      mlir::tosa::AddOp op,
      infra::PatternRewriter& rewriter) const final {  // NOLINT
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
    c1.getType();

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

    rewriter.ReplaceOp(op, new_add_op->getResults());
    return mlir::success();
  }
};

class TestPatternDriver : public infra::Pass {
 public:
  TestPatternDriver() : infra::Pass("TestPatternDriver", 1) {}

  void Run(mlir::Operation* op) override {
    infra::RewritePatternSet patterns(op->getContext());
    patterns.Add<TestPattern, AddPattern>(op->getContext());

    auto an = GetAnalysisManager().GetCachedAnalysis<CountOpAnalysis>();
    if (an) {
      llvm::outs() << "In TestPatternDriverPass, last pass analysis "
                   << an->get().count << " ops.\n";
    }
    this->pass_state_->preserved_analyses.Unpreserve<CountOpAnalysis>();

    infra::GreedyRewriteConfig config;
    config.use_top_down_traversal = true;

    (void)infra::ApplyPatternsGreedily(op, std::move(patterns), config);
  }

  bool CanScheduleOn(mlir::Operation* op) const override {
    return op->getNumRegions() > 0 &&
           op->getName().getStringRef() != "builtin.module";
  }
};

void RegisterPass() {
  infra::PassRegistration<TestPass>();
  infra::PassRegistration<TestPatternDriver>();
}

int main(int argc, char** argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "...");

  mlir::MLIRContext context;
  mlir::registerAllDialects(context);
  context.allowsUnregisteredDialects();

  mlir::OwningOpRef<mlir::ModuleOp> module = LoadMLIR(context, inputFilename);
  llvm::outs() << "src mod\n";
  module->dump();

  RegisterPass();

  infra::PassManager pm(&context, opt_level);
  pm.EnableTiming();
  pm.EnableIRPrinting(std::make_unique<infra::PassManager::IRPrinterConfig>(
      [](infra::Pass* pass, mlir::Operation* op) {
        return pass->GetPassInfo().name == "TestPass";
      },
      [](infra::Pass* pass, mlir::Operation* op) {
        return pass->GetPassInfo().name == "TestPass";
      },
      true,
      false));
  auto pass = std::make_unique<TestPass>();
  pm.addPass(std::move(pass));

  auto pass2 = std::make_unique<TestPatternDriver>();
  pm.addPass(std::move(pass2));

  (void)pm.Run(module.get());

  llvm::outs() << "\ndst mod\n";
  module->dump();
  return 0;
}
