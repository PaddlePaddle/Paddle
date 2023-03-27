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
#include <algorithm>
#include <memory>
#include <utility>
#include "Pass/AnalysisManager.h"
#include "Pass/PassDetail.h"
#include "Pass/PassInstrumentation.h"
#include "Pass/PassManager.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/LogicalResult.h"

namespace infra {

namespace detail {

void AdaptorPass::Run(mlir::Operation* op, int opt_level, bool verify) {
  RunImpl(op, opt_level, verify);
}

void AdaptorPass::RunImpl(mlir::Operation* op, int opt_level, bool verify) {
  auto last_am = GetAnalysisManager();

  for (mlir::Region& region : op->getRegions()) {
    for (mlir::Block& block : region.getBlocks()) {
      for (mlir::Operation& inner_op : block.getOperations()) {
        AnalysisManagerHolder am(&inner_op, last_am.GetPassInstrumentor());
        if (mlir::failed(RunPipeline(*mgr, &inner_op, am, opt_level, verify)))
          return SignalPassFailure();
      }
    }
  }
}

mlir::LogicalResult AdaptorPass::RunPipeline(PassManager& pm,
                                             mlir::Operation* op,
                                             AnalysisManager am,
                                             int opt_level,
                                             bool verify) {
  auto* instrumentor = am.GetPassInstrumentor();
  if (instrumentor && op->getNumRegions()) {
    instrumentor->RunBeforePipeline(op);
  }

  for (Pass& pass : pm.GetPasses()) {
    // llvm::outs() << "run Pass: " << pass.info_.name << " on " <<
    // op->getName().getStringRef() << ", can schedule on: " <<
    // pass.CanScheduleOn(op) <<"\n";
    if (pass.CanScheduleOn(op)) {
      if (mlir::failed(RunPass(&pass, op, am, opt_level, verify)))
        return mlir::failure();
    }
  }

  if (instrumentor && op->getNumRegions()) {
    instrumentor->RunAfterPipeline(op);
  }

  // Apply pass manager on all nested ir.
  if (mlir::failed(
          RunPass(pm.adaptor_pass_.get(), op, am, opt_level, verify))) {
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult AdaptorPass::RunPass(Pass* pass,
                                         mlir::Operation* op,
                                         AnalysisManager am,
                                         int opt_level,
                                         bool verify) {
  if (opt_level < pass->info_.opt_level) return mlir::success();

  pass->pass_state_.emplace(op, am);
  PassInstrumentor* instrumentor = am.GetPassInstrumentor();

  if (auto* adaptor = dynamic_cast<AdaptorPass*>(pass)) {
    adaptor->Run(op, opt_level, verify);
    return mlir::failure(pass->pass_state_->pass_failed);
  }

  if (instrumentor && op->getNumRegions())
    instrumentor->RunBeforePass(pass, op);
  pass->Run(op);

  // Invalidate any non preserved analyses.
  am.Invalidate(pass->pass_state_->preserved_analyses);

  // TODO(wilber): failed?

  if (instrumentor && op->getNumRegions()) instrumentor->RunAfterPass(pass, op);

  if (verify) {
    bool verify_recursively = !dynamic_cast<AdaptorPass*>(pass);
    (void)mlir::verify(op, verify_recursively);
  }

  return mlir::success();
}

}  // namespace detail

//===----------------------------------------------------------------------===//
// PassManager
//===----------------------------------------------------------------------===//

PassManager::~PassManager() = default;

PassManager::PassManager(mlir::MLIRContext* context, int opt_level)
    : context_(context), opt_level_(opt_level) {
  auto pass = std::make_unique<infra::detail::AdaptorPass>(this);
  // this->addPass(std::move(pass));
  adaptor_pass_ = std::move(pass);
}

mlir::LogicalResult PassManager::Run(mlir::Operation* op) {
  // TODO(wilber): Has some problem in reinit.
  llvm::hash_code new_init_key = context_->getRegistryHash();
  if (new_init_key != init_key_) {
    if (mlir::failed(Initialize(context_))) return mlir::failure();
    init_key_ = new_init_key;
  }

  // Construct a analysis manager for the pipeline.
  AnalysisManagerHolder am(op, instrumentor_.get());

  bool crash_recovery = false;
  return crash_recovery ? RunWithCrashRecovery(op, am) : RunPasses(op, am);
}
mlir::LogicalResult PassManager::RunWithCrashRecovery(mlir::Operation* op,
                                                      AnalysisManager am) {
  // TODO(wilber): support crash recovery.
  return mlir::failure();
}

mlir::LogicalResult PassManager::RunPasses(mlir::Operation* op,
                                           AnalysisManager am) {
  return detail::AdaptorPass::RunPipeline(*this, op, am, opt_level_, verify_);
}

mlir::LogicalResult PassManager::Initialize(mlir::MLIRContext* context) {
  // Maybe the adaptor initialize will have a different signature in the future.
  for (Pass& pass : GetPasses()) {
    if (mlir::failed(pass.Initialize(context))) return mlir::failure();
    // auto* adaptor = dynamic_cast<detail::AdaptorPass*>(&pass);
    // if (!adaptor) {
    //   if (mlir::failed(pass.Initialize(context)))
    //     return mlir::failure();
    //   continue;
    // }
    // if (mlir::failed(adaptor->Initialize(context)))
    //   return mlir::failure();
  }

  return mlir::success();
}

void PassManager::AddInstrumentation(std::unique_ptr<PassInstrumentation> pi) {
  if (!instrumentor_) instrumentor_ = std::make_unique<PassInstrumentor>();

  instrumentor_->AddInstrumentation(std::move(pi));
}

//===----------------------------------------------------------------------===//
// PassInstrumentation
//===----------------------------------------------------------------------===//

void PassInstrumentation::RunBeforePipeline(mlir::Operation* op) {}

void PassInstrumentation::RunAfterPipeline(mlir::Operation* op) {}

void PassInstrumentation::RunBeforePass(Pass* pass, mlir::Operation* op) {}

void PassInstrumentation::RunAfterPass(Pass* pass, mlir::Operation* op) {}

void PassInstrumentation::RunBeforeAnalysis(const std::string& name,
                                            mlir::TypeID id,
                                            mlir::Operation* op) {}

void PassInstrumentation::RunAfterAnalysis(const std::string& name,
                                           mlir::TypeID id,
                                           mlir::Operation* op) {}

PassInstrumentation::~PassInstrumentation() = default;

//===----------------------------------------------------------------------===//
// PassInstrumentor
//===----------------------------------------------------------------------===//
namespace detail {
struct PassInstrumentorImpl {
  // TODO(wilber): not support multi-thread now.
  std::vector<std::unique_ptr<PassInstrumentation>> instrumentations;
};
}  // namespace detail

PassInstrumentor::PassInstrumentor()
    : impl(new detail::PassInstrumentorImpl()) {}
PassInstrumentor::~PassInstrumentor() = default;

void PassInstrumentor::RunBeforePipeline(mlir::Operation* op) {
  if (op->getNumRegions() == 0) return;
  for (auto& instr : impl->instrumentations) {
    instr->RunBeforePipeline(op);
  }
}

void PassInstrumentor::RunAfterPipeline(mlir::Operation* op) {
  for (auto& instr : llvm::reverse(impl->instrumentations)) {
    instr->RunAfterPipeline(op);
  }
}

void PassInstrumentor::RunBeforePass(Pass* pass, mlir::Operation* op) {
  for (auto& instr : impl->instrumentations) {
    instr->RunBeforePass(pass, op);
  }
}

void PassInstrumentor::RunAfterPass(Pass* pass, mlir::Operation* op) {
  for (auto& instr : llvm::reverse(impl->instrumentations)) {
    instr->RunAfterPass(pass, op);
  }
}

void PassInstrumentor::RunBeforeAnalysis(const std::string& name,
                                         mlir::TypeID id,
                                         mlir::Operation* op) {
  for (auto& instr : impl->instrumentations) {
    instr->RunBeforeAnalysis(name, id, op);
  }
}

void PassInstrumentor::RunAfterAnalysis(const std::string& name,
                                        mlir::TypeID id,
                                        mlir::Operation* op) {
  for (auto& instr : llvm::reverse(impl->instrumentations)) {
    instr->RunBeforeAnalysis(name, id, op);
  }
}

void PassInstrumentor::AddInstrumentation(
    std::unique_ptr<PassInstrumentation> pi) {
  impl->instrumentations.emplace_back(std::move(pi));
}

}  // namespace infra
