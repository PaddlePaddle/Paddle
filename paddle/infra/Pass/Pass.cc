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
#include <memory>
#include "Pass/PassDetail.h"
#include "Pass/PassManager.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/LogicalResult.h"

namespace infra {

namespace detail {

void AdaptorPass::Run(mlir::Operation* op, int opt_level, bool verify) {
  RunImpl(op, opt_level, verify);
}

void AdaptorPass::RunImpl(mlir::Operation* op, int opt_level, bool verify) {
  for (mlir::Region& region : op->getRegions()) {
    for (mlir::Block& block : region.getBlocks()) {
      for (mlir::Operation& inner_op : block.getOperations()) {
        (void)RunPipeline(*mgr, &inner_op, opt_level, verify);
      }
    }
  }
}

mlir::LogicalResult AdaptorPass::RunPipeline(PassManager& pm,
                                             mlir::Operation* op,
                                             int opt_level,
                                             bool verify) {
  for (Pass& pass : pm.GetPasses()) {
    // llvm::outs() << "run Pass: " << pass.info_.name << " on " <<
    // op->getName().getStringRef() << ", can schedule on: " <<
    // pass.CanScheduleOn(op) <<"\n";
    if (pass.CanScheduleOn(op)) {
      if (mlir::failed(RunAPass(&pass, op, opt_level, verify)))
        return mlir::failure();
    }
  }

  return mlir::success();
}

mlir::LogicalResult AdaptorPass::RunAPass(Pass* pass,
                                          mlir::Operation* op,
                                          int opt_level,
                                          bool verify) {
  if (opt_level < pass->info_.opt_level) return mlir::success();

  if (auto* adaptor = dynamic_cast<AdaptorPass*>(pass)) {
    adaptor->Run(op, opt_level, verify);
  } else {
    pass->Run(op);
  }

  if (verify) {
    bool verify_recursively = !dynamic_cast<AdaptorPass*>(pass);
    (void)mlir::verify(op, verify_recursively);
  }

  return mlir::success();
}

}  // namespace detail

mlir::LogicalResult PassManager::Run(mlir::Operation* op) {
  // TODO(wilber): Has some problem in reinit.
  llvm::hash_code new_init_key = context_->getRegistryHash();
  if (new_init_key != init_key_) {
    (void)FinalizePassList();

    if (mlir::failed(Initialize(context_))) return mlir::failure();
    init_key_ = new_init_key;
  }

  bool crash_recovery = false;
  return crash_recovery ? runWithCrashRecovery(op) : runPasses(op);
}

mlir::LogicalResult PassManager::FinalizePassList() {
  auto pass = std::make_unique<infra::detail::AdaptorPass>(this);
  this->addPass(std::move(pass));
  return mlir::success();
}

mlir::LogicalResult PassManager::runPasses(mlir::Operation* op) {
  return detail::AdaptorPass::RunPipeline(*this, op, opt_level_, verify_);
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

}  // namespace infra
