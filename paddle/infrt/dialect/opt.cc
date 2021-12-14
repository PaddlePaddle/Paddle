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

#include <glog/logging.h>
#include <llvm/Support/CommandLine.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/Dialect.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/FileUtilities.h>
#include <mlir/Support/MlirOptMain.h>
#include <mlir/Transforms/Passes.h>

#include <iostream>

#include "paddle/infrt/common/global.h"
#include "paddle/infrt/dialect/init_infrt_dialects.h"
#include "paddle/infrt/dialect/mlir_loader.h"

int main(int argc, char **argv) {
  mlir::MLIRContext *context = infrt::Global::getMLIRContext();

  auto &registry = context->getDialectRegistry();
  infrt::RegisterCinnDialects(registry);

  mlir::registerCanonicalizerPass();

  return mlir::failed(
      mlir::MlirOptMain(argc, argv, "INFRT mlir pass driver", registry));
}
