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
#include <llvm/Support/CommandLine.h>
#include <mlir/Pass/PassManager.h>
#include <iostream>
#include <string>
#include "paddle/infrt/common/global.h"
#include "paddle/infrt/dialect/infrt/pass/infrt_op_fuse_pass.h"
#include "paddle/infrt/dialect/mlir_loader.h"
#include "paddle/infrt/dialect/phi/pass/phi_op_convert_pass.h"

int main(int argc, char** argv) {
  static llvm::cl::opt<std::string> input_file(
      llvm::cl::Positional,
      llvm::cl::desc("Specify input filename"),
      llvm::cl::init("-"));

  llvm::cl::ParseCommandLineOptions(argc, argv);

  mlir::MLIRContext* context = infrt::Global::getMLIRContext();
  auto module = infrt::dialect::LoadMlirFile(input_file.c_str(), context);
  context->loadAllAvailableDialects();
  module->dump();
  mlir::PassManager pm(context);

  mlir::OpPassManager& phi_pass_manager = pm.nest<mlir::FuncOp>();
  std::vector<infrt::Place> valid_places = {{infrt::TargetType::CPU,
                                             infrt::PrecisionType::FLOAT32,
                                             infrt::LayoutType::NCHW}};
  phi_pass_manager.addPass(infrt::createPhiOpCvtPass(valid_places));
  phi_pass_manager.addPass(infrt::createInfrtOpFusePass());
  if (mlir::failed(pm.run(*module))) {
    std::cout << "\npass failed!\n" << std::endl;
    return 4;
  }
  module->dump();
  return 0;
}
