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
#include "paddle/infrt/dialect/mlir_loader.h"
#include "paddle/infrt/dialect/tensorrt/trt_graph_fuse_pass.h"
#include "paddle/infrt/dialect/tensorrt/trt_graph_split_pass.h"
#include "paddle/infrt/dialect/tensorrt/trt_op_converter_pass.h"
#include "paddle/infrt/dialect/tensorrt/trt_op_teller_pass.h"
#include "paddle/infrt/dialect/tensorrt/trt_type_convert_pass.h"

#include "paddle/infrt/host_context/core_runtime.h"
#include "paddle/infrt/host_context/kernel_registry.h"
#include "paddle/infrt/host_context/mlir_to_runtime_translate.h"

#include "paddle/infrt/kernel/basic_kernels.h"
#include "paddle/infrt/kernel/control_flow_kernels.h"
#include "paddle/infrt/kernel/tensor_kernels.h"
#include "paddle/infrt/kernel/tensor_shape_kernels.h"
#include "paddle/infrt/kernel/test_kernels.h"

#include "paddle/infrt/kernel/tensorrt/registry.h"

#ifdef INFRT_WITH_PHI
#include "paddle/infrt/dialect/infrt/pass/infrt_op_fuse_pass.h"
#include "paddle/infrt/dialect/phi/pass/phi_op_convert_pass.h"
#include "paddle/infrt/kernel/phi/infershaped/infershaped_kernel_launchers.h"
#include "paddle/infrt/kernel/phi/registry.h"
#endif

int main(int argc, char** argv) {
  static llvm::cl::opt<std::string> input_file(
      llvm::cl::Positional,
      llvm::cl::desc("Specify input filename"),
      llvm::cl::init("-"));

  llvm::cl::ParseCommandLineOptions(argc, argv);

  mlir::MLIRContext* context = infrt::Global::getMLIRContext();
  auto module = infrt::dialect::LoadMlirFile(input_file.c_str(), context);

  infrt::host_context::KernelRegistry registry;

  ::infrt::kernel::RegisterBasicKernels(&registry);
  ::infrt::kernel::RegisterTestKernels(&registry);
  ::infrt::kernel::RegisterTensorShapeKernels(&registry);
  ::infrt::kernel::RegisterTensorKernels(&registry);
  ::infrt::kernel::RegisterControlFlowKernels(&registry);
#ifdef INFRT_WITH_PHI
  ::infrt::kernel::RegisterPhiKernels(&registry);
  ::infrt::kernel::RegisterInferShapeLaunchers(&registry);
#endif
#if defined(INFRT_WITH_GPU) && defined(INFRT_WITH_TRT)
  ::infrt::kernel::RegisterTrtKernels(&registry);
#endif

  context->loadAllAvailableDialects();
  module->dump();
  mlir::PassManager pm(context);

  mlir::OpPassManager& trt_pass_manager = pm.nest<mlir::FuncOp>();
  trt_pass_manager.addPass(std::make_unique<infrt::trt::TRTOpTellerPass>());
  trt_pass_manager.addPass(std::make_unique<infrt::trt::TRTGraphFusePass>());
  trt_pass_manager.addPass(std::make_unique<infrt::trt::TRTGraphSplitPass>(1));
  trt_pass_manager.addPass(std::make_unique<infrt::trt::TRTOpConverterPass>());
  trt_pass_manager.addPass(infrt::trt::createTrtTypeConvertPass());
  if (mlir::failed(pm.run(*module))) {
    std::cout << "\npass failed!\n" << std::endl;
    return 4;
  }
  module->dump();
  ::infrt::host_context::TestMlir(module.get(), &registry);
  return 0;
}
