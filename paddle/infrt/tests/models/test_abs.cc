// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <gtest/gtest.h>
#include <llvm/Support/CommandLine.h>
#include <mlir/Pass/PassManager.h>
#include <iostream>
#include <string>

#include "llvm/Support/DynamicLibrary.h"
#include "paddle/infrt/common/global.h"
#include "paddle/infrt/dialect/mlir_loader.h"
#include "paddle/infrt/host_context/core_runtime.h"
#include "paddle/infrt/host_context/kernel_registry.h"
#include "paddle/infrt/host_context/mlir_to_runtime_translate.h"
#include "paddle/infrt/kernel/basic_kernels.h"
#include "paddle/infrt/kernel/control_flow_kernels.h"
#include "paddle/infrt/kernel/phi/infershaped/infershaped_kernel_launchers.h"
#include "paddle/infrt/kernel/phi/registry.h"
#include "paddle/infrt/kernel/tensor_kernels.h"
#include "paddle/infrt/kernel/tensor_shape_kernels.h"
#include "paddle/infrt/kernel/test_kernels.h"

#include "paddle/infrt/kernel/phi/infershaped/infershaped_utils.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/meta_tensor.h"

#include "paddle/infrt/dialect/infrt/ir/basic_kernels.h"
#include "paddle/infrt/dialect/infrt/ir/infrt_dialect.h"

#include "paddle/infrt/dialect/infrt/pass/infrt_op_fuse_pass.h"
#include "paddle/infrt/dialect/phi/pass/phi_op_convert_pass.h"
#include "paddle/infrt/host_context/paddle_mlir.h"

#include "paddle/infrt/dialect/dense_tensor.h"
#include "paddle/infrt/dialect/phi/ir/infrt_phi_tensor.h"
#include "paddle/infrt/dialect/phi/ir/phi_base.h"
#include "paddle/infrt/dialect/phi/ir/phi_kernels.h"

static llvm::cl::list<std::string> cl_shared_libs(  // NOLINT
    "shared_libs",
    llvm::cl::desc("Specify shared library with kernels."),
    llvm::cl::ZeroOrMore,
    llvm::cl::MiscFlags::CommaSeparated);

TEST(ABS_MODEL, convert_and_execute) {
  std::string model_file_name = "./abs.pdmodel";
  std::string params_file_name = "./abs.pdiparams";
  // convert model
  MLIRModelGenImpl myGen;
  auto module_ = myGen.ImportPaddleModel(model_file_name, params_file_name);
  module_.dump();
  // pick kernel
  mlir::MLIRContext* context = infrt::Global::getMLIRContext();
  context->allowUnregisteredDialects();
  context->getOrLoadDialect<mlir::StandardOpsDialect>();

  context->getOrLoadDialect<infrt::InfrtDialect>();
  context->getOrLoadDialect<infrt::ts::TensorShapeDialect>();
  context->getOrLoadDialect<infrt::InfrtDialect>();
  context->getOrLoadDialect<infrt::dt::DTDialect>();
  context->getOrLoadDialect<infrt::pd::PaddleDialect>();

  context->getOrLoadDialect<infrt::phi::PHIDenseTensorDialect>();
  context->getOrLoadDialect<infrt::phi::PHICPUKernelDialect>();
  context->getOrLoadDialect<infrt::phi::PHIGPUKernelDialect>();
  context->getOrLoadDialect<infrt::phi::PHIDialect>();

  context->loadAllAvailableDialects();
  mlir::PassManager pm(context);

  mlir::OpPassManager& phi_pass_manager = pm.nest<mlir::FuncOp>();
  std::vector<infrt::Place> valid_places = {{infrt::TargetType::CPU,
                                             infrt::PrecisionType::FLOAT32,
                                             infrt::LayoutType::NCHW}};
  phi_pass_manager.addPass(infrt::createPhiOpCvtPass(valid_places));
  phi_pass_manager.addPass(infrt::createInfrtOpFusePass());

  if (mlir::failed(pm.run(module_))) {
    std::cout << "\npass failed!\n" << std::endl;
  }
  module_.dump();

  // executate
  infrt::host_context::KernelRegistry registry;
  infrt::kernel::RegisterBasicKernels(&registry);
  infrt::kernel::RegisterTestKernels(&registry);
  infrt::kernel::RegisterTensorShapeKernels(&registry);
  infrt::kernel::RegisterTensorKernels(&registry);
  infrt::kernel::RegisterControlFlowKernels(&registry);
  infrt::kernel::RegisterPhiKernels(&registry);
  infrt::kernel::RegisterInferShapeLaunchers(&registry);
  // load extra shared library
  for (const auto& lib_path : cl_shared_libs) {
    std::string err;
    llvm::sys::DynamicLibrary dynLib =
        llvm::sys::DynamicLibrary::getPermanentLibrary(lib_path.c_str(), &err);
    if (!dynLib.isValid()) {
      llvm::errs() << "Load shared library failed. Error: " << err << "\n";
      break;
    }
    if (auto reg_sym = dynLib.SearchForAddressOfSymbol("RegisterKernels")) {
      auto reg_func =
          reinterpret_cast<void (*)(infrt::host_context::KernelRegistry*)>(
              reg_sym);
      reg_func(&registry);
    } else {
      llvm::outs() << "Symbol \"RegisterKernels\" not found in \"" << lib_path
                   << "\". Skip.\n";
    }
  }
  infrt::host_context::TestMlir(module_, &registry);
}
