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
#include "paddle/infrt/kernel/tensor_kernels.h"
#include "paddle/infrt/kernel/tensor_shape_kernels.h"
#include "paddle/infrt/kernel/test_kernels.h"

static llvm::cl::list<std::string> cl_shared_libs(  // NOLINT
    "shared_libs",
    llvm::cl::desc("Specify shared library with kernels."),
    llvm::cl::ZeroOrMore,
    llvm::cl::MiscFlags::CommaSeparated);

int main(int argc, char** argv) {
  using namespace llvm;   // NOLINT
  using namespace infrt;  // NOLINT
  cl::opt<std::string> input_file("i",
                                  cl::desc("Specify input filename"),
                                  cl::value_desc("input file name"));
  cl::ParseCommandLineOptions(argc, argv);

  mlir::MLIRContext* context = infrt::Global::getMLIRContext();
  auto module = dialect::LoadMlirFile(input_file.c_str(), context);

  host_context::KernelRegistry registry;

  kernel::RegisterBasicKernels(&registry);
  kernel::RegisterTestKernels(&registry);
  kernel::RegisterTensorShapeKernels(&registry);
  kernel::RegisterTensorKernels(&registry);
  kernel::RegisterControlFlowKernels(&registry);

  // load extra shared library
  for (const auto& lib_path : cl_shared_libs) {
    std::string err;
    llvm::sys::DynamicLibrary dynLib =
        llvm::sys::DynamicLibrary::getPermanentLibrary(lib_path.c_str(), &err);
    if (!dynLib.isValid()) {
      llvm::errs() << "Load shared library failed. Error: " << err << "\n";
      return 1;
    }
    if (auto reg_sym = dynLib.SearchForAddressOfSymbol("RegisterKernels")) {
      auto reg_func =
          reinterpret_cast<void (*)(host_context::KernelRegistry*)>(reg_sym);
      reg_func(&registry);
    } else {
      llvm::outs() << "Symbol \"RegisterKernels\" not found in \"" << lib_path
                   << "\". Skip.\n";
    }
  }

  host_context::TestMlir(module.get(), &registry);

  std::cout << std::endl;
  return 0;
}
